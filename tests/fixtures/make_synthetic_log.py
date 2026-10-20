"""Deterministic synthetic event log in the upstream (Teinemaa et al. 2019) processed-log format.

Used only to validate the compat port (no real benchmark logs available). 500 cases, seed 22,
trace lengths 2-30, case label "deviant"/"regular" that depends learnably on static and dynamic
attributes. Time features are computed as in upstream preprocessing/preprocess_logs_*.py.

Usage: python tests/fixtures/make_synthetic_log.py [out_dir]   (default: <repo>/data/synthetic)
Writes <out_dir>/synthetic.csv (sep=";").
"""
import os
import sys

import numpy as np
import pandas as pd

SEED = 22
N_CASES = 500
CASE_ID, ACT, RES, TS, LABEL = "Case ID", "Activity", "Resource", "Complete Timestamp", "label"
ACTIVITIES = ["Review", "Request_Info", "Escalate", "Approve_Step", "Notify", "Pay"]


def make_log(seed=SEED, n_cases=N_CASES):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2020-01-01 08:00:00")
    # case start times spread over ~2 years (in minutes, integer)
    starts = np.sort(rng.integers(0, 2 * 365 * 24 * 60, size=n_cases))
    rows = []
    for i in range(n_cases):
        case_id = "case_%04d" % i
        case_type = rng.choice(["A", "B", "C"], p=[0.4, 0.35, 0.25])
        region = rng.choice(["north", "south", "east", "west"])
        amount = float(np.round(np.exp(rng.normal(8.0, 0.8)), 2))
        age = float(rng.integers(20, 81))
        length = int(rng.integers(2, 31))  # 2..30 events
        risk = 0.9 * (case_type == "C") - 0.5 * (case_type == "A") + 0.6 * (region == "east") \
            + 0.8 * (np.log(amount) - 8.0)
        p_esc = 1.0 / (1.0 + np.exp(-(risk - 1.5)))
        t = int(starts[i])
        n_esc, total_cost, high_prio = 0, 0.0, 0
        events = []
        for j in range(length):
            if j == 0:
                act = "Register"
            else:
                probs = np.array([0.3, 0.2, 0.0, 0.2, 0.15, 0.15])
                probs = probs * (1 - p_esc) / probs.sum()
                probs[2] = p_esc
                act = ACTIVITIES[int(rng.choice(len(ACTIVITIES), p=probs / probs.sum()))]
                t += int(rng.exponential(600 if act != "Escalate" else 2400)) + 1
            resource = "R%d" % int(rng.integers(1, 9))
            priority = "high" if rng.random() < (0.2 + 0.4 * (act == "Escalate")) else "low"
            cost = float(np.round(rng.gamma(2.0, 50.0 + 100.0 * (act == "Escalate")), 2))
            n_esc += act == "Escalate"
            high_prio += priority == "high"
            total_cost += cost
            events.append([case_id, act, resource, base + pd.Timedelta(minutes=t), case_type, region,
                           amount, age, priority, cost])
        score = risk + 1.2 * (n_esc > 0) + 0.4 * high_prio / length + rng.normal(0, 0.7)
        label = "deviant" if score > 1.3 else "regular"
        for ev in events:
            rows.append(ev + [label])
    data = pd.DataFrame(rows, columns=[CASE_ID, ACT, RES, TS, "case_type", "region", "amount", "age",
                                       "priority", "cost", LABEL])

    # time features, as in upstream preprocessing (minutes)
    data["timesincemidnight"] = data[TS].dt.hour * 60 + data[TS].dt.minute
    data["month"] = data[TS].dt.month
    data["weekday"] = data[TS].dt.weekday
    data["hour"] = data[TS].dt.hour
    data = data.sort_values([CASE_ID, TS], kind="mergesort").reset_index(drop=True)
    grp = data.groupby(CASE_ID)[TS]
    data["timesincelastevent"] = ((data[TS] - grp.shift(1)).dt.total_seconds() / 60.0).fillna(0.0)
    data["timesincecasestart"] = (data[TS] - grp.transform("min")).dt.total_seconds() / 60.0
    data["event_nr"] = data.groupby(CASE_ID).cumcount() + 1
    # inter-case feature: number of cases open at event time (start <= t < end), as upstream
    data = data.sort_values([TS], kind="mergesort").reset_index(drop=True)
    first_last = data.groupby(CASE_ID)[TS].agg(["min", "max"])
    st = np.sort(first_last["min"].values)
    en = np.sort(first_last["max"].values)
    tv = data[TS].values
    data["open_cases"] = np.searchsorted(st, tv, side="right") - np.searchsorted(en, tv, side="right")
    data[TS] = data[TS].dt.strftime("%Y-%m-%d %H:%M:%S")
    return data


def main(out_dir=None):
    if out_dir is None:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_dir = os.path.join(repo, "data", "synthetic")
    os.makedirs(out_dir, exist_ok=True)
    data = make_log()
    path = os.path.join(out_dir, "synthetic.csv")
    data.to_csv(path, sep=";", index=False)
    return path


if __name__ == "__main__":
    p = main(sys.argv[1] if len(sys.argv) > 1 else None)
    print(p)
