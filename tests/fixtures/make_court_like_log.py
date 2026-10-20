"""Small synthetic event log shaped like court/SCHEMA.md (for unit tests only).

    python -m tests.fixtures.make_court_like_log OUT_DIR   # writes court_phase{1,2,3}.csv + court_columns.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ACTS = ["Answer", "Motion", "Order", "Hearing", "Notice"]
JUDGES = ["J01", "J02", "J10", "missing"]


def columns_json():
    return {
        "case_id_col": "segment_id", "activity_col": "Activity", "resource_col": "judge",
        "timestamp_col": "timestamp", "label_col": "label", "regression_label_col": "remaining_in_phase",
        "static_cat": ["M_case_type", "M_city"],
        "static_num": ["M_related", "PP_prev_phase_days_1", "S_counsel_count"],
        "dynamic_cat": ["Activity", "judge"],
        "dynamic_num": ["timesincelastevent", "timesincecasestart", "event_nr", "month", "weekday", "open_cases",
                        "P_attr_a", "P_judge_is_magistrate", "PP_self_loops", "G_court_open", "G_judge_load"],
        "views": {"M": ["M_case_type", "M_city", "M_related"],
                  "P": ["Activity", "judge", "timesincelastevent", "timesincecasestart", "event_nr", "month",
                        "weekday", "P_attr_a", "P_judge_is_magistrate"],
                  "PP": ["PP_self_loops", "PP_prev_phase_days_1"],
                  "B0G": ["open_cases"], "G": ["G_court_open", "G_judge_load"], "S": ["S_counsel_count"]},
        "never_features": ["ucid", "in_phase", "n_hist", "entry_date", "exit_date", "filing_year",
                           "remaining_in_phase", "label"],
        "excluded": {"resolution_type": "D8 outcome column"},
    }


def make_log(n_cases=60, phase=1, seed=0, max_hist=6, max_inphase=14):
    rng = np.random.default_rng(seed + 100 * phase)
    rows = []
    for c in range(n_cases):
        ucid = "C%04d" % c
        seg = "%s|p%d" % (ucid, phase)
        h = int(rng.integers(0, max_hist + 1))
        m = int(rng.integers(1, max_inphase + 1))
        n = h + m
        filing = pd.Timestamp("2005-01-01") + pd.Timedelta(days=int(rng.integers(0, 3000)))
        gaps = rng.choice([0, 0, 1, 3, 7, 15, 40], size=n)
        gaps[0] = 0
        if h > 0:
            gaps[h] = max(gaps[h], 1)  # entry date strictly after the last pre-phase event
        dates = filing + pd.to_timedelta(np.cumsum(gaps), unit="D")
        entry = dates[h]
        exit_ = dates[-1] + pd.Timedelta(days=int(rng.integers(1, 60)))
        label = "deviant" if rng.random() < 0.45 else "regular"
        ctype = rng.choice(["civil", "criminal", "other"])
        city = rng.choice(["Haifa", "Tel Aviv", None])
        related = float(rng.integers(0, 4)) if rng.random() > 0.2 else np.nan
        prev_days = float(rng.integers(0, 500)) if phase > 1 else 0.0
        counsel = float(rng.integers(0, 9))
        loops = 0
        prev_act = None
        for i in range(n):
            act = ACTS[int(rng.integers(0, len(ACTS)))]
            loops += int(act == prev_act); prev_act = act
            judge = JUDGES[int(rng.integers(0, len(JUDGES)))]
            d = dates[i]
            rows.append({
                "segment_id": seg, "ucid": ucid, "Activity": act, "judge": judge,
                "timestamp": d.strftime("%Y-%m-%d"), "label": label,
                "remaining_in_phase": int((exit_ - d).days), "in_phase": int(d >= entry), "n_hist": h,
                "entry_date": entry.strftime("%Y-%m-%d"), "exit_date": exit_.strftime("%Y-%m-%d"),
                "filing_year": filing.year,
                "timesincelastevent": float((d - dates[i - 1]).days) if i else 0.0,
                "timesincecasestart": float((d - dates[0]).days), "event_nr": float(i + 1),
                "month": float(d.month), "weekday": float(d.weekday()),
                "open_cases": float(15000 + rng.integers(-300, 300)),
                "M_case_type": ctype, "M_city": city, "M_related": related,
                "P_attr_a": float(rng.random() < 0.3), "P_judge_is_magistrate": float(judge == "J10"),
                "PP_self_loops": float(loops), "PP_prev_phase_days_1": prev_days,
                "G_court_open": float(rng.integers(100, 9000)) if rng.random() > 0.15 else np.nan,
                "G_judge_load": float(rng.normal(50, 10)),
                "S_counsel_count": counsel, "resolution_type": rng.choice(["verdict", "settled"]),
            })
    return pd.DataFrame(rows)


def write_fixture(out_dir, n_cases=60, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    for k in (1, 2, 3):
        make_log(n_cases, k, seed).to_csv(os.path.join(out_dir, "court_phase%d.csv" % k), sep=";", index=False)
    with open(os.path.join(out_dir, "court_columns.json"), "w") as f:
        json.dump(columns_json(), f, indent=1)
    return out_dir


if __name__ == "__main__":
    write_fixture(sys.argv[1])
