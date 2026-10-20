"""Step 5: per-event Pre-phase features (view P and P+).

Computed once on the case-level event sequence sorted by (case, date, row). Because a
phase-k trace holds every event from the start of the case (D9), the value at event i of a
trace equals the value at event i of the case sequence, and every feature at event i uses
only events 0..i of the case (no look-ahead; tested in tests/test_data_prep.py).

* time features in days: timesincelastevent, timesincecasestart, event_nr, month, weekday
* P_attr_*: event attribute flags as 0/1 (P_attr_duplicates is the collapsed-duplicates count)
* judge: first judge ID in event_judge ("missing" if none); P_judge_is_magistrate: its role
* PP_*: cumulative self-loops, return loops (A->B->A), repeats, distinct judges seen;
  events in the last 30/90 days (positions <= i with date > d_i - w)
"""
import numpy as np
import pandas as pd

from .segments import B, DAY0

JUDGE_RE = r"\('(SJ\d+)', '([^']*)'\)"


def parse_judges(s):
    """First judge id and magistrate flag from strings like "(('SJ000112', 'Magistrate Judge'),)"."""
    first = s.fillna("").str.extract(JUDGE_RE)
    judge = first[0].fillna("missing")
    mag = (first[1] == "Magistrate Judge").fillna(False).astype(np.int8)
    return judge, mag


def event_features(ev, attr_flags):
    """ev: sorted events with ci, d, Activity, event_judge, attribute_*. Returns a DataFrame aligned to ev."""
    n = len(ev)
    ci = ev.ci.values
    d = ev.d.values.astype(np.int64)
    newcase = np.r_[True, ci[1:] != ci[:-1]]
    pos = np.arange(n)
    case_start = np.maximum.accumulate(np.where(newcase, pos, 0))
    out = {}
    dates = DAY0 + pd.to_timedelta(d, unit="D")
    prev_d = np.r_[d[0], d[:-1]]
    out["timesincelastevent"] = np.where(newcase, 0, d - prev_d).astype(np.int32)
    out["timesincecasestart"] = (d - d[case_start]).astype(np.int32)
    out["event_nr"] = (pos - case_start + 1).astype(np.int32)
    out["month"] = dates.month.values.astype(np.int8)
    out["weekday"] = dates.weekday.values.astype(np.int8)
    out["P_attr_duplicates"] = ev["attribute_duplicates"].fillna(1).values.astype(np.int32)
    for a in attr_flags:
        out["P_" + a.replace("attribute_", "attr_")] = ev[a].fillna(False).values.astype(np.int8)
    judge, mag = parse_judges(ev.event_judge)
    out["judge"] = judge.values
    out["P_judge_is_magistrate"] = mag.values

    def ccum(x):  # per-case inclusive cumulative sum
        cs = np.cumsum(x.astype(np.int64))
        base = (cs - x)[case_start]
        return (cs - base).astype(np.int32)

    act = pd.factorize(ev.Activity.values)[0].astype(np.int64)
    prev1 = np.r_[-1, act[:-1]]
    prev2 = np.r_[-1, -1, act[:-2]]
    same1 = np.r_[False, ci[1:] == ci[:-1]]
    same2 = np.r_[False, False, ci[2:] == ci[:-2]]
    selfloop = same1 & (act == prev1)
    retloop = same2 & (act == prev2) & ~selfloop
    repeat = pd.Series(ci * 1000 + act).duplicated().values  # activity seen earlier in this case
    out["PP_n_self_loops"] = ccum(selfloop)
    out["PP_n_return_loops"] = ccum(retloop)
    out["PP_n_repeats"] = ccum(repeat)
    # distinct judges seen so far (all judges listed on each event)
    allj = ev.event_judge.fillna("").str.extractall(JUDGE_RE)
    first_seen = np.zeros(n, np.int64)
    if len(allj):
        r = allj.index.get_level_values(0).values
        jc = pd.factorize(allj[0].values)[0].astype(np.int64)
        new = ~pd.Series(ci[r] * 100000 + jc).duplicated().values
        np.add.at(first_seen, r[new], 1)
    out["PP_n_distinct_judges"] = ccum(first_seen)
    key = ci.astype(np.int64) * B + d
    for w in (30, 90):
        first_in = np.searchsorted(key, key - w, side="right")   # first position with date > d - w
        out[f"PP_events_last{w}"] = (pos - first_in + 1).astype(np.int32)
    return pd.DataFrame(out)
