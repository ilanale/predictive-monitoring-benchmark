"""Step 6: Global (inter-case) features at each event date t, from the case table (ALL cases).

Definition of "open at t" (day-level, state at the END of day t):
    a case is open at t  <=>  filing_day <= t < terminating_day
    a case is in phase j at t  <=>  phase_j_entry <= t < phase_j_exit   (intervals with exit <= entry ignored)
so a case filed on t counts as open on t, a case closed on t does not. "Excluding this case"
subtracts this case's own indicator under the same rule (so the result counts other cases only).
Windows: filed/closed in the last w days = date in (t - w, t], this case excluded (RD-06),
so net_flow = filed - closed counts other cases only.
Judge history: median length (terminating - filing, days) of the judge's cases closed STRICTLY
before t (conservative), 0 if none; G_judge_hist_n_closed = their number.
Judge = the case's District_Judge in the case table; judge features are 0 when it is missing.
Everything is vectorized with searchsorted on composite keys group * B + day; running medians
are built once per closure (two heaps per judge), then looked up.
"""
import heapq

import numpy as np
import pandas as pd

from .segments import B, DAY0, to_day

G_COLS = None  # filled at import time below


def _open_count(start_key, end_key, qkey):
    """# intervals with start <= q < end in the same group (keys = group*B + day)."""
    S = np.sort(start_key); E = np.sort(end_key)
    return np.searchsorted(S, qkey, side="right") - np.searchsorted(E, qkey, side="right")


def _window_count(time_key, qkey, w):
    T = np.sort(time_key)
    return np.searchsorted(T, qkey, side="right") - np.searchsorted(T, qkey - w, side="right")


def case_arrays(cases):
    """Day arrays and group codes for the case table (all cases)."""
    A = {"f": to_day(cases.filing_date), "e": to_day(cases.terminating_date)}
    for k in (1, 2, 3):
        s, x = to_day(cases[f"phase{k}_entry_date"]), to_day(cases[f"phase{k}_exit_date"])
        ok = ~np.isnan(s) & ~np.isnan(x) & (x > s)
        A[f"s{k}"] = np.where(ok, s, np.nan); A[f"x{k}"] = np.where(ok, x, np.nan)
    judge = cases.District_Judge.fillna("__none__").astype(str)
    mag = cases.Magistrate_Judge.fillna("__none__").astype(str)
    A["judge_codes"], A["judge_levels"] = pd.factorize(judge)
    A["judge_missing"] = cases.District_Judge.isna().values
    A["mag_codes"], _ = pd.factorize(mag)
    A["mag_missing"] = cases.Magistrate_Judge.isna().values
    A["type_codes"], _ = pd.factorize(cases.case_type.fillna("__none__").astype(str))
    return A


def _running_medians(gcode, end, los):
    """Sorted keys (g*B+end) and the median of los over each group's closures up to that one."""
    order = np.lexsort((end, gcode))
    keys = gcode[order].astype(np.int64) * B + end[order].astype(np.int64)
    lo, hi, med = [], [], np.empty(len(order))
    g_prev = None
    for i, idx in enumerate(order):
        g = gcode[idx]
        if g != g_prev:
            lo, hi, g_prev = [], [], g
        v = los[idx]
        if lo and v > -lo[0]:
            heapq.heappush(hi, v)
        else:
            heapq.heappush(lo, -v)
        if len(lo) > len(hi) + 1:
            heapq.heappush(hi, -heapq.heappop(lo))
        elif len(hi) > len(lo):
            heapq.heappush(lo, -heapq.heappop(hi))
        med[i] = -lo[0] if len(lo) > len(hi) else (-lo[0] + hi[0]) / 2.0
    return keys, med


def global_features(cases, q_case, q_day, q_phase=None):
    """cases: full case table (fixed); q_case: row positions into `cases`; q_day: event day ints.

    q_phase (optional): the segment phase per query, for G_judge_open_same_phase.
    """
    A = case_arrays(cases)
    t = q_day.astype(np.int64)
    qc = q_case
    f, e = A["f"], A["e"]
    out = {}
    zero = np.zeros(len(cases), np.int64)

    def self_open(s, x):
        ss, xx = s[qc], x[qc]
        return ((ss <= t) & (t < xx)).astype(np.int64)  # NaN compares False

    def open_in(group, s, x):
        ok = ~np.isnan(s)
        sk = group[ok].astype(np.int64) * B + s[ok].astype(np.int64)
        ek = group[ok].astype(np.int64) * B + x[ok].astype(np.int64)
        return _open_count(sk, ek, group[qc].astype(np.int64) * B + t) - self_open(s, x)

    out["open_cases"] = open_in(zero, f, e)
    tg = A["type_codes"]
    out["G_court_open_same_type"] = open_in(tg, f, e)
    for k in (1, 2, 3):
        out[f"G_court_open_phase{k}"] = open_in(zero, A[f"s{k}"], A[f"x{k}"])
    def self_in_window(x, w):
        xx = x[qc]
        return ((t - w < xx) & (xx <= t)).astype(np.int64)

    for w in (30, 90):
        out[f"G_court_filed_last{w}"] = _window_count(f.astype(np.int64), t, w) - self_in_window(f, w)
        out[f"G_court_closed_last{w}"] = _window_count(e.astype(np.int64), t, w) - self_in_window(e, w)
        out[f"G_court_net_flow{w}"] = out[f"G_court_filed_last{w}"] - out[f"G_court_closed_last{w}"]
    jg = A["judge_codes"].astype(np.int64)
    jmiss = A["judge_missing"][qc]
    ntype = int(tg.max()) + 1
    out["G_judge_open"] = open_in(jg, f, e)
    out["G_judge_open_same_type"] = open_in(jg * ntype + tg, f, e)
    for k in (1, 2, 3):
        out[f"G_judge_open_phase{k}"] = open_in(jg, A[f"s{k}"], A[f"x{k}"])
    if q_phase is not None:
        out["G_judge_open_same_phase"] = np.choose(
            np.asarray(q_phase) - 1, [out["G_judge_open_phase1"], out["G_judge_open_phase2"], out["G_judge_open_phase3"]])
    qk = jg[qc] * B + t
    for w in (30, 90):
        out[f"G_judge_filed_last{w}"] = _window_count(jg * B + f.astype(np.int64), qk, w) - self_in_window(f, w)
        out[f"G_judge_closed_last{w}"] = _window_count(jg * B + e.astype(np.int64), qk, w) - self_in_window(e, w)
        out[f"G_judge_net_flow{w}"] = out[f"G_judge_filed_last{w}"] - out[f"G_judge_closed_last{w}"]
    out["G_judge_share_of_court"] = out["G_judge_open"] / np.maximum(out["open_cases"], 1)
    mg = A["mag_codes"].astype(np.int64)
    out["G_mag_open"] = np.where(A["mag_missing"][qc], 0, open_in(mg, f, e))
    # judge history: running median of case length over closures strictly before t
    keys, med = _running_medians(jg, e, e - f)
    hi = np.searchsorted(keys, qk, side="left")          # closures with end < t (and earlier groups)
    lo = np.searchsorted(keys, jg[qc] * B, side="left")  # first closure of this judge
    n_closed = hi - lo
    out["G_judge_hist_median_los"] = np.where(n_closed > 0, med[np.clip(hi - 1, 0, None)], 0.0)
    out["G_judge_hist_n_closed"] = n_closed
    for c in list(out):
        if c.startswith("G_judge") and c != "G_judge_share_of_court":
            out[c] = np.where(jmiss, 0, out[c])
    out["G_judge_share_of_court"] = np.where(jmiss, 0.0, out["G_judge_share_of_court"])
    # seasonality
    dates = DAY0 + pd.to_timedelta(t, unit="D")
    m = dates.month.values
    out["G_month_sin"] = np.sin(2 * np.pi * m / 12)
    out["G_month_cos"] = np.cos(2 * np.pi * m / 12)
    out["G_quarter"] = dates.quarter.values
    ye = pd.to_datetime(pd.DataFrame({"year": dates.year, "month": 12, "day": 31})).values
    out["G_days_to_year_end"] = ((ye - dates.values) // np.timedelta64(1, "D")).astype(np.int64)
    out["G_is_december"] = (m == 12).astype(np.int8)
    df = pd.DataFrame(out)
    for c in df.columns:
        if df[c].dtype == np.int64:
            df[c] = df[c].astype(np.int32)
    return df
