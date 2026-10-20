"""Steps 3-4: segments (case, phase k) and their traces (D6, D7, D9).

Day-level dates are int days since 1950-01-01 (`DAY0`).
* Segment k exists for every windowed case that entered phase k with an entry date
  (after D1-D4). Label (D6): phase 1/2 -> enters phase k+1; phase 3 -> completes_phase_3.
* Trace (D9, D7): every event of the case with date < exit_date, sorted by (date, row).
  in_phase = date >= entry_date; n_hist = # trace events before entry.
  Segments without any in-phase event strictly before the exit are dropped and counted.
* remaining_in_phase = exit_date - event date (days, >= 1).
"""
import numpy as np
import pandas as pd

DAY0 = pd.Timestamp("1950-01-01")
B = 1_000_000  # composite-key base: key = group * B + day (days < B)


def to_day(s):
    s = pd.to_datetime(s)
    out = (s - DAY0).dt.days
    return out.astype("float64").values


def build_segments(win, report=None, log=print):
    report = report if report is not None else {}
    rows = []
    p1x, p1e = to_day(win.phase1_exit_date), to_day(win.phase1_entry_date)
    p2x, p2e = to_day(win.phase2_exit_date), to_day(win.phase2_entry_date)
    for k in (1, 2, 3):
        if k == 1:
            m = np.ones(len(win), bool)
            lab = win.moved_on_1.values
            prev1 = np.zeros(len(win)); prev2 = np.zeros(len(win))
        elif k == 2:
            m = win.phase2_entered.values.astype(bool) & win.phase2_entry_date.notna().values
            lab = win.moved_on_2.values
            prev1 = p1x - p1e; prev2 = np.zeros(len(win))
        else:
            m = win.phase3_entered.values.astype(bool) & win.phase3_entry_date.notna().values
            lab = win.completes_3.values
            prev1 = p1x - p1e
            prev2 = np.where(np.isnan(p2e), 0.0, p2x - p2e)  # D2: zero-length phase 2
        s = pd.DataFrame({
            "ucid": win.ucid.values[m], "phase": k,
            "entry": to_day(win[f"phase{k}_entry_date"])[m],
            "exit": to_day(win[f"phase{k}_exit_date"])[m],
            "label01": lab[m].astype(int),
            "PP_prev_phase_days_1": prev1[m], "PP_prev_phase_days_2": prev2[m],
            "filing_year": win.filing_date.dt.year.values[m],
        })
        rows.append(s)
    seg = pd.concat(rows, ignore_index=True)
    bad = seg.exit.isna() | seg.entry.isna() | (seg.exit < seg.entry)
    report.setdefault("segments", {})["invalid_dates_dropped"] = (
        seg[bad].groupby("phase").size().reindex([1, 2, 3], fill_value=0).astype(int).to_dict())
    seg = seg[~bad].reset_index(drop=True)
    seg["segment_id"] = seg.ucid + "|p" + seg.phase.astype(str)
    report["segments"]["built"] = seg.groupby("phase").size().astype(int).to_dict()
    log(f"[segments] {report['segments']}")
    return seg


def sort_events(ev, case_index):
    """Sort events by (case, date, original row); add ci (case code) and d (day int)."""
    ev = ev.copy()
    ev["ci"] = case_index.get_indexer(ev.ucid)
    assert (ev.ci >= 0).all()
    ev["d"] = to_day(ev.date_filed).astype(np.int64)
    ev = ev.sort_values(["ci", "d", "row"], kind="stable").reset_index(drop=True)
    return ev


def build_traces(seg, ev, case_index, report=None, phase=None, log=print):
    """Return (seg_kept, ev_idx, seg_idx): trace rows as indices into the sorted events `ev`.

    `ev` must come from sort_events with the same case_index.
    """
    report = report if report is not None else {}
    key = ev.ci.values.astype(np.int64) * B + ev.d.values
    sci = case_index.get_indexer(seg.ucid).astype(np.int64)
    start = np.searchsorted(key, sci * B, side="left")
    stop = np.searchsorted(key, sci * B + seg.exit.values.astype(np.int64), side="left")   # date < exit
    entry_pos = np.searchsorted(key, sci * B + seg.entry.values.astype(np.int64), side="left")
    n_hist = np.minimum(entry_pos, stop) - start
    n_in = stop - np.maximum(entry_pos, start)
    keep = n_in > 0
    r = report.setdefault("traces", {})
    r[f"p{phase}_empty_dropped"] = int((~keep).sum())
    r[f"p{phase}_empty_dropped_by_label"] = {int(k): int(v) for k, v in
                                             seg.loc[~keep, "label01"].value_counts().items()}
    seg = seg[keep].reset_index(drop=True).copy()
    start, stop, n_hist, n_in = start[keep], stop[keep], n_hist[keep], n_in[keep]
    seg["n_hist"] = n_hist
    seg["n_in_phase"] = n_in
    lens = stop - start
    seg_idx = np.repeat(np.arange(len(seg)), lens)
    offs = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    ev_idx = np.repeat(start, lens) + offs
    log(f"[traces] phase {phase}: segments {len(seg):,} (empty dropped {r[f'p{phase}_empty_dropped']}), rows {len(ev_idx):,}")
    return seg, ev_idx, seg_idx
