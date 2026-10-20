"""Step 2: decisions D1-D4 (phase-table fixes) and D5 (study window) on the case table.

All rules work on the case table only. Every count is written into `report`.
* D1: phase2_entered & no phase2_entry_date & not phase3_entered (summary judgment, no answer)
      -> phase2_entered := False (closed in phase 1 by a court decision; T1 label 0; no phase-2 segment).
* D2: phase2_entered & no phase2_entry_date & phase3_entered (trial with no answer)
      -> moved on (T1 label 1); phase 2 has zero length (no phase-2 segment); phase-3 segment kept.
* D3: phase2_exit_date < phase2_entry_date -> drop the whole case.
* D4: phase1_completed & ~phase2_entered (answered on the termination day) -> follow the phase
      table: phase2_entered stays False (T1 label 0). Counted only.
* D5: keep cases filed 2003-01-01 .. 2016-12-31.
"""
import pandas as pd

WINDOW = ("2003-01-01", "2016-12-31")


def _phase_counts(c):
    return {f"p{k}": {"entered": int(c[f"phase{k}_entered"].sum()),
                      "completed": int(c[f"phase{k}_completed"].sum())} for k in (1, 2, 3)}


def flag_cases(c):
    """Boolean flags for D1-D4 on the raw case table (no change)."""
    no_p2_date = c.phase2_entered.astype(bool) & c.phase2_entry_date.isna()
    return pd.DataFrame({
        "D1": no_p2_date & ~c.phase3_entered.astype(bool),
        "D2": no_p2_date & c.phase3_entered.astype(bool),
        "D3": c.phase2_exit_date < c.phase2_entry_date,
        "D4": c.phase1_completed.astype(bool) & ~c.phase2_entered.astype(bool),
    }, index=c.index)


def apply_fixes(c, report=None, window=WINDOW, log=print):
    """Return the fixed case table (all cases, D1-D4 applied, D3 dropped) and the windowed one."""
    report = report if report is not None else {}
    c = c.copy()
    f = flag_cases(c)
    rep = {"all_cases": len(c), "flags_all": {k: int(v.sum()) for k, v in f.items()}}
    in_win = (c.filing_date >= window[0]) & (c.filing_date <= window[1])
    rep["flags_window"] = {k: int((v & in_win).sum()) for k, v in f.items()}
    rep["window_before_fixes"] = {"cases": int(in_win.sum()), **_phase_counts(c[in_win])}
    # anchors of PLAN.md §2: window, without D1/D2/D3 cases, "moved on" = phaseK_completed
    a = c[in_win & ~f.D1 & ~f.D2 & ~f.D3]
    rep["anchor_recompute"] = {"p1_entered": len(a), "p1_completed": int(a.phase1_completed.sum()),
                               "p2_entered": int(a.phase2_entered.sum()), "p2_completed": int(a.phase2_completed.sum()),
                               "p3_entered": int(a.phase3_entered.sum()), "p3_completed": int(a.phase3_completed.sum())}
    # D1: relabel
    c["phase2_entered"] = c.phase2_entered.astype(bool)
    c.loc[f.D1, "phase2_entered"] = False
    c.loc[f.D1, ["phase2_exit_date"]] = pd.NaT
    c["fix_D1"] = f.D1.values
    # D2: moved on, zero-length phase 2 (no phase-2 segment because entry date is missing)
    c["fix_D2"] = f.D2.values
    c["fix_D4"] = f.D4.values
    c.loc[f.D4, ["phase2_entry_date", "phase2_exit_date"]] = pd.NaT  # D4: no phase 2 (follow the phase table)
    # T1 label after fixes: moved on = phase2_entered (D2 stays True, D1 now False, D4 False)
    c["moved_on_1"] = c.phase2_entered.astype(bool)
    c["moved_on_2"] = c.phase3_entered.astype(bool)
    c["completes_3"] = c.phase3_completed.astype(bool)
    # D3: drop the whole case
    fixed_all = c[~f.D3.values].reset_index(drop=True)
    rep["after_D3_all_cases"] = len(fixed_all)
    win = fixed_all[(fixed_all.filing_date >= window[0]) & (fixed_all.filing_date <= window[1])].reset_index(drop=True)
    rep["window_after_fixes"] = {"cases": len(win),
                                 "p1_entered": len(win), "p1_moved_on": int(win.moved_on_1.sum()),
                                 "p2_entered_with_date": int((win.phase2_entered & win.phase2_entry_date.notna()).sum()),
                                 "p2_moved_on": int((win.phase2_entered & win.phase2_entry_date.notna() & win.moved_on_2).sum()),
                                 "p3_entered": int(win.phase3_entered.sum()),
                                 "p3_completed": int((win.phase3_entered & win.completes_3).sum())}
    report["phase_fixes"] = rep
    log(f"[phase_fixes] {rep}")
    return fixed_all, win
