# DECISIONS — data decisions D1–D12

**Status:** all twelve approved by Niv on 2026-10-21, each with its default option (PLAN.md §0.2).
These are fixed for the run. This file holds **only** D1–D12. Any decision taken during the autonomous run goes into `RUN_DECISIONS.md`.

Counts come from the cloud-session checks on the full inputs (PLAN.md §0.3). WS1 re-measures every count and writes it into `data/processed/data_report.md` (before → after).

---

## D1 ⚠️ Summary-judgment cases with no answer
- **Issue:** 1,431 civil cases are flagged "entered phase 2" but have no phase-2 entry date. They were closed by summary judgment without an answer. These are *all* the summary-judgment cases in the data.
- **Decision:** treat them as **closed in phase 1 by a court decision**.
  - Task-1 label = 0 (did not move on).
  - No phase-2 segment.
- **Sensitivity:** a Tier-2 run excludes these cases altogether.
- **In code:** `court/data_prep/phase_fixes.py` relabels them; `tests/` checks it on a hand-built case.

## D2 Cases that went to trial with no answer
- **Issue:** 90 cases (26 criminal, 64 civil) reached trial with no answer.
- **Decision:** they **moved on**.
  - Task-1 label = 1.
  - Phase 2 has zero length, so there is no phase-2 segment.
  - Their phase-3 segment is kept.
- **In code:** `phase_fixes.py`, with a hand-built test case.

## D3 Inverted phase-2 dates
- **Issue:** 50 cases have a phase-2 exit date earlier than their phase-2 entry date.
- **Decision:** drop the whole case (0.03% of cases).
- **In code:** `phase_fixes.py`; the count is logged in the data report.

## D4 Answered on the day the case ended
- **Issue:** 582 cases answered on their termination day.
- **Decision:** follow the phase table (`phase2_entered = False`).
  - Task-1 label = 0.
  - No phase-2 segment.
- **In code:** `phase_fixes.py`, with a hand-built test case.

## D5 ⚠️ Study window
- **Decision:** keep cases **filed 2003-01-01 to 2016-12-31**.
- **Why:** the data holds only closed cases.
  - Earlier filings survive only if they lasted long.
  - Later filings are missing their long cases (right-censoring).
- **In code:** a filter on `filing_date` in `court/data_prep/`; the count is in the data report.

## D6 Labels
- **Tk (classification):** 1 (`"deviant"`) if the case enters phase k+1, else 0 (`"regular"`).
  - For phase 3: 1 if the case ends with a verdict or sentence (`completes_phase_3`), 0 if it ends any other way.
  - The upstream `deviant`/`regular` strings are kept, so label handling is unchanged.
- **Rk (regression):** days from the prediction point (the last event in the prefix) to the phase-k exit date. The per-event column is `remaining_in_phase`.
- **In code:** `court/data_prep/segments.py`; tests check that regression labels equal `exit_date − last event date` and change per prefix.

## D7 ⚠️ Same-day leakage
- Dates are day-level. A prefix may only use events dated **strictly before** the phase-exit date. Events on the entry day are allowed.
- Segments left with no eligible events are dropped and counted.
- **In code:** trace building in `segments.py`; a perturbation test changes future events and checks the features stay the same.

## D8 Possibly leaky meta columns
- **Excluded from main runs:** counsel and party counts, and `*_share_pro_hac_vice`. They may be end-of-case values: `Defendants_counsels_count` has ρ ≈ 0.44, the largest single correlation.
  - They are included only in the Tier-2 D8 sensitivity run.
- **Never features, in any run:** these are outcome columns, and code asserts they are absent from every feature matrix.
  - `resolution_type`, `disposition_group`, `is_case_*`
  - `activity_count_*`, `total_*`, terminated counts (e.g. `Defendants_terminated_counts`, `Defendants_highest_offense_terminated`)
  - phase exit types, `phase*_los`, `phase*_activities`, `phase*_completed`, `case_status`, `terminating_date`
  - The same outcome columns repeated on every event-log row are dropped at ingest.
- **In code:** an exclusion list in `court/data_prep/export.py`, with a test that checks the assertion.

## D9 Trace scope
- A phase-k trace holds **every event from filing** up to the exit of phase k.
- Prefixes are cut only at events **inside** phase k.
- This matches the paper (aggregation over the whole case so far) and the Pre-phase definition.
- **In code:** `n_hist` per segment (events before phase entry); prefix lengths are `n_hist + j` and `prefix_nr = j` (PLAN.md C6).

## D10 Regression target
- Train on `log1p(days)`; predictions go back through `expm1`.
- Report MAE in days (primary), the median absolute error, and R² on the log scale.
- **In code:** `court/evaluate.py`.

## D11 Phase 3
- About 2,400 cases enter phase 3.
- T3 and R3 are run, but reported as **secondary** because the test set is small (≈ 480 cases).
- They come last in every queue tier.

## D12 Out of scope
- Not run:
  - KNN bucketing: significantly worse in the paper, and slow.
  - Index-based encoding: it needs equal-length traces per bucket, which D9 breaks.
  - SVM and LSTM.
- **In code:** the upstream KNN import becomes optional (WS2); these methods are not registered in `configs/experiments.yaml`.

---

### Inputs these decisions apply to (PLAN.md §0.3; MD5 re-checked 2026-10-21)
| Input | MD5 |
|---|---|
| `~/Downloads/Copy of df_event_log.csv` (full log, 4,811,483 events) | `0f18ebeb01d366898a29896d69e45091` |
| `~/Downloads/case_level_df.csv` (155,525 cases) | `97951dac7164a31a38e617503216c57a` |
| `~/Downloads/phases_event_log_cln.csv` | `fced1e192da99a5dec2adff4e1542e89` |

The event-log copies truncated at 1,000,000 rows are never used.
