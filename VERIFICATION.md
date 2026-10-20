# VERIFICATION

## G2 — re-audit

**Auditor:** A4 Verifier, 2026-10-21. **Verdict: PASS.**

**What was audited.** The rebuilt `data/processed/` from fix commit 335f8aa, merged into `court-adaptation`. Files are dated 02:27; phase 1 now has 1,498,497 rows.

**Method.**
- Same independence rules as the first pass. I did not open the builder's code; everything is re-derived from `data/raw/*` with `verification/g2_data_audit.py`.
- Command: `G2_FSEED=2026 G2_CACHE=<dir> python verification/g2_data_audit.py`, then the same with `G2_TARGETED=1`.
- Evidence files: `verification/g2_results.json`, `g2_results_targeted.json`, `g2_feature_cells.csv` and `g2_feature_cells_targeted.csv`. These files now hold the re-audit results.

**What changed in my reference definitions.**
- `filed_last*` and `closed_last*` exclude the case itself. This is my original reading (RD-06), which the builder has now adopted.
- `PP_n_distinct_judges` counts every judge ID listed on an event. This follows data_report item 14, which now documents it; first-judge-only is kept only as a diagnostic.
- Nothing else changed.

| Check | Sample | Mismatches |
|---|---|---|
| **1 Labels/dates/traces**, all segments | 190,506 segments (127,610 / 60,412 / 2,484); label, entry, exit, n_events, last event date, n_hist, segment set | **0** (0 missing, 0 extra) |
| **1 Labels/dates/traces**, stratified sample (seed 7; 68 D1/D2/D4: D1 11, D2 50, D4 7) | 200 segments, including the full (date, Activity) sequence | **0** |
| **2 Features**, fresh sample (seed 2026) | 500 cells: 180 G_/open_cases, 320 P/PP; 107 columns; phases 182/207/111 | **0** |
| **2 Features**, targeted sample (seed 2027) | 300 cells: 200 filed/closed/net_flow, phase-1 and same_phase columns; 100 distinct-judges and prev-phase-days cells | **0** |
| **2 Full population** | every row: `open_cases`, `PP_prev_phase_days_1/2` | **0** |
| **3 Leakage scan**, every row of p1/p2/p3 | timestamp ≥ exit; `remaining ≠ exit − ts`; `in_phase`; `n_hist`; sort order; `event_nr`; empty segments; 2003–2016 window; `filing_year`; D3 present; segment_id | **0** on every item |
| **3 D2 identity** | phase-3 D2 rows where `remaining_in_phase == PP_prev_phase_days_1 − timesincecasestart` | **0 of 6,077** (was 6,077 of 6,077) |
| **3 D2 traces** | D2 phase-1 exit = phase-3 entry for all 83 windowed D2 cases | covered by the all-segment check, so no phase-1 event dates on or after the phase-3 entry |
| **3 Column review** | views M/P/PP/B0G/G vs the D8 outcome list; never_features; S-only suspects; file columns accounted for; typed lists = views | **0** issues (`P_attr_transferred_entry` is a pattern false positive) |
| **Twins p1/p2** | row counts; every 100th row compared on all 147 columns | p1 1,498,497 = 1,498,497; p2 2,491,716 = 2,491,716; **0** differing columns (14,985 and 24,918 rows). The p3 twin was identical in the first pass |
| **Rare levels → "other"** (RD-10), every row and segment | `judge`: raw first-judge ID re-derived for every row and aligned on (segment, position), < 10 rows per file → other. `M_` categoricals: < 10 segments per file → other | **0** in all three files (judge levels → other: p1 68, p2 53, p3 18; M levels → other match the data-report counts) |
| **M values** | 200 random cells against the case table (seed 2028), plus every segment for all 22 M and 6 S columns | **0** and **0** |

**Note on the twin "column hash".** The string-rendered MD5 of the 1% sample differs between CSV and parquet, because the text rendering of a float such as `1.0` is not the same as the text read back from the CSV. I therefore compared every column exactly as strings, or numerically within 1e-9 where the strings differ. All 147 columns matched.

**Non-blocking concerns still open.** These are already written up as limitations in the data report:
1. Global counts use only closed case-table cases, which causes undercounting near 2016 and a mild selection on future closure.
2. The case-table `District_Judge` and `Magistrate_Judge` may be the judge at the end of the case.
3. RD-13: the 875 dropped phase-1 segments are all positive.

The first-pass findings (D2 exit, self-inclusion in the filed counts, the undocumented distinct-judges definition) are all resolved.

## G2 — data audit (first pass, superseded by the re-audit above)

**Auditor:** A4 Verifier, 2026-10-21. **Verdict (first pass): FAIL** (one blocking defect, the D2 phase-1 exit date; it also causes leakage).

**Method.** I re-derived everything from `data/raw/*` with my own code (`verification/g2_data_audit.py`) and never opened `court/data_prep/`, `court/court_confs.py` or `tests/test_data_prep.py`. Specs used: PLAN §0.3/§1/§2/§6, DECISIONS D1–D12, `court/SCHEMA.md`, RUN_DECISIONS RD-06..RD-14. The first-pass evidence files are in commit 8563f49; they were overwritten by the re-audit. Run it with `G2_CACHE=<dir> python verification/g2_data_audit.py`; it takes about 15 s once the event-log subset is cached. Adding `G2_TARGETED=1` runs the supplementary sample instead.

**My reading of D1–D4 on the raw case table**

| Rule | Definition | All cases | In window |
|---|---|---|---|
| D1 | `phase1_exit_type == summary_judgment`, the same as "p2 entered, no p2 entry date, no p3" | 1,431 | 1,037 |
| D2 | `phase1_exit_type == trial`, the same as "p3 entered, no p2 entry date" | 90 | 83 |
| D3 | p2 exit < p2 entry | 50 | 47 |
| D4 | `phase1_completed & !phase2_entered` | 582 | 501 |

These counts equal the counts in the data report.

**Labels and dates I derived**

| Phase | Label | Entry | Exit |
|---|---|---|---|
| 1 | `phase2_entered & !D1` | filing | `phase1_exit_date`; for D2 cases, `phase3_entry_date` (D2: "phase 2 has zero length", so phase 1 ends when phase 3 starts) |
| 2 | `phase3_entered` | p2 entry | p2 exit |
| 3 | `phase3_completed` | p3 entry | p3 exit |

- Phase-2 segments exist only when `phase2_entered & !D1 & !D2`.
- D5 window: filing date from 2003-01-01 to 2016-12-31.
- Traces: every event with date < exit, sorted by (date, raw row).
- D7: a segment is dropped when it has no event with entry ≤ date < exit.

### Check 1 — labels, dates and traces

**Full population (all 190,506 output segments, not only a sample)**
- The segment sets match exactly: 0 segments missing, 0 extra. Segments per phase: 127,610 / 60,412 / 2,484.
- D7 drops match: p1 875 (all positive), p2 12, p3 13.
- `label`, `entry_date` and `n_hist`: 0 mismatches.
- `exit_date`, trace event count and last event date: **80 mismatches. All 80 are phase-1 segments of D2 cases.** The output uses the raw `phase1_exit_date`, which for D2 cases equals `terminating_date`. My reading uses the phase-3 entry date.
  - Example: `ilnd;;1:03-cv-05407|p1`. The output has exit 2006-09-05 and 71 events, last on 2006-09-01. My derivation has exit 2005-09-21 and 48 events, last on 2005-09-14.
  - Example: `ilnd;;1:03-cv-08825|p1`. The output has exit 2006-09-11 and 21 events. My derivation has exit 2005-04-05 and 15 events.
  - The other 3 of the 83 D2 cases have `phase3_entry == terminating_date`, so both readings give the same exit.

**Required stratified sample (200 segments, seed 7)**
- Strata: 34/34/33/33/33/33 over phase × label.
- D1/D2/D4-affected cases were oversampled: 68 of the 200 (D1 11, D2 50, D4 7).
- Fields compared: label, entry, exit, event count, last event date and the full (date, Activity) event sequence.
- **Mismatches: 17 segments.** All are D2 phase-1 segments, with wrong `exit_date`, `n_events` and `last_event_date`. The other 183 match exactly, event sequence included.

**Why this is also leakage (blocking)**
- **T1/R1:** the 80 D2 phase-1 traces (all labelled deviant) contain 2,555 events dated on or after the phase-3 entry: 237 `trial`, 44 `verdict`, 59 `judgment` and others. Their `remaining_in_phase` runs to the end of the case.
- **R3:** for all 6,077 phase-3 rows of D2 cases, `PP_prev_phase_days_1` = termination − filing, not p3 entry − filing. As a result, `remaining_in_phase == PP_prev_phase_days_1 − timesincecasestart` exactly on 6,077 of 6,077 rows. That is the R3 target, recoverable from two features. This affects 83 of 2,484 phase-3 segments (3.3%).
- **Global features (RD-07):** D2 cases are counted as "open in phase 1" while they are really in phase 3. This inflates `G_court_open_phase1`, `G_judge_open_phase1` and `G_judge_open_same_phase` (for phase-1 segments) by up to about 15 cases. It is small, but it is a mismatch.

### Check 2 — feature cells (brute force over all 155,475 non-D3 cases and the raw log)

**Required sample (500 cells, seed 7)**
- 180 cells from G_ and `open_cases`, 320 from P and PP; 108 distinct columns; phases 1/2/3 = 203/207/90.
- Every cell is recomputed from scratch at its own trace position. The trace timestamp aligned in 500 of 500 cells.
- **Mismatches: 17**, each explained by one of three causes:

| Cause | Cells | Columns |
|---|---|---|
| D2 phase-1 exit (Check 1 defect) | 10 | `G_court_open_phase1` 7, `G_judge_open_phase1` 3 |
| The case's own filing is counted in `filed_last30/90`, and so in `net_flow` | 6 | |
| `PP_n_distinct_judges` counts every judge listed in `event_judge`, not only the first (the "resource") | 1 | |

For each explained cell, recomputing with the stated alternative reproduces the output value exactly.

**Supplementary targeted sample (300 cells, seed 8)** over filed/net_flow, phase-1, same_phase, distinct-judges and prev-phase columns:
- 54 mismatches, all explained by the same three causes (filed self 28, D2 25, distinct judges 1). None is unexplained.

**Full-population checks (every row)**
- `open_cases` (B0G, RD-06: filed ≤ t < terminated, minus the case itself): 0 mismatches in 4,324,641 rows.
- `PP_prev_phase_days_1/2`:
  - 0 mismatches outside D2.
  - 6,077 mismatches on D2 phase-3 rows (see above).
  - Always 0 in phase 1, and `_2` is always 0 in phase 2, as RD-12 says.

All other columns sampled matched on every cell:
- time features in days (weekday with Monday = 0)
- every `P_attr_*`
- `P_judge_is_magistrate` (role of the first judge)
- `judge` (first ID, or "missing")
- `Activity`
- self, return and repeat loops
- `events_last30/90` over positions ≤ i
- court and judge open counts by type and by phase
- closed counts
- `judge_share_of_court` (judge open / court open)
- `mag_open`
- judge history median and n_closed (cases closed strictly before t; los = termination − filing; 0 if none)
- the seasonality columns

### Check 3 — leakage review of the data (every row of all three files)

| Check | p1 | p2 | p3 |
|---|---|---|---|
| trace events with timestamp ≥ exit_date | 0 | 0 | 0 |
| `remaining_in_phase ≠ exit − timestamp` | 0 | 0 | 0 |
| `in_phase ≠ (timestamp ≥ entry)` | 0 | 0 | 0 |
| `n_hist ≠` number of out-of-phase rows; history after in-phase rows | 0; 0 | 0; 0 | 0; 0 |
| traces not sorted by date; `event_nr ≠ position + 1` | 0; 0 | 0; 0 | 0; 0 |
| segments with no in-phase row (D7) | 0 | 0 | 0 |
| filing outside 2003–2016; `filing_year ≠` raw; D3 cases present | 0; 0; 0 | 0; 0; 0 | 0; 0; 0 |

These checks are internally consistent, but they do **not** catch the D2 defect, because the wrong `exit_date` is used consistently. Check 1 catches it.

**Column review (`court_columns.json`)**
- No outcome column (D8 list, plus `pending`, `unique_*_judge_count` and the phase/termination fields) is in views M, P, PP, B0G or G. The only pattern hit is `P_attr_transferred_entry`, an event attribute and a false positive.
- No `never_features` column is in any view.
- The six D8 suspect columns appear only as `S_*` (view S).
- Every file column is accounted for, and the typed lists equal the union of the views.
- Upstream twin: `court_phase3.csv` is identical to its parquet (all 147 columns, 331,873 rows). Phases 1 and 2 were not compared.

### Check 4 — `data_report.md` completeness (WS1 step 9)
- All required items are present:
  - counts at each cleaning step (D1–D5 before → after, with the PLAN anchors reconciled)
  - segments and prefixes per phase
  - label rates
  - in-phase length quantiles
  - excluded columns with reasons
  - input MD5s
- My recomputed counts agree: window after D3 128,485; segments built 128,485 / 60,424 / 2,497; D7 drops; segments kept.
- **Gap:** the report does not say how the D2 phase-1 exit is set, and its global-feature note ("this case excluded") is not true for the filed counts.

### Blocking
1. **D2 phase-1 exit date.**
   - The fix: set `exit_date(p1) = phase3_entry_date` (= raw `phase2_exit_date`) for D2 cases.
   - Apply the fix in segment building, in the trace cut, in the global phase-1 intervals (RD-07) and in `PP_prev_phase_days_1`.
   - Then rerun this audit. The expected result is 0 mismatches apart from the two definitional items below.

### Non-blocking concerns (ranked)
2. **`filed_last30/90` (court and judge) include the case itself.**
   - This is not leakage: the case's own filing is known at t.
   - It contradicts the data report's "this case excluded" and RD-06's "always excluding the case itself".
   - Either exclude it or document it.
3. **`PP_n_distinct_judges` counts every judge listed on an event, not only the first-judge "resource".** This is acceptable, but should be documented.
4. **The global features use only the 155,475 closed case-table cases.**
   - The full log has 175,268 cases, 7,161 of them still open. Cases still open at extraction are therefore absent, so open counts are too low, increasingly toward 2016+.
   - Membership in the case table depends on a future event (closure), which is a mild look-ahead or selection effect. Note it in the limitations.
5. **`District_Judge` and `Magistrate_Judge` (static, from the case table) may be the judge at the end of the case.** They drive M and every `G_judge_*` feature, while `unique_*_judge_count` was excluded as end-of-case. The risk is low: in the raw data, `unique_district_judge_count` > 0 for only 7,712 cases. Note it in the limitations.
6. **RD-13:** dropping the 875 phase-1 segments with no event before exit is all-positive, which is a selection effect. It is already noted.
7. **Not covered by this audit:**
   - M-column values and the rare-level → "other" mapping (no "other" cell was drawn)
   - CSV twins of phases 1 and 2
