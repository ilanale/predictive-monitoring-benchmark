# Interface contract: WS1 output → WS3 input

Written by the orchestrator at bootstrap so that A2 (data) and A3 (pipeline) can work in parallel. Changing it requires a line in `RUN_DECISIONS.md`.

## Files (in `data/processed/`, git-ignored)
- `court_phase{1,2,3}.csv`: `;`-separated, one row per event of a segment trace (benchmark format). Each file also has a `court_phase{k}.parquet` twin with identical content (faster to read; the CSV stays canonical).
- `court_columns.json`: column roles and view membership (below).
- `data_report.md`: WS1 step 9.

## Rows
- A **segment** is (case, phase k) for every phase the case entered after D1–D5.
- Its trace holds every event of the case with `date < exit_date` (D7, D9), sorted by (date, original row order).

## Fixed columns
| Column | Kind | Meaning |
|---|---|---|
| `segment_id` | case id | `"{ucid}\|p{k}"` |
| `ucid` | static, never a feature | original case id |
| `Activity` | dynamic cat (activity col) | event activity |
| `judge` | dynamic cat (resource col) | first judge ID parsed from `event_judge`; `"missing"` if none |
| `timestamp` | timestamp col | event date, `YYYY-MM-DD` |
| `label` | static | `"deviant"` = moved on / completed (D6), `"regular"` otherwise |
| `remaining_in_phase` | dynamic, label only | `exit_date − event date` in days (int); the regression target (D6, D10) |
| `in_phase` | dynamic, never a feature | 1 if `date ≥ entry_date` |
| `n_hist` | static, never a feature | number of trace events before phase entry (C6) |
| `entry_date`, `exit_date` | static, never a feature | phase-k dates after D1–D4 fixes |
| `filing_year` | static, never a feature | for the drift check |
| `timesincelastevent`, `timesincecasestart`, `event_nr`, `month`, `weekday` | dynamic num | the paper's time features, in **days** (no hour) |
| `open_cases` | dynamic num | the paper's single inter-case feature: court-wide open cases at the event date, excluding this case |

## Feature columns, by view prefix
| Prefix | View | Kind |
|---|---|---|
| `M_` | Meta (static encoding) | static cat or num; D8 suspect columns are **not** exported under `M_` but as `S_` (sensitivity only) |
| `P_` | Pre-phase, paper part | dynamic num: `P_attr_*` (0/1), `P_judge_is_magistrate` |
| `PP_` | Pre-phase extras (P+) | dynamic num: cumulative self-loops, return loops, repeats, distinct judges, events in the last 30/90 days; static num: `PP_prev_phase_days_*` |
| `G_` | Global extras (G+) | dynamic num, computed at each event date |

`Activity`, `judge` and the five time features belong to view **P** (paper part); `open_cases` belongs to view **B0G**.

## `court_columns.json`
```json
{
  "case_id_col": "segment_id", "activity_col": "Activity", "resource_col": "judge",
  "timestamp_col": "timestamp", "label_col": "label",
  "regression_label_col": "remaining_in_phase",
  "static_cat": [], "static_num": [], "dynamic_cat": [], "dynamic_num": [],
  "views": {"M": [], "P": [], "PP": [], "B0G": ["open_cases"], "G": [], "S": []},
  "never_features": ["ucid", "in_phase", "n_hist", "entry_date", "exit_date", "filing_year", "remaining_in_phase", "label"],
  "excluded": {"<raw column>": "<reason, e.g. D8 outcome column>"}
}
```

## Feature sets (PLAN.md §1.2)
| FS | Views |
|---|---|
| FS-M | M |
| FS-MP | M + P + PP |
| FS-B0 | M + P + B0G |
| FS-B0+P | M + P + PP + B0G |
| FS-B0+G | M + P + B0G + G |
| FS-ALL | M + P + PP + B0G + G |
| *+S (D8 sensitivity)* | any FS + S |
