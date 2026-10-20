# PLAN — Court-case phase prediction on top of Teinemaa et al. (2019)

**Status:** APPROVED by Niv on 2026-10-21, with all D1–D12 defaults and the run cut from 24h to **8h**. T0 is in `STATUS.md`.
**Owner:** Claude (orchestrator). **Mode after approval:** 8 hours of autonomous, multi-agent work, with no questions to Niv.
**Where it runs:** a **local Claude Code session on Niv's MacBook Pro** (Apple silicon), started in `~/Documents/Thesis/claude-run/`. The plan was written in a claude.ai cloud chat; read `session_migration.md` first.
**Base paper:** Teinemaa, Dumas, La Rosa, Maggi. *Outcome-Oriented Predictive Process Monitoring: Review and Benchmark.* ACM TKDD 13(2):17, 2019.
**Base code:** `github.com/irhete/predictive-monitoring-benchmark` (Apache-2.0, last upstream commit `1047d6a`).

---

## 0. Before kickoff: what I need from you (≈15 minutes)

You'll be offline for the whole run, so every choice is made here. Items marked ⚠️ change results a lot.

### 0.1 Prerequisites

| # | What | Why | If you skip it |
|---|---|---|---|
| P1 ✅ | **Full event log: DONE.** Found on the Mac and checked (see §0.3) | — | — |
| P2 ✅ | **Local Python environment: DONE (2026-10-21).** `claude-run/.venv` on Python 3.15.0 with xgboost 3.4.1, hyperopt 0.3.0, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.4, matplotlib 3.11.2, pyarrow 26.0.0. `brew install libomp` (23.1.4) was needed for xgboost; an `XGBClassifier` test fit works. Mac: Apple M5, 10 cores (4P+6E), 24 GB RAM, 267 GB free | — | — |
| P2b ✅ | **Mac plugged in and awake: confirmed by Niv.** The run also wraps itself in `caffeinate -dimsu` | — | — |
| P3 ✅ | **Fork: `https://github.com/ilanale/predictive-monitoring-benchmark.git`** (checked: at upstream `1047d6a`, parent `irhete/predictive-monitoring-benchmark`). Push access checked 2026-10-21: `gh` logged in as `ilanale` (HTTPS, `repo` scope), and a dry-run push of `court-adaptation` succeeds | Work gets pushed to a branch `court-adaptation` of the fork, with the upstream history kept | — |
| P4 ✖ | **Skipped by Niv.** Authors' benchmark datasets will not be provided | — | Port is checked on a synthetic log only (G1, DoD-4), noted as a limitation |

### 0.2 Decisions (all defaults approved by Niv on 2026-10-21; full record in `DECISIONS.md`)

| ID | Decision | Default |
|---|---|---|
| D1 ⚠️ | **1,431 civil cases closed by summary judgment with no answer** (flagged "entered phase 2" but no entry date). These are *all* the summary-judgment cases in the data | Treat as **closed in phase 1 by a court decision**: task-1 label = 0 (did not move on), no phase-2 segment. A sensitivity run excludes them |
| D2 | **90 cases that went to trial with no answer** (26 criminal, 64 civil) | They **moved on** (task-1 label = 1). Phase 2 has zero length, so no phase-2 segment. Their phase-3 segment is kept |
| D3 | **50 cases whose phase-2 exit date is before the phase-2 entry date** | Drop the whole case (0.03%) |
| D4 | **582 cases that answered on the day the case ended** | Follow your own phase table (`phase2_entered = False`): task-1 label = 0 |
| D5 ⚠️ | Study window | Cases **filed 2003-01-01 to 2016-12-31**. Earlier filings are only the long-lasting ones; later filings are missing their long cases, since only closed cases are in the data |
| D6 | Labels | **Tk (classification):** 1 if the case enters phase k+1, else 0. For phase 3: 1 if it ends with a verdict or sentence (`completes_phase_3`), 0 if it ends any other way. **Rk (regression):** days from the prediction point to the phase-k exit date |
| D7 ⚠️ | Same-day leakage | Dates are day-level. A prefix may only use events dated **strictly before** the phase-exit date (events on the entry day are allowed). Segments with no eligible events are dropped and counted |
| D8 | Possibly leaky meta columns (counsel and party counts, `pro_hac_vice` shares) | **Excluded** from main runs; included only in a sensitivity run. Outcome columns (`resolution_type`, `disposition_group`, `is_case_*`, `activity_count_*`, `total_*`, terminated counts, phase exit types) are **never** features, and this is asserted in code |
| D9 | Trace scope | A phase-k trace holds **every event from filing** up to the exit of phase k. Prefixes are only cut at events inside phase k. This matches the paper (aggregation over the whole case so far) and your Pre-phase definition |
| D10 | Regression target | Train on `log1p(days)`. Report MAE in days (primary), median absolute error, and R² on the log scale |
| D11 | Phase 3 (~2,400 cases) | Run it, but report it as **secondary** (small test set ≈ 480 cases) |
| D12 | Out of scope | KNN bucketing (significantly worse in the paper, and slow), index-based encoding (needs equal-length traces per bucket, which breaks with D9), SVM, LSTM |

### 0.3 Data on this Mac (read-only inputs: never modify, move or delete)

Use **only** these files. Link them into the repo as `data/raw/` (symlinks, not copies).

| Input | Absolute path on the Mac | Size | Rows (incl. header) | MD5 |
|---|---|---|---|---|
| **Event log (FULL)** | `/Users/ilanalecht/Downloads/Copy of df_event_log.csv` | 2,868,742,483 B | 4,811,484 (4,811,483 events) | `0f18ebeb01d366898a29896d69e45091` |
| Event log (FULL, gzip copy of the same file) | `/Users/ilanalecht/Downloads/df_event_log_full.csv.gz` | 78,025,243 B | same | uncompressed MD5 = the line above |
| Case table | `/Users/ilanalecht/Downloads/case_level_df.csv` | 67,255,833 B | 155,526 (155,525 cases) | `97951dac7164a31a38e617503216c57a` |
| Phase log | `/Users/ilanalecht/Downloads/phases_event_log_cln.csv` (identical copy: `~/Documents/Thesis/data/phases_event_log_cln.csv`) | 87,558,837 B | 385,759 | `fced1e192da99a5dec2adff4e1542e89` |
| Base paper (PDF) | `~/Documents/Thesis/claude-run/references/Teinemaa2019_outcome_oriented_PPM.pdf` | 22.9 MB | — | — |

⚠️ **Do NOT use these files.** They are **truncated at 1,000,000 rows**, the Apple Numbers limit, and cover only ~36k cases from 2000–2006:
- `~/Downloads/df_event_log.csv`
- `~/Desktop/df_event_log.csv`
- `~/Downloads/df_event_log.numbers` and `~/Documents/Thesis/data/df_event_log.numbers`

The `*.numbers` copies of the case table hold the same data as the CSV; use the CSV.

**Facts already checked in the cloud session** (WS1 re-asserts them as tests):
- The full log has 175,268 cases (7,161 still open; the rest are closed). **All 155,525 case-table cases are in it**, with 4,237,231 events. **Use only the case-table cases.**
- For every case, `sum(attribute_duplicates)` = `total_activity_count` exactly. The plain number of event rows is lower than `total_activity_count` in 43% of cases (median difference 0, mean −2.07), because collapsed duplicates count once.
- Event dates run from 1991-01-08 to 2021-07-11. Booleans are written `True`/`False` (the truncated copies use `TRUE`/`FALSE`).
- The first column of every CSV is an unnamed pandas index. Drop it.

---

## 1. Research framing (what the code must make possible)

**Thesis claim:** we apply the outcome-oriented predictive process monitoring framework of Teinemaa et al. (2019) to court event logs. We use it for two tasks, *phase-transition prediction* and *time remaining in the current phase*. We also extend its encoding with a structured set of **global (inter-case) features**. The paper used only one such feature (open cases) and named a richer set as future work (Sec. 7).

| RQ | Question | Tasks | Primary metric |
|---|---|---|---|
| RQ1 | How well can we predict, at any point in phase k, whether the case moves to phase k+1? | T1, T2, T3 (classification) | AUC overall (prefix-weighted, as in the paper) + AUC by in-phase prefix length |
| RQ2 | How well can we predict the days left in the current phase? | R1, R2, R3 (regression) | MAE (days), overall + by prefix length |
| RQ3 | What does each view add (Meta / Pre-phase / Global) beyond the paper's baseline? | All 6 | Change against the paper baseline, with 95% bootstrap confidence intervals |

### 1.1 The three views mapped onto the paper's taxonomy

| Your view | Paper term | Content |
|---|---|---|
| **Meta (M)** | static encoding | case_type, city, nature_suit, District_Judge, Magistrate_Judge, offense at opening, is_mdl, party flags, related_case_count, Other_courts, party *shares* (non-suspect) |
| **Pre-phase (P)** | aggregation encoding of event attributes + timestamp features | activity counts; `attribute_*` flags as 0/1 numeric (aggregated); event judge as the "resource"; the paper's time features in **days**: `timesincelastevent`, `timesincecasestart`, `event_nr`, `month`, `weekday` (no hour, since timestamps are dates only); **plus court extras P+:** self-loops, return loops (A→B→A), repeats, distinct judges seen, events in the last 30/90 days, durations of earlier phases |
| **Global (G)** | inter-case features | the paper's `open_cases`, **plus G+:** open cases in the court by type and by phase; the judge's open cases, by type and by phase, and in *this case's* phase; filed/closed in the last 30/90 days for the court and the judge; net flow; the judge's share of the court; magistrate load; the judge's median case length on cases closed before t; seasonality (month sin/cos, quarter, days to year end, December flag). Computed **at each event date**, from the case table covering all 155k cases |

### 1.2 Feature sets for the ablation

| FS | Contents | Role |
|---|---|---|
| FS-M | M only | lower bound, ignores the trace |
| FS-MP | M + P (no global at all) | intra-case only |
| **FS-B0** | M + P (paper part only, no P+) + `open_cases` | **the paper's baseline, as it would run on our data** |
| FS-B0+P | FS-B0 + P+ | adds court pre-phase extras |
| FS-B0+G | FS-B0 + G+ | adds global extras |
| **FS-ALL** | M + P + P+ + G + G+ | full model |

The main comparisons are FS-ALL vs FS-B0, FS-B0+G vs FS-B0, and FS-B0+P vs FS-B0. G columns are encoded with the paper's aggregation (mean/max/min/sum/std) **plus** their last value, which is the state at the moment of prediction. The last-value part is new; it is documented and has its own test.

---

## 2. Workstream WS1: data preprocessing

**Inputs (paths in §0.3):**
- `case_level_df.csv` (155,525 cases)
- `phases_event_log_cln.csv`
- the **full** event log (4,811,483 events). Read it in chunks or with selected `usecols` (it's 2.9 GB); keep only case-table cases; save a compact parquet in `data/interim/`.

**Output:** `data/processed/court_phase{1,2,3}.csv` (`;`-separated, the benchmark's format) + `data/processed/data_report.md`.

### Steps
1. **Ingest and check.**
   - Load all three files and check that the case IDs match.
   - Check the event log's MD5 and row count against §0.3, and stop if they differ. Keep only case-table cases, and assert `sum(attribute_duplicates) == total_activity_count` for each.
   - Compare each case's event count with `total_activity_count` and report the distribution of differences.
2. **Apply D1–D4 and D5.** Log the count affected by each rule into the data report (before → after).
3. **Build segments.** For each case and each phase k it entered: segment id `ucid|pk`. Columns:
   - `entry_date`, `exit_date`
   - `label` (D6, as `"deviant"` = 1 / `"regular"` = 0, so the upstream label handling is unchanged)
   - `n_hist` (events before phase entry)
4. **Build the trace per segment (D9, D7).**
   - Take all case events with date < `exit_date`, sorted by (date, original row order).
   - Mark `in_phase = date ≥ entry_date`.
   - Per event, compute `remaining_in_phase = exit_date − event date` in days (the regression label, D10–D11).
5. **Pre-phase columns per event.**
   - Upstream timestamp features (days).
   - `attribute_*` as 0/1.
   - `event_judge` parsed into the first judge ID (dynamic categorical "resource") and a magistrate flag.
   - Cumulative P+ counters: self-loops, return loops, repeats, distinct judges.
   - The durations of earlier phases are static per segment.
6. **Global columns per event** (event date = t):
   - Vectorized interval counting (`searchsorted`), reused from the existing `build_features.py`.
   - Judge history median: build each judge's running-median array once, at each case closure, then look up t. Never use a loop of medians over every event.
   - `open_cases` (the paper's single feature) = court-wide open cases at t, excluding this case.
7. **Rare categories.** Keep only categorical levels with ≥ 10 occurrences, except `Activity`, as upstream does.
8. **Export and register.**
   - One CSV per phase holding both labels.
   - Register 6 tasks × 6 feature sets in `experiments/dataset_confs.py`, with names like `court_p1_clf__FS-B0`. The upstream CLI keeps working unchanged.
9. **Data report** (`data_report.md`):
   - counts at every cleaning step
   - segments and prefixes per task
   - label rates
   - in-phase length quantiles
   - list of excluded columns and why
   - input file checksums (from §0.3)

**Known reference numbers** (2003–2016 window, before D1/D2 relabeling — used as sanity anchors):

| Phase | Entered | Moved on |
|---|---|---|
| 1 | 127,365 | 60,925 (48%) |
| 2 | 60,424 | 2,414 (4%) |
| 3 | 2,414 | 1,588 (66%) |

---

## 3. Workstream WS2: fork and modernize the upstream code (no research changes)

The goal is the paper's code running on Python 3.13 / pandas 3 / scikit-learn 1.9, with **behavior unchanged**. This happens in one dedicated commit, `compat:`.

Known breakages to fix (minimum list; the agent adds any it finds):

| File | Problem | Fix |
|---|---|---|
| `transformers/AggregateTransformer.py` | dict-renaming `groupby().agg({'mean':np.mean,...})`, removed in pandas ≥ 1.0 | named aggregation, same columns and order, `std` with ddof=1 (as pandas did) |
| `experiments/BucketFactory.py` | imports `bucketers/KNNBucketer.py`, which isn't in the repo | lazy/optional import (KNN is out of scope, D12) |
| all scripts | `DataFrame.append`, `.ix`, chained assignment, `np.float`/`np.int`, `get_dummies` dtype changes | modern equivalents |
| `experiments/*.py` | xgboost/hyperopt API drift (`XGBClassifier` args, `fmin` signature) | current APIs, same search space (paper Table 7) |
| `experiments/*.py` | relative `logs_dir` and argv-only interface | keep argv; add an optional `--config` wrapper in `court/` |

- **Check:** the port runs end to end on a synthetic 500-case log. If P4 data is provided, `single_agg` and `prefix_agg` with XGBoost reproduce the paper's Table 8 AUC within ±0.02 on that dataset.
- **Apache-2.0 compliance:** every modified upstream file gets a header line `# Modified by <fork> for court-case adaptation — see CHANGES_FROM_UPSTREAM.md`, and LICENSE is kept.

---

## 4. Workstream WS3: research adaptation (the changes we discussed, made concrete)

Everything is switched on by config, so upstream datasets still behave exactly as before. The plain-language overview:

| # | Change | Where | Detail |
|---|---|---|---|
| C1 | **Task type flag** | `dataset_confs.py` (new dict `task_type[dataset] ∈ {classification, regression}`), read by `DatasetManager` | No CLI change |
| C2 | **Per-prefix label for regression** | `DatasetManager.get_label` | Classification: `groupby(case).first()[label]` (unchanged). Regression: `groupby(case).last()['remaining_in_phase']`, because the target changes with every prefix |
| C3 | **Learner** | `experiments.py`, `optimize_params.py` | `XGBClassifier → XGBRegressor` and `RandomForestClassifier → RandomForestRegressor`; same TPE space and `n_estimators=500`; `predict_proba → predict` (and `expm1` back from the log) |
| C4 | **Metric and tuning loss** | same files | AUC → MAE (days). The TPE loss becomes `+MAE` (it was `−AUC`). Classification also reports F1 at 0.5 and PR-AUC (new, because T2 is imbalanced) |
| C5 | **Classification-only helpers** | `DatasetManager` | `StratifiedKFold → KFold` (regression). Small-bucket fallback: class ratio → train median. `get_pos_case_length_quantile` → quantile over all segments (regression). The "one class only" check → "constant target" |
| C6 | **Prefixes inside the phase** (D9) | `DatasetManager.generate_prefix_data` | Per-segment offset `n_hist`: prefix lengths are `n_hist + j`, for j = 1, 1+g, … ≤ L. `prefix_nr := j` (the in-phase length), so prefix-length bucketing and the per-length evaluation use the in-phase length. Upstream behavior is unchanged when `n_hist = 0` |
| C7 | **Max prefix length** | experiments scripts | The paper's rule: `min(40, q90 of in-phase length of the positive class)`, i.e. T1 ≈ 30, T2/T3 = 40; regression uses all segments. Training gap **g = 3** (a paper setting, RQ3.3); test gap 1 (as in the paper) |
| C8 | **G encoding: aggregation + last value** | new `transformers/LastValueOfColsTransformer.py`, combined in `EncoderFactory` as `agg+lastG` | Only for G columns, only in FS with G+. Upstream encoders are untouched |
| C9 | **Fast path, same results** | new `court/fast_pipeline.py` | Upstream's pandas prefix log (`head(n)` + concatenation) would build 3–10M rows × >100 columns, which is too slow and memory-hungry at this data size. The fast path computes running counts/sums/sums-of-squares/min/max once per event and picks the rows at prefix ends. It produces **the same feature matrix** as upstream `generate_prefix_data` + `FeatureUnion`. It reuses upstream's split (`split_data_strict`), bucketers, learners, search space and metric code |
| C10 | **Saved predictions** | runner | Per run, write `segment_id, prefix_len, y_true, y_pred` as `.csv.gz`, so every reported number can be recomputed |
| C11 | **Paired bootstrap confidence intervals** | `court/evaluate.py` | 1,000 resamples of test segments; Δ(FS_x − FS-B0) with a 95% CI for AUC / MAE |
| C12 | **Importance by view** | `court/evaluate.py` | XGBoost gain summed per view (M / P / P+ / G / G+) for each final model |
| C13 | **Baselines for regression** | `court/evaluate.py` | Training median of remaining days (i) overall and (ii) per in-phase prefix length |

**Split (unchanged from the paper):** `split_data_strict`. Segments are ordered by their first in-phase event. The first 80% train, the last 20% test. Training events on or after the test start are cut.

**Tuning:**
- TPE with 3-fold CV inside train.
- **Budget (8h run):** `n_iter = 10` per (task, method), run on a time-ordered sample of ≤ 20k training segments, with the search space unchanged. (The 24h plan had 15 / 40k.)
- Hyperparameters are tuned on **FS-ALL** and reused for the other feature sets of the same (task, method).
- A Tier-2 fairness check re-tunes FS-B0 separately for T1 and R1.

---

## 5. Experiment matrix and compute budget

Hardware (measured 2026-10-21): **Apple M5 MacBook Pro, 10 cores (4P+6E), 24 GB RAM, 267 GB free.** Never plan on more than 70% of RAM (≈ 16.8 GB). XGBoost `n_jobs = 8`. One heavy job at a time, through a resumable queue (`court/run_queue.py`). The queue skips jobs that are already done, writes results as it goes, and stops any job past its time cap.

**8h budget.** The experiment window is G4 (T0+3:30) to T0+6:15. Queue order inside each tier: T1, R1, T2, R2, then T3, R3 (phase 3 last, D11).

| Tier | Runs | Count |
|---|---|---|
| **Tier 1a (must)** | 6 tasks × single_agg × 6 FS, XGBoost; tuning 6 × 10 iterations on FS-ALL | 36 final fits + 6 tunings |
| **Tier 1b (must, after 1a)** | 6 tasks × prefix_agg × 6 FS, XGBoost; tuning 6 × 10 iterations on FS-ALL | 36 final fits + 6 tunings |
| Tier 2 (should, only if time) | in this order: (1) FS-B0 re-tune for T1 and R1; (2) D1 sensitivity (exclude); (3) D8 sensitivity (with suspect columns); (4) state_agg, cluster_agg, single_laststate on FS-B0 and FS-ALL; (5) RF on the best Tier-1 method | up to ~40 |
| Tier 3 | drift check only (per-year test performance, paper Sec. 5.3), computed from saved predictions with no retraining. Logistic/linear regression: **not run (8h budget)** | — |

Time caps: tuning 20 min per job, final fit 10 min. Over the cap, the job is killed, rerun on a smaller sample, and flagged.

---

## 6. Multi-agent organization

| Agent | Owns | Starts | Must not |
|---|---|---|---|
| **Orchestrator** (me) | plan, `RUN_DECISIONS.md`, `STATUS.md`, merges, gates, experiment queue, final handoff | T0 | change D1–D12 |
| **A1 Port** | WS2 + environment (packages, xgboost/hyperopt build) | T0 | make research changes |
| **A2 Data** | WS1 + data report | T0 | touch upstream code |
| **A3 Pipeline** | WS3 (C1–C13), unit tests | after gate G1 (~T0+1:15) | run full-scale experiments |
| **A4 Verifier** (independent, never sees the build process, only the outputs) | leakage audit, label/feature spot checks against raw data, diff review against upstream, results-vs-raw-files check → `VERIFICATION.md` | G2, G4, G6 | write pipeline code |
| **A5 Writer** | `results.md` (generated from raw result files by `court/report.py`), figures, `README.md`, `CHANGES_FROM_UPSTREAM.md` | ~T0+4:00 | type any number by hand |

**How they work together:**
- Each agent works on its own git branch or worktree. The orchestrator merges at the gates.
- Agents report through `STATUS.md` (one line per milestone, with a timestamp).
- Development agents test on samples ≤ 5% of the data, so they don't fight the queue for CPU.

### Gates (timeline from kickoff T0)

| Gate | By | Pass condition |
|---|---|---|
| Bootstrap | T0+0:15 | repo cloned, branch `court-adaptation`, symlinks, worktrees, first commit |
| G1 port | T0+1:15 | upstream runs on the synthetic log (P4 skipped by Niv, so no published-AUC check); `compat:` commit merged |
| G2 data | T0+2:00 | `data_report.md` complete; A4 data audit passes (labels for 200 random segments re-derived from raw data: 0 mismatches; 500 random G/P feature values recomputed by brute force: 0 mismatches) |
| G3 pipeline | T0+3:15 | all unit tests green; fast path = upstream path (max difference < 1e-6, identical predictions given the same parameters) on a 1,000-segment sample per task; dry run of all 6 tasks on a 2% sample |
| G4 review | T0+3:30 | A4 leakage review passes → **launch Tier 1a** |
| G5 Tier 1 | T0+6:00 | all Tier-1 runs done (or cut/flagged with a reason); Tier-2 cut-off |
| G6 results | T0+7:30 | `results.md`, `README.md` and `VERIFICATION.md` done; A4 confirms every number in results.md matches the raw files |
| Handoff | **≤ T0+7:45** | push / zip / final message |

Gate failures: one fix cycle of ≤ 30 min, then the conservative fallback, logged in `RUN_DECISIONS.md`.

### Autonomy rules (no questions to Niv)
1. **Precedence:** D1–D12 are fixed (`DECISIONS.md` holds only those). Anything new goes into `RUN_DECISIONS.md` (time, ID, decision, options, choice, why). The conservative option (less leakage, fewer assumptions) wins.
2. **Fallbacks, in order:**
   - **xgboost:** pip (with `brew install libomp` if import fails) → build from GitHub source (≤ 90 min) → sklearn `HistGradientBoosting*`
   - **hyperopt:** pip → GitHub source → `RandomizedSearch` with the same space and budget
   - **pytest:** missing → tests use stdlib `unittest`, which pytest can still run
   Each fallback is logged as a deviation in results.md.
3. **Out of memory:** process in chunks, then gap g = 5, then cap the number of segments per job. Each step is logged.
4. **Running late:** at T0+5:30 the orchestrator projects when Tier 1b will finish. If it won't finish by T0+6:15, Tier 1b is cut to FS-B0 and FS-ALL only (logged). Tier 2 only runs in what's left before T0+6:00. Writing starts at T0+6:15 whatever happens (A5 drafts from ~T0+4:00). A partial result is better than a missing one; incomplete runs are marked in results.md.
5. **Safety:** never modify, move or delete the input files in §0.3; never push to upstream; push only to the fork branch `court-adaptation`; no secrets or data in git (`data/` is ignored).
6. **Liveness (local Mac):**
   - The experiment queue runs as a background process, wrapped as `caffeinate -dimsu python -m court.run_queue ...`, logging to `results/queue.log`.
   - The orchestrator checks the queue at least every 45 min. If this Claude Code build offers a wake-up or scheduling tool, use it for those check-ins; otherwise poll between other tasks.
   - If the Mac slept or restarted, the queue resumes from `results/runs.csv`.
   - Commits at every milestone.

---

## 7. Definition of Done

The work is **done** only when every item below is true. A4 checks them all, and the results go into `VERIFICATION.md`.

**Code and repository**
- [ ] DoD-1: The repo is a fork of upstream with history kept. The branch `court-adaptation` holds: one `compat:` commit, then the research commits. Every modified upstream file carries the change notice. `CHANGES_FROM_UPSTREAM.md` lists each changed file → what changed → why (C1–C13 + compat).
- [ ] DoD-2: One command rebuilds everything from raw inputs: `python -m court.run --config configs/experiments.yaml` (data → features → tuning → runs → `results.md`). The queue resumes after an interruption.
- [ ] DoD-3: `python -m unittest discover -s tests` passes. At minimum:
  - fast path = upstream (C9)
  - no event after the prediction point is used (D7): checked by perturbing future events, which must leave the features unchanged
  - excluded columns never appear in any feature matrix (D8)
  - labels follow D1–D4/D6 on hand-built cases
  - regression labels change per prefix and equal `exit − last event date`
  - the split has no segment in both train and test, and no train event on or after the test start
  - G features match a brute-force recomputation
- [ ] DoD-4: Upstream behavior is preserved: with `task_type=classification` and `n_hist=0`, the modified code gives the same outputs as the `compat:` commit on the synthetic log (and on P4 data, if given).

**Experiments**
- [ ] DoD-5: All Tier-1 runs are finished, each with saved predictions, parameters, runtime, and an entry in `results/runs.csv`.
- [ ] DoD-6: For every task: overall metrics, per-prefix-length curves, paired-bootstrap ΔCIs against FS-B0, importance by view, and regression baselines (C13).

**Documents**
- [ ] DoD-7: `results.md` exists and is **generated** from `results/` (no number typed by hand). It contains, in this order:
  1. a summary (≤ 8 bullets)
  2. data and cleaning counts, with the input checksums
  3. the protocol and each deviation from the paper, with its reason
  4. RQ1 tables + AUC-by-prefix plots
  5. RQ2 tables + MAE-by-prefix plots against the baselines
  6. RQ3 ablation with CIs + importance by view
  7. Tier 2/3 results (or "not run: reason")
  8. drift check
  9. runtime
  10. limitations and open issues
  11. commands to reproduce
- [ ] DoD-8: `README.md` covers:
  - what the repo is (one paragraph) and the citation of the base paper
  - quick start
  - environment setup (including the offline/GitHub-source fallbacks)
  - where to put the data and the expected file formats (no data in git)
  - the full repo tree, with one line per directory and file
  - how to run each stage separately and all stages together
  - the config reference (tasks, feature sets, methods, budgets)
  - how to add a new feature set or task
  - outputs and where they land
  - how upstream behavior was preserved
  - troubleshooting (memory, time caps)
  - license
- [ ] DoD-9: Also present: `DECISIONS.md` (D1–D12 only), `RUN_DECISIONS.md` (every autonomous decision), `data/processed/data_report.md`, `VERIFICATION.md` (A4: each DoD item ✔/✘ with evidence; zero open blockers), and `PLAN.md` (this file).

**Handoff**
- [ ] DoD-10: Pushed to your fork (if P3) **and** saved as `~/Documents/Thesis/claude-run/deliverables/court-ppm_<date>.zip` (code + results, no data). A final message in the local session, also saved as `claude-run/deliverables/HANDOFF.md`, with: what's done, the headline results, what isn't done and why, the next three steps. All of this before T0+8h.

---

## 8. Target repository layout

Location: `~/Documents/Thesis/claude-run/predictive-monitoring-benchmark/` (a clone of the fork, or of upstream if there's no fork yet).

```
predictive-monitoring-benchmark/          (fork; branch court-adaptation)
├── LICENSE                               upstream Apache-2.0 (kept)
├── README.md                             rewritten for this project (DoD-8); upstream README → docs/UPSTREAM_README.md
├── PLAN.md  DECISIONS.md  RUN_DECISIONS.md  STATUS.md  CHANGES_FROM_UPSTREAM.md  VERIFICATION.md
├── results.md                            generated (DoD-7)
├── bucketers/  transformers/             upstream (+ compat fixes; + LastValueOfColsTransformer, C8)
├── experiments/                          upstream scripts (+ compat; + C1–C7); dataset_confs.py registers court_* tasks
├── preprocessing/                        upstream preprocessing scripts (untouched)
├── court/
│   ├── data_prep/  ingest.py  phase_fixes.py  segments.py  prefix_features.py  global_features.py  export.py
│   ├── fast_pipeline.py                  C9
│   ├── evaluate.py                       metrics, bootstrap, importance by view, baselines (C4, C11–C13)
│   ├── run_queue.py                      resumable job queue with time caps
│   ├── report.py                         builds results.md + figures from results/
│   └── run.py                            single entry point (DoD-2)
├── configs/experiments.yaml              tasks, feature sets, methods, tiers, budgets, caps
├── tests/                                unittest suite (DoD-3)
├── results/  runs.csv  params/  predictions/  figures/  tables/
├── docs/UPSTREAM_README.md
└── data/  (git-ignored)  raw/ (symlinks to the §0.3 files)  interim/  processed/  README.md (what goes where)
```

---

## 9. Risks I already know about

| Risk | Effect | Mitigation |
|---|---|---|
| Someone uses a truncated copy of the event log by mistake | Wrong results | §0.3 lists the right file and its MD5; WS1 checks MD5 and row count before anything else and stops if they don't match |
| Mac sleeps, shuts down or goes offline | The run stalls | P2b + `caffeinate`; resumable queue |
| Mac RAM/cores unknown in advance | Tier 2/3 may not finish | Measure at kickoff; tier order + time caps + fast path (C9) |
| xgboost/libomp install on macOS | Up to ~1h lost | Fallback chain, logged as a deviation |
| Tuning on FS-ALL then reusing | Could slightly favor FS-ALL | Tier-2 re-tune of FS-B0 for T1 and R1 |
| T3 is small | Wide CIs | Reported as secondary (D11) |
| Only closed cases in the data | Bias at the edges of the time range | Window D5; drift check; stated in limitations |
| Work lost if the session ends early | Lost work | Commits at every milestone; zip in `deliverables/` and push before T0+7:45 |

---

**Approved 2026-10-21.** The 8-hour clock (T0) starts at approval (time in `STATUS.md`); no questions to Niv until the final handoff message.
