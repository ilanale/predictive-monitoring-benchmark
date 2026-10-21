# Court-case predictive process monitoring (fork of the Teinemaa et al. 2019 benchmark)

This repository applies the outcome-oriented predictive process monitoring framework of Teinemaa et al. (2019)
to court event logs. It answers three questions. RQ1: at any point in phase *k* of a case, will the case move on to
phase *k+1*? (classification, tasks T1–T3). RQ2: how many days are left in the current phase? (regression,
R1–R3). RQ3: what do three views of the case add beyond the paper's baseline encoding? The views are Meta,
Pre-phase and Global (inter-case) features. The code is a fork of the authors' benchmark. Upstream behaviour is preserved
(see [How upstream behaviour was preserved](#how-upstream-behaviour-was-preserved)). The court adaptation lives in
`court/` and in config-gated changes to the upstream files, listed in `CHANGES_FROM_UPSTREAM.md`. The research plan
is `PLAN.md`; the fixed decisions D1–D12 are in `DECISIONS.md`, and the decisions taken during the run are in
`RUN_DECISIONS.md`. The upstream README is kept as [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md).

**Base paper.** Irene Teinemaa, Marlon Dumas, Marcello La Rosa, Fabrizio Maria Maggi. *Outcome-Oriented Predictive
Process Monitoring: Review and Benchmark.* ACM Transactions on Knowledge Discovery from Data 13(2):17, 2019.
Code: <https://github.com/irhete/predictive-monitoring-benchmark> (upstream commit `1047d6a`).

```
@article{teinemaa2019outcome,
  author  = {Irene Teinemaa and Marlon Dumas and Marcello La Rosa and Fabrizio Maria Maggi},
  title   = {Outcome-Oriented Predictive Process Monitoring: Review and Benchmark},
  journal = {{TKDD}}, volume = {13}, number = {2}, pages = {17:1--17:57}, year = {2019}
}
```

## Quick start

```bash
python3 -m venv ../.venv && ../.venv/bin/pip install pandas numpy scikit-learn xgboost hyperopt pyarrow matplotlib pyyaml
source ../.venv/bin/activate
export PYTHONPATH=$PWD:$PWD/experiments
# put the three input files in data/raw/ (see "Data"), then:
python -m court.run --config configs/experiments.yaml        # data → features → tuning → runs → results.md
python -m unittest discover -s tests -t .                    # test suite
```

`court.run` can be resumed. If it is interrupted, run the same command again: finished jobs (status `done` in
`results/runs.csv`) are skipped. Add `--skip-data` to reuse an existing `data/processed/`.

## Environment

- Developed on Python 3.15, pandas 3.0, numpy 2.5, scikit-learn 1.9, xgboost 3.4, hyperopt 0.3, pyarrow, matplotlib,
  and pyyaml. The upstream code was ported to this stack in the `compat:` commit (`2212f02`).
- **macOS / xgboost.** xgboost needs the OpenMP runtime. If `import xgboost` fails with a `libomp.dylib` error,
  run `brew install libomp`.
- **Fallbacks** (PLAN.md §6). If any fallback is used, it is logged in `RUN_DECISIONS.md` and in `results.md`.
  - **xgboost:** install from pip; if that fails, build from the GitHub source; as a last resort, use scikit-learn's
    `HistGradientBoosting*`.
  - **hyperopt:** install from pip; if that fails, install from the GitHub source (`pip install
    git+https://github.com/hyperopt/hyperopt`); as a last resort, use a randomized search with the same space and
    budget.
  - **Tests:** they use only the stdlib `unittest`. pytest is optional and can run them too.
- **Threads.** The court runner takes the XGBoost thread count from `budget.xgb_n_jobs` in the config. The upstream
  scripts take it from the env var `PPM_N_JOBS` (default 4). `court.report` caps BLAS/OpenMP threads at 2.

## Data

No data is committed. Everything under `data/` except `data/README.md` is git-ignored.

| Path | Content |
|---|---|
| `data/raw/df_event_log.csv` | full court event log: one row per docket event (`ucid`, `date_filed`, `Activity`, `event_judge`, `attribute_*`, …). Comma-separated, with an unnamed pandas index as the first column. The MD5 is checked against PLAN.md §0.3 |
| `data/raw/case_level_df.csv` | case table: one row per case (case type, court, judges, parties, phase dates, outcome columns) |
| `data/raw/phases_event_log_cln.csv` | phase log (phase entry and exit per case) |
| `data/interim/` | compact parquet of the event log and the cached input check (built by `court/data_prep/ingest.py`) |
| `data/processed/court_phase{1,2,3}.csv` / `.parquet` | one row per event of a segment trace (benchmark format, `;`-separated); the columns are in `court/SCHEMA.md` |
| `data/processed/court_columns.json` | column roles and the view membership (M, P, PP, B0G, G, S) |
| `data/processed/data_report.{md,json}` | cleaning counts, label rates, length quantiles, input MD5s (a copy is in `docs/data_report.md`) |

Make `data/raw/` files symlinks to the read-only originals; never edit, move or delete the originals. A *segment* is
(case, phase *k*). Its trace holds every event from filing up to the phase exit (D9). Prefixes are cut only at events
inside the phase.

## Repository tree

```
├── README.md                     this file
├── LICENSE                       Apache-2.0 (upstream, unchanged)
├── PLAN.md                       research plan, experiment matrix, Definition of Done
├── DECISIONS.md                  fixed decisions D1–D12
├── RUN_DECISIONS.md              decisions taken during the autonomous run (RD-xx)
├── STATUS.md                     run log and gate status
├── VERIFICATION.md               independent audits (G2 data, G4 leakage & code review)
├── CHANGES_FROM_UPSTREAM.md      every changed upstream file: what and why (compat, C1–C13)
├── results.md                    generated by court/report.py (do not edit)
├── .gitignore                    ignores data/, predictions, row-level caches, venvs
├── configs/
│   └── experiments.yaml          tasks, feature sets, tiers, budgets, caps (config reference below)
├── court/                        court adaptation (new code)
│   ├── SCHEMA.md                 WS1 → WS3 interface: columns and views of the processed files
│   ├── __init__.py               package marker
│   ├── court_confs.py            registers court tasks × feature sets in upstream dataset_confs (C1)
│   ├── data_io.py                task loading, max prefix length (C7), strict temporal split
│   ├── data_prep/                WS1 data build (raw → processed)
│   │   ├── __init__.py           package marker
│   │   ├── build.py              entry point: python -m court.data_prep.build
│   │   ├── ingest.py             read and check the raw inputs (MD5, line counts, duplicate sums)
│   │   ├── phase_fixes.py        D1–D5 rules and the study window
│   │   ├── segments.py           segments, labels (D6), traces cut at the exit (D7, D9)
│   │   ├── prefix_features.py    pre-phase (P, P+) event features
│   │   ├── global_features.py    inter-case (G, G+) features at each event date
│   │   └── export.py             CSV/parquet export, rare levels, leakage assertions (D8)
│   ├── evaluate.py               metrics, paired bootstrap, importance by view, baselines, per-year (C4, C11–C13)
│   ├── fast_pipeline.py          fast prefix encoding, identical to upstream (C9)
│   ├── feature_sets.py           FS definitions = unions of views (PLAN.md §1.2)
│   ├── report.py                 builds results.md, figures and tables from results/ (DoD-7)
│   ├── run.py                    single entry point: data → queue → report (DoD-2)
│   ├── run_queue.py              resumable job queue with time caps
│   ├── runner.py                 one experiment job: split, encode, fit, predict, save (C10)
│   ├── tune.py                   TPE tuning with the upstream search space (C3–C5)
│   ├── upstream.py               imports the adapted upstream modules for court tasks
│   └── verify_fast_path.py       G3 check: fast path = upstream path on a segment sample
├── experiments/                  upstream scripts (compat + C1–C8 gated by task config)
│   ├── BucketFactory.py          bucketer factory (compat: lazy KNN import, KMeans n_init)
│   ├── DatasetManager.py         reading, splitting, prefixes, labels (C1, C2, C5–C7 for court data)
│   ├── EncoderFactory.py         encoder factory (+ lastG, C8)
│   ├── dataset_confs.py          dataset registry (+ synthetic, + court tasks when PPM_COURT_DATA_DIR is set)
│   ├── experiments.py            final training and evaluation (C3, C4)
│   ├── optimize_params.py        TPE tuning (C3, C4)
│   ├── experiments_factor_levels.py, experiments_knn.py, experiments_long_cases.py   upstream variants (untouched)
│   ├── optimize_params_knn.py    upstream KNN tuning (untouched; KNN out of scope, D12)
│   ├── extracting_best_params_revision.ipynb   upstream notebook: per-bucket best params (untouched)
│   ├── get_bucket_sizes.py, get_number_of_trivial_predictions.py, write_case_length_counts.py   upstream helpers (untouched)
├── bucketers/                    upstream bucketers
│   ├── __init__.py               package marker
│   ├── ClusterBasedBucketer.py   k-means bucketing (untouched)
│   ├── NoBucketer.py             single bucket (compat)
│   ├── PrefixLengthBucketer.py   prefix-length buckets (compat; in-phase length for court data, C6)
│   └── StateBasedBucketer.py     last-activity buckets (compat)
├── transformers/                 upstream sequence encoders
│   ├── __init__.py               package marker
│   ├── AggregateTransformer.py   aggregation encoding (compat: named aggregation)
│   ├── IndexBasedExtractor.py    index-based helper (untouched)
│   ├── IndexBasedTransformer.py  index-based encoding (compat)
│   ├── LastStateTransformer.py   last-state encoding (compat)
│   ├── LastValueOfColsTransformer.py   new: last value of the G columns (C8)
│   ├── PreviousStateTransformer.py     previous-state encoding (compat)
│   └── StaticTransformer.py      static encoding (compat)
├── preprocessing/                upstream preprocessing of the BPIC/other logs (untouched, 8 scripts)
├── critical_difference_plots.R   upstream R plotting script (untouched)
├── tests/                        unittest suite (DoD-3)
│   ├── __init__.py               package marker
│   ├── compat_baseline.py        writes the compat-commit baseline outputs (DoD-4)
│   ├── reference_upstream.py     reference upstream implementations used by the tests
│   ├── test_compat_smoke.py      upstream behaviour = compat baseline on the synthetic log (DoD-4)
│   ├── test_data_prep.py         labels D1–D4/D6, leakage D7/D8, G features vs brute force
│   ├── test_fast_pipeline.py     fast path = upstream (C9), future-event perturbation
│   ├── test_pipeline_misc.py     split, D8 exclusions, FS views, bootstrap, baselines, runner, queue resume
│   ├── test_report.py            report.py on a tiny synthetic results dir
│   ├── test_upstream_adapted.py  adapted upstream with n_hist = 0 equals the original
│   └── fixtures/                 synthetic logs, notebook port, compat baseline files
│       ├── __init__.py, make_synthetic_log.py, make_court_like_log.py, extract_best_params_prefix.py
│       └── compat_baseline/      baseline AUC, features and predictions of the compat commit
├── verification/                 A4 audit scripts and their raw outputs (g2_*, g4_*)
├── docs/
│   ├── UPSTREAM_README.md        the original upstream README
│   └── data_report.md            WS1 data report (copy of data/processed/data_report.md)
├── data/                         git-ignored except README.md (see "Data")
└── results/                      runs.csv, params/, predictions/ (ignored), tables/, importance/, figures/, queue.log
```

## Running

All commands are run from the repo root, with `PYTHONPATH=$PWD:$PWD/experiments`.

| Stage | Command |
|---|---|
| everything | `python -m court.run --config configs/experiments.yaml [--skip-data] [--tier tier1a]` |
| 1. data build | `python -m court.data_prep.build` (dev sample: `--sample 0.02`) |
| 2. fast-path check (optional) | `python -m court.verify_fast_path --data-dir data/processed --n-segments 1000 --out results/verify_fast_path.json` |
| 3. experiment queue | `python -m court.run_queue --config configs/experiments.yaml --tier tier1a --tier tier1b` (`--only court_p1`, `--dry-run`) |
| 3a. one job | `python -m court.runner --job-id tier1a__court_p1_clf__single_agg__FS-B0__xgboost --config configs/experiments.yaml` |
| 3b. one tuning | `python -m court.runner --job-id tune__court_p1_clf__single_agg__xgboost --config configs/experiments.yaml` |
| 4. report | `python -m court.report --results-dir results --out results.md` (`--no-bootstrap` to use cached CIs only) |
| tests | `python -m unittest discover -s tests -t .` |

For long runs on macOS, wrap the queue: `caffeinate -dimsu python -m court.run_queue ...`.

The upstream argv interface still works, e.g. for a court task:
`PPM_COURT_DATA_DIR=data/processed python experiments/experiments.py court_p1_clf__FS-B0 <params_dir> <results_dir> single agg xgboost 1 10`.

## Config reference (`configs/experiments.yaml`)

| Key | Meaning |
|---|---|
| `tasks` | `court_p{1,2,3}_{clf,reg}`, the queue order (T1, R1, T2, R2, T3, R3) |
| `feature_sets` | `FS-M`, `FS-MP`, `FS-B0` (paper baseline), `FS-B0+P`, `FS-B0+G`, `FS-ALL`; suffix `+S` adds the D8 sensitivity view |
| `tiers.<tier>.methods` | bucketing + encoding (`single_agg`, `prefix_agg`; `state_agg`, `cluster_agg`, `single_laststate` are not wired into the court runner) |
| `tiers.<tier>.learners` | `xgboost` or `rf` |
| `tiers.<tier>.feature_sets` | overrides `feature_sets` for that tier |
| `tiers.<tier>.extra_jobs` | additional `{task, method, fs, learner}` jobs |
| `tier_order` | order in which tiers are queued |
| `tuned_fs` | the FS on which hyperparameters are tuned and then reused for the other FS of the same (task, method) |
| `budget.tuning` | `n_iter` (TPE iterations), `cv_folds`, `max_train_segments` (time-ordered sample), `cap_minutes`, `gap` (CV prefix gap) |
| `budget.final_fit.cap_minutes` | time cap per final fit |
| `budget.xgb_n_jobs` | XGBoost threads |
| `budget.gap_train` | training prefix gap g (test gap is always 1) |
| `budget.max_prefix_cap` | cap on L = min(cap, q90 of the in-phase length) |
| `data_dir`, `results_dir` | input and output locations |

## Adding a feature set or a task

- **Feature set.** Add it to `FS_VIEWS` in `court/feature_sets.py` as a list of views (`M`, `P`, `PP`, `B0G`, `G`,
  `S`). Then add its name to `feature_sets` in the config. `court/court_confs.py` registers it for the upstream
  scripts automatically. A new *view* needs a column prefix in `court/data_prep/export.py` and an entry under
  `views` in `court_columns.json` (see `court/SCHEMA.md`). Give it a label in `feature_sets.view_label` so that
  importance by view picks it up.
- **Task.** Tasks are named `court_p{k}_{clf|reg}` and read `data/processed/court_phase{k}.*`. Classification uses
  `label` (`deviant` = positive); regression uses `remaining_in_phase`. A new phase file only has to follow
  `court/SCHEMA.md`. A new target needs a column in the export, plus `target()` in `court/runner.py`, plus the
  matching branch in `DatasetManager.get_label`. Add the task to `tasks` in the config, and give it a short name
  in `TASK_NAMES` in `court/report.py`.

## Outputs

| Where | What |
|---|---|
| `results/runs.csv` | one row per job: status, runtime, sizes, L, split time, metrics, C13 baselines, params source, flags |
| `results/params/<job>.json` | the parameters used (fits) and all TPE trials (`tune__*`) |
| `results/predictions/<job>.csv.gz` | `segment_id, prefix_len, y_true, y_pred` for every test prefix (C10; git-ignored) |
| `results/importance/<job>.csv` | XGBoost total gain per view (C12) |
| `results/tables/` | per-prefix metrics, bootstrap Δ and CIs, summaries, drift, runtime (written by runner and report) |
| `results/figures/` | AUC/MAE by prefix length, importance, drift plots |
| `results/queue.log` | queue log (START/DONE/TIMEOUT/FAILED) |
| `results.md` | the generated report (DoD-7) |

## How upstream behaviour was preserved

- **One `compat:` commit (`2212f02`).** It ports the upstream code to the current library stack without research
  changes. The commit and the library drift it does not compensate are listed in `CHANGES_FROM_UPSTREAM.md`.
- **Config-gated research changes.** The research changes (C1–C8) apply only to datasets registered by
  `court/court_confs.py`, i.e. only when `PPM_COURT_DATA_DIR` is set. Upstream datasets keep
  `task_type = classification` and no `n_hist` offset.
- **DoD-4 test.** `tests/test_compat_smoke.py` runs the modified code on a synthetic 500-case log. It must
  reproduce the outputs of the compat commit (`tests/fixtures/compat_baseline/`): identical AUC rows, feature max
  abs diff < 1e-9, identical predictions.
- **Further checks.** `tests/test_upstream_adapted.py` checks that the in-phase prefix generation with
  `n_hist = 0` equals the original. `court/verify_fast_path.py` and `tests/test_fast_pipeline.py` check that the
  fast path (C9) gives the same features and predictions as the adapted upstream path.
- **Change notice.** Every modified upstream file starts with a change-notice line pointing to
  `CHANGES_FROM_UPSTREAM.md`.

## Troubleshooting

- **Memory.** The phase-2 file has about 2.5M event rows. The fast path encodes prefixes in chunks
  (`FastEncoder.max_chunk_cells`), but a fit with FS-ALL still needs several GB. Keep one heavy job at a time (the
  queue does this) and stay under about 70% of RAM. If memory runs out, follow PLAN.md §6 rule 3: process in
  chunks, then use gap g = 5, then cap the number of segments per job, and log each step in `RUN_DECISIONS.md`.
- **Time caps.** A job over its cap (`budget.*.cap_minutes`) is killed and rerun once on half the training segments
  (flag `timeout_halved`). If it times out again, it is recorded as `timeout`. Fits whose tuning did not finish
  use the default XGBoost params (flag `default_params`). `results.md` lists every flag.
- **`--sample-frac` needs its own `--results-dir`.** The queue skips jobs that are `done` in `runs.csv`. A sampled
  run written into `results/` would therefore stand in for the full run, and its rows would end up in the report.
  Always use, for example, `--sample-frac 0.02 --results-dir /tmp/ppm_sample_results`.
- **Light load next to a running queue.** Use a copy of the config with `budget.xgb_n_jobs: 2`.
- **Report on a running queue.** `court.report` only reads `results/`, but it writes `tables/`, `figures/` and the
  bootstrap cache there. To avoid touching a directory that is still being written, copy it first
  (`rsync -a results/ /tmp/snap/`) and run with `--results-dir /tmp/snap --out /tmp/snap/results.md`.
- **The upstream scripts cannot find the data.** Set `PPM_LOGS_DIR` (upstream logs) or `PPM_COURT_DATA_DIR`
  (court tasks), and make sure both the repo root and `experiments/` are on `PYTHONPATH`.

## License

Apache License 2.0 (see `LICENSE`, unchanged from upstream). Modified upstream files carry a change notice, and
the modifications are described in `CHANGES_FROM_UPSTREAM.md`.
