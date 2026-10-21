# Changes from upstream

Upstream: [irhete/predictive-monitoring-benchmark](https://github.com/irhete/predictive-monitoring-benchmark)
(Teinemaa et al. 2019), Apache-2.0 (see `LICENSE`, kept unchanged). Every modified upstream file
starts with the line `# Modified by ilanale/predictive-monitoring-benchmark for court-case adaptation — see CHANGES_FROM_UPSTREAM.md`.

## compat — port to Python 3.15 / pandas 3.0 / numpy 2.5 / scikit-learn 1.9 / xgboost 3.4 / hyperopt 0.3

Goal: the upstream code runs on the current stack with **behaviour unchanged** (no research changes).
Untouched on purpose: `experiments/experiments_*.py`, `optimize_params_knn.py`, the helper scripts
(`get_*`, `write_case_length_counts.py`), the notebook, `preprocessing/`, the R file.

| File | What changed | Why |
|---|---|---|
| `transformers/AggregateTransformer.py` | `groupby()[num_cols].agg({'mean':np.mean,...})` → named aggregation (`pd.NamedAgg`) producing the same columns `<func>_<col>` in the same function-major order (mean, max, min, sum, std; each over `num_cols`); `std` is pandas' groupby std, ddof=1 | dict-renaming agg was removed in pandas 1.0. Old pandas concatenated one block per dict key (key = outer column level) and dispatched `np.std` to the cythonized groupby std (ddof=1) |
| `transformers/AggregateTransformer.py`, `StaticTransformer.py`, `LastStateTransformer.py`, `PreviousStateTransformer.py`, `IndexBasedTransformer.py` | `pd.get_dummies(..., dtype=np.uint8)` | pandas ≥ 2.0 returns `bool` dummies; the old default was `uint8`. With `bool`, `groupby().sum()` of dummies is still a count but `max()` stays boolean; forcing `uint8` reproduces the old numeric 0/1 matrices |
| `transformers/PreviousStateTransformer.py` | `groupby().nth(-2)` followed by `.set_index(case_id_col)` | since pandas 2.0 `nth` is a filter that keeps the original row index; the old result was indexed by case id (needed by the later `reindex`) |
| `transformers/IndexBasedTransformer.py` | `max_events` from `X.groupby(case_id).size().max()`; case-id frame built from the sorted group keys instead of `grouped.apply(lambda x: x.name)` | with `as_index=False`, `size()` now returns a DataFrame (so `.max()` was a Series); `apply` on group frames changed semantics. Same values as before |
| `bucketers/NoBucketer.py` | `dtype=np.int` → `dtype=int` | `np.int` removed in numpy 1.24 (it was an alias of `int`) |
| `bucketers/PrefixLengthBucketer.py`, `bucketers/StateBasedBucketer.py` | `.as_matrix()` → `.to_numpy()` | `as_matrix` removed in pandas 1.0 |
| `experiments/BucketFactory.py` | `KNNBucketer` imported lazily inside the `knn` branch | `bucketers/KNNBucketer.py` does not exist in the repo, so the module-level import broke every run. KNN is out of scope (D12) |
| `experiments/BucketFactory.py` | `KMeans(..., n_init=10)` | scikit-learn 1.4 changed the default `n_init` from 10 to `"auto"` (=1 for k-means++); 10 keeps the old behaviour of cluster bucketing |
| `experiments/experiments.py`, `experiments/optimize_params.py` | `XGBClassifier(seed=...)` → `random_state=...` (same value 22); added `n_jobs` from env `PPM_N_JOBS` (default 4) | `seed` is a deprecated alias in the sklearn API; thread cap keeps the machine light and does not change the model |
| `experiments/experiments.py`, `experiments/optimize_params.py` | `LogisticRegression(..., solver="liblinear")` | default solver changed from `liblinear` to `lbfgs` in scikit-learn 0.22; the paper's runs used liblinear |
| `experiments/dataset_confs.py` | `logs_dir = os.environ.get("PPM_LOGS_DIR", "labeled_logs_csv_processed")` | lets callers point to the data without editing code; default unchanged |

Unchanged on purpose (and why it is still correct): the search spaces and settings in `optimize_params.py`
(paper Table 7; `hp.uniform`/`hp.quniform`/`scope.int`, `fmin(..., algo=tpe.suggest, max_evals=n_iter)`
and `space_eval` work as before on hyperopt 0.3), `n_estimators=500`, `train_ratio=0.8`, `n_splits=3`,
`random_state=22`, the argv interface, `DatasetManager.py` (runs unchanged on pandas 3).

### Library drift that is *not* compensated (documented deviations)
- **XGBoost defaults.** The paper used xgboost ≤ 0.7, where `tree_method="auto"` meant `exact` (small data)
  or `approx` (large data) and `base_score=0.5`. xgboost ≥ 2.0 defaults to `tree_method="hist"` (256 bins)
  and estimates `base_score` from the data. We keep the library defaults (old xgboost cannot be bit-reproduced
  anyway, and `exact` would be very slow on the court data). Expect small AUC differences vs. the paper.
- **hyperopt seeding.** Upstream never seeded `fmin`, so tuning is not reproducible run-to-run (unchanged).
  hyperopt 0.3 honours the env var `HYPEROPT_FMIN_SEED` if a caller wants determinism.
- **AggregateTransformer column order.** Reconstructed from pandas 0.2x semantics (dict insertion order on
  Python ≥ 3.6); old pandas could not be installed to verify bit-for-bit. Order only affects which columns
  `colsample_bytree` samples, not the feature values.
- **Prefix-bucket parameters.** `optimize_params.py` writes one flat dict for `prefix_*`, while `experiments.py`
  reads `args[bucket]`. Upstream bridged this with `extracting_best_params_revision.ipynb` (untouched);
  `tests/fixtures/extract_best_params_prefix.py` ports that notebook cell.
- **Known upstream bug kept:** in `experiments.py`, when a test bucket has no training prefixes, `test_y_all`
  is not extended, so the final results DataFrame would have mismatched lengths. Not triggered by
  `single`/`prefix` bucketing on our data; left as is (behaviour unchanged).
- `IndexBasedTransformer` emits pandas `PerformanceWarning` (column-by-column inserts); harmless.

### How to run (unchanged argv interface)
Upstream modules import each other by bare name and `EncoderFactory` imports `transformers.*`, so both the
repo root and `experiments/` must be on `PYTHONPATH`:
```
export PYTHONPATH=$PWD:$PWD/experiments PPM_LOGS_DIR=<dir with the csv logs> PPM_N_JOBS=4
python experiments/optimize_params.py <dataset> <params_dir> <n_iter> <bucket> <encoding> <cls>
python experiments/experiments.py <dataset> <params_dir> <results_dir> <bucket> <encoding> <cls> <gap> <n_iter>
```

## test — synthetic log and compat baseline (validation only)

| File | What | Why |
|---|---|---|
| `experiments/dataset_confs.py` | dataset `synthetic` (`synthetic.csv` under `logs_dir`) | lets the unchanged upstream scripts run on the synthetic log |
| `tests/fixtures/make_synthetic_log.py` | deterministic (seed 22) 500-case log in the upstream processed format, trace lengths 2–30, learnable deviant/regular label, upstream time features | the paper's datasets are not available; validates the port end to end. Writes `data/synthetic/synthetic.csv` (git-ignored) |
| `tests/fixtures/extract_best_params_prefix.py` | port of the notebook's per-bucket best-params cell | `experiments.py` needs `{nr_events: params}` for `prefix_*` |
| `tests/compat_baseline.py`, `tests/fixtures/compat_baseline/` | outputs of the compat code with fixed XGBoost params (single_agg, prefix_agg): AUC results files from `experiments.py`, encoded test features (first 200 single-bucket prefixes; first 70 of prefix buckets 1/5/10), predictions for all test prefixes | regression baseline (DoD-4) |
| `tests/test_compat_smoke.py` | `python -m unittest discover -s tests` reproduces the baseline (AUC rows identical, features max abs diff < 1e-9, predictions identical) | guards later changes against silent behaviour drift |

## research — court adaptation (PLAN.md §4, C1–C9)

All changes are switched on only for datasets registered by `court/court_confs.py` (names
`court_p{k}_{clf|reg}__{FS}`), i.e. only when `PPM_COURT_DATA_DIR` is set. Upstream datasets keep
`task_type = classification` and no `n_hist` offset, so their code path is the compat one
(`tests/test_compat_smoke.py` stays green; `tests/test_upstream_adapted.py` checks that the
in-phase prefix generation with `n_hist = 0` equals the original one).

| File | What changed | Why (PLAN.md) |
|---|---|---|
| `experiments/dataset_confs.py` | when `PPM_COURT_DATA_DIR` is set, calls `court.court_confs.register` (adds dicts `task_type`, `regression_label_col`, `global_cols`, `feature_set`; court tasks × 6 FS, plus `+S`) | C1, RD-03 |
| `experiments/DatasetManager.py` | reads `task_type`, `regression_label_col`, `global_cols`; `n_hist_col = "n_hist"` for court datasets | C1 |
| `experiments/DatasetManager.py` | `get_label`: regression → `groupby(case).last()[remaining_in_phase]`; `get_label_numeric` → `log1p(days)` | C2, D6, D10 |
| `experiments/DatasetManager.py` | `get_class_ratio` (small-bucket fallback) → `log1p(median remaining days)`; `get_stratified_split_generator` / `get_idx_split_generator` → `KFold` for regression | C5 |
| `experiments/DatasetManager.py` | `generate_prefix_data` → `_generate_in_phase_prefix_data` for court data: lengths `n_hist + j`, j = 1, 1+gap, … ≤ max_length; `prefix_nr = j`; ids `case` / `case_<n_hist+j>`; `case_length` = in-phase length capped | C6, D9 |
| `experiments/DatasetManager.py` | `get_pos_case_length_quantile` → quantile of the in-phase length (positive class; all segments for regression) | C7 |
| `experiments/DatasetManager.py` | `split_data_strict` (court): segments ordered by first in-phase event; test start = earliest test segment's phase entry; train events on/after it cut | PLAN.md §4 "Split", D9 (traces start at filing) |
| `bucketers/PrefixLengthBucketer.py` | court prefixes (with `n_hist`) are bucketed by `prefix_nr` (in-phase length) | C6 |
| `experiments/experiments.py`, `experiments/optimize_params.py` | `XGBRegressor(objective='reg:squarederror')` / `RandomForestRegressor` and `predict` for regression; same search space and `n_estimators=500` | C3 |
| `experiments/experiments.py` | regression results rows: `mae` (days, after `expm1`) per `nr_events` and overall | C4 |
| `experiments/optimize_params.py` | regression score = −MAE (days), so the loss is +MAE and the per-bucket best-trial extraction (max score) is unchanged | C4 |
| `experiments/experiments.py`, `experiments/optimize_params.py` | encoders: `agg` + `lastG` when the FS has G columns; `static` only for FS-M (no dynamic columns) | C8, FS-M |
| `experiments/EncoderFactory.py`, new `transformers/LastValueOfColsTransformer.py` | method `lastG`: last non-null value of the G columns, named `last_<col>`, `fillna(0)` | C8 |
| new `court/fast_pipeline.py` (+ `court/verify_fast_path.py`) | prefix features without materialising the prefix log; bit-identical to the adapted upstream path (running Kahan sums and FMA-Welford reproduce pandas' cython groupby mean/sum/std); identical XGBoost predictions on 1,000-segment samples of all six tasks | C9 |

## research — C10–C13 and the court-only modules (no upstream file changed)

| File | What | Why (PLAN.md) |
|---|---|---|
| `court/runner.py` | per job: `predictions/<job>.csv.gz` (`segment_id, prefix_len, y_true, y_pred`), `params/<job>.json`, per-prefix table, a `runs.csv` row | C10: every reported number can be recomputed |
| `court/evaluate.py` | `paired_bootstrap`: Δ(FS_x − FS-B0) with a 95% percentile CI, 1,000 resamples of test segments | C11 |
| `court/evaluate.py`, `court/runner.py` | `importance_by_view`: XGBoost total gain summed per view (M / P / P+ / G / G+ / S) for each final model | C12 |
| `court/evaluate.py`, `court/runner.py` | `regression_baselines`: training median of remaining days, overall and per in-phase prefix length | C13 |
| `court/report.py` | builds `results.md`, `results/figures/`, `results/tables/` from `results/` only; vectorised paired bootstrap (same resamples as `evaluate.paired_bootstrap`, checked in `tests/test_report.py`), cached; C13 per-length curves rebuilt from `data/processed` and checked against `runs.csv`; drift check and RD-26 restricted evaluation | DoD-7 |
| `court/run_queue.py`, `court/run.py`, `court/tune.py`, `court/data_io.py`, `court/fast_pipeline.py`, `court/feature_sets.py`, `court/court_confs.py`, `court/upstream.py`, `court/verify_fast_path.py`, `court/data_prep/*` | new modules: queue with time caps, single entry point, TPE tuning, split/L, fast path, feature sets, task registration, data build | DoD-2, C1, C7, C9, WS1 |

## Index: every changed upstream file

Checked against `git diff 1047d6a..HEAD --stat -- experiments transformers bucketers` (15 files: 14 modified, 1 new).
Files in those directories that are not listed here are unchanged.

| Upstream file | compat | test | research |
|---|---|---|---|
| `bucketers/NoBucketer.py` | `np.int` → `int` | – | – |
| `bucketers/PrefixLengthBucketer.py` | `.as_matrix()` → `.to_numpy()` | – | C6: bucket by `prefix_nr` (in-phase length) when `n_hist` and `prefix_nr` are present |
| `bucketers/StateBasedBucketer.py` | `.as_matrix()` → `.to_numpy()` | – | – |
| `experiments/BucketFactory.py` | lazy `KNNBucketer` import; `KMeans(n_init=10)` | – | – |
| `experiments/DatasetManager.py` | – | – | C1 task type and court columns; C2 per-prefix regression label; C5 fallbacks and `KFold`; C6 in-phase prefixes; C7 in-phase length quantile; court `split_data_strict` |
| `experiments/EncoderFactory.py` | – | – | C8: `lastG` method, `global_cols` argument |
| `experiments/dataset_confs.py` | `logs_dir` from `PPM_LOGS_DIR` | dataset `synthetic` | C1: court tasks registered via `court_confs` when `PPM_COURT_DATA_DIR` is set |
| `experiments/experiments.py` | `random_state` (was `seed`), `n_jobs` from `PPM_N_JOBS`, `solver="liblinear"` | – | C3 regressors; C4 MAE rows; C8 `agg+lastG`; FS-M static-only |
| `experiments/optimize_params.py` | same as `experiments.py` | – | C3 regressors; C4 score = −MAE; C8 `agg+lastG`; FS-M static-only |
| `transformers/AggregateTransformer.py` | named aggregation; `get_dummies(dtype=np.uint8)` | – | – |
| `transformers/IndexBasedTransformer.py` | `max_events` and case-id frame without `apply`; `uint8` dummies | – | – |
| `transformers/LastStateTransformer.py` | `uint8` dummies | – | – |
| `transformers/LastValueOfColsTransformer.py` (new) | – | – | C8: last non-null value of the G columns |
| `transformers/PreviousStateTransformer.py` | `nth(-2)` + `set_index`; `uint8` dummies | – | – |
| `transformers/StaticTransformer.py` | `uint8` dummies | – | – |

Each change is explained, with its reason, in the compat, test and research sections above. `README.md` was
rewritten for this project; the upstream README is kept unchanged as `docs/UPSTREAM_README.md`.
