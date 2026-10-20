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
