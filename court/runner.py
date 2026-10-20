"""One experiment job: (task, method, FS, learner, params) -> predictions, params, runs.csv row.

Mirrors upstream ``experiments.py`` (strict temporal split, bucketing, per-bucket learner,
small-bucket fallbacks) on top of the C9 fast path, with the C1-C5 regression changes.

    python -m court.runner --job-id tier1a__court_p1_clf__single_agg__FS-B0__xgboost \
        --config configs/experiments.yaml [--sample-frac 0.02] [--train-frac 0.5]
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import pandas as pd

from court import data_io, evaluate
from court.fast_pipeline import FastEncoder, fit_columns, prepare_log
from court.feature_sets import get_spec

RANDOM_STATE = 22
SUPPORTED_METHODS = ["single_agg", "prefix_agg"]
PHASE2_METHODS = ["state_agg", "cluster_agg", "single_laststate"]
DEFAULT_XGB = {"learning_rate": 0.05, "subsample": 0.8, "max_depth": 6, "colsample_bytree": 0.8,
               "min_child_weight": 1}
DEFAULT_RF = {"max_features": 0.5}
RUNS_COLUMNS = ["job_id", "tier", "task", "method", "fs", "learner", "status", "runtime_s",
                "n_train_prefixes", "n_test_prefixes", "n_train_segments", "n_test_segments", "max_prefix_len",
                "split_ts", "auc", "f1", "pr_auc", "mae", "median_ae", "r2_log", "baseline_mae_overall",
                "baseline_mae_per_prefix", "params_source", "flags", "finished_at"]


def make_job_id(tier, task, method, fs, learner):
    return "%s__%s__%s__%s__%s" % (tier, task, method, fs, learner)


def parse_job_id(job_id):
    tier, task, method, fs, learner = job_id.split("__")
    return dict(tier=tier, task=task, method=method, fs=fs, learner=learner)


def tune_id(task, method, learner):
    return "tune__%s__%s__%s" % (task, method, learner)


def append_run(results_dir, row):
    path = os.path.join(results_dir, "runs.csv")
    os.makedirs(results_dir, exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RUNS_COLUMNS})


def read_runs(results_dir):
    path = os.path.join(results_dir, "runs.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=RUNS_COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def make_learner(learner, task_type, params, n_jobs):
    if learner == "xgboost":
        import xgboost as xgb
        kw = dict(n_estimators=500, learning_rate=float(params["learning_rate"]),
                  subsample=float(params["subsample"]), max_depth=int(params["max_depth"]),
                  colsample_bytree=float(params["colsample_bytree"]),
                  min_child_weight=int(params["min_child_weight"]), random_state=RANDOM_STATE, n_jobs=n_jobs)
        if task_type == "classification":
            return xgb.XGBClassifier(objective="binary:logistic", **kw)
        return xgb.XGBRegressor(objective="reg:squarederror", **kw)
    if learner == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if task_type == "classification" else RandomForestRegressor
        return cls(n_estimators=500, max_features=float(params["max_features"]), random_state=RANDOM_STATE,
                   n_jobs=n_jobs)
    raise ValueError(learner)


def target(meta, task_type):
    """C2/D10: classification label per segment; regression log1p(remaining days at prefix end)."""
    if task_type == "classification":
        return meta["y_clf"].to_numpy()
    return np.log1p(meta["y_reg"].to_numpy())


def bucket_of(method, train_meta, test_meta):
    """Upstream NoBucketer / PrefixLengthBucketer (C6: bucket = in-phase length j; a test j unseen
    in train goes to the largest train j below it)."""
    if method.startswith("single"):
        return np.ones(len(train_meta), int), np.ones(len(test_meta), int)
    if method.startswith("prefix"):
        tr = train_meta["prefix_nr"].to_numpy()
        known = np.unique(tr)
        te = test_meta["prefix_nr"].to_numpy()
        pos = np.searchsorted(known, te, side="right") - 1
        te_b = np.where(pos >= 0, known[np.maximum(pos, 0)], -1)
        return tr, te_b
    raise NotImplementedError("method %s needs the upstream bucketers (phase 2)" % method)


def fit_predict_buckets(tr, te, method, learner, task_type, params, n_jobs, fallback, train_events_ratio=None):
    """Returns (pred in model space: proba or log1p days, list of (model, colnames) per bucket,
    per-bucket info). ``params``: dict or {"per_bucket": {j: dict}, "best": dict}."""
    b_tr, b_te = bucket_of(method, tr.meta, te.meta)
    y_tr = target(tr.meta, task_type)
    pred = np.full(len(te.meta), np.nan)
    models = []
    for b in np.unique(b_te):
        te_idx = np.flatnonzero(b_te == b)
        tr_idx = np.flatnonzero(b_tr == b)
        if len(tr_idx) == 0:
            pred[te_idx] = fallback
            continue
        yb = y_tr[tr_idx]
        if len(np.unique(yb)) < 2:  # upstream: one class only -> predict it (C5: constant target)
            pred[te_idx] = yb[0]
            continue
        p = params
        if isinstance(params, dict) and "per_bucket" in params:
            p = params["per_bucket"].get(str(int(b)), params["best"])
        elif isinstance(params, dict) and "best" in params:
            p = params["best"]
        # avoid copies of the (large) matrices when the bucket is the whole set (single_agg)
        trb = tr if len(tr_idx) == len(tr.meta) else tr.subset_rows(tr_idx)
        mask = fit_columns(trb)
        Xte = te.X if len(te_idx) == len(te.meta) else te.X[te_idx]
        if not mask.all():
            trb = trb.subset_cols(mask)
            Xte = Xte[:, mask]
        model = make_learner(learner, task_type, p, n_jobs)
        model.fit(trb.X, yb)
        if task_type == "classification":
            pos_idx = int(np.where(model.classes_ == 1)[0][0])
            pred[te_idx] = model.predict_proba(Xte)[:, pos_idx]
        else:
            pred[te_idx] = model.predict(Xte)
        models.append((model, trb.names, trb.sources))
    return pred, models


def build_matrices(df, task_type, spec, max_len, gap_train, gap_test=1, dtype=np.float32, dm=None, train_frac=None):
    """Strict temporal split, then C9 fast-path matrices (train gap ``gap_train``, test gap 1).

    ``dm``: upstream DatasetManager (court.upstream.dataset_manager); its ``split_data_strict``
    (C6-adapted) is used when given, else the court re-implementation in data_io (identical;
    checked by court.verify_fast_path and tests)."""
    if dm is not None:
        train, test = dm.split_data_strict(df, 0.8, split="temporal")
        split_ts = data_io.segment_start_times(test).min() if len(test) else None
    else:
        train, test, split_ts = data_io.split_data_strict(df)
    if train_frac and train_frac < 1:
        # timeout rerun: same split and test set, evenly spaced (in time) half of the train segments
        start = data_io.segment_start_times(train).sort_values(kind="mergesort")
        keep = systematic_segments(start.index.to_numpy(), frac=train_frac)
        train = train[train["segment_id"].isin(set(keep))]
    enc = FastEncoder(spec, dtype=dtype)
    train_p = prepare_log(train)
    enc.build_vocab(train_p)
    tr = enc.encode(train_p, max_len, gap_train, prepared=True)
    te = enc.encode(prepare_log(test), max_len, gap_test, prepared=True) if len(test) else None
    return tr, te, split_ts, train


def systematic_segments(meta_or_df, frac=None, cap=None, order_col=None):
    """Evenly spaced subset of segments in time order (keeps the whole period)."""
    segs = meta_or_df
    n = len(segs)
    k = n if cap is None else min(n, cap)
    if frac is not None:
        k = min(k, max(1, int(round(n * frac))))
    if k >= n:
        return segs
    return segs[np.linspace(0, n - 1, k).round().astype(int)]


def load_params(results_dir, task, method, learner):
    path = os.path.join(results_dir, "params", tune_id(task, method, learner) + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f), "tuned:" + os.path.basename(path)
    return {"best": dict(DEFAULT_XGB if learner == "xgboost" else DEFAULT_RF)}, "default"


def run_job(job_id, cfg, data_dir, results_dir, sample_frac=None, train_frac=None, flags=""):
    t0 = time.time()
    j = parse_job_id(job_id)
    task, method, fs, learner = j["task"], j["method"], j["fs"], j["learner"]
    phase, task_type = data_io.parse_task(task)
    if method not in SUPPORTED_METHODS:
        raise NotImplementedError("%s: phase-2 method (needs upstream bucketers)" % method)
    n_jobs = int(cfg.get("budget", {}).get("xgb_n_jobs", 4))
    gap_train = int(cfg.get("budget", {}).get("gap_train", 3))
    df = data_io.load_task_log(task, data_dir, sample_frac=sample_frac)
    cols = data_io.load_columns(data_dir)
    spec = get_spec(fs, cols, df.columns)
    from court import upstream
    dm = upstream.dataset_manager(task, fs, data_dir)
    # C7 via the adapted upstream quantile (in-phase length; positive class / all segments)
    L = min(int(cfg.get("budget", {}).get("max_prefix_cap", 40)), dm.get_pos_case_length_quantile(df, 0.90))
    if train_frac and train_frac < 1:
        flags = (flags + ";" if flags else "") + "train_frac=%s" % train_frac
    tr, te, split_ts, train_events = build_matrices(df, task_type, spec, L, gap_train, dm=dm, train_frac=train_frac)
    del df
    keep = fit_columns(tr)  # upstream: dummy columns exist only for values seen in train
    if not keep.all():
        tr, te = tr.subset_cols(keep), te.subset_cols(keep)
    params, params_source = load_params(results_dir, task, method, learner)
    if params_source == "default":
        flags = (flags + ";" if flags else "") + "default_params"
    if task_type == "classification":
        fallback = float((train_events["label"] == "deviant").mean())  # upstream get_class_ratio(train)
    else:
        fallback = float(np.log1p(np.median(tr.meta["y_reg"])))     # C5: train median
    pred, models = fit_predict_buckets(tr, te, method, learner, task_type, params, n_jobs, fallback)
    out = pd.DataFrame({"segment_id": te.meta["segment_id"], "prefix_len": te.meta["prefix_nr"]})
    if task_type == "classification":
        out["y_true"] = te.meta["y_clf"].to_numpy(); out["y_pred"] = pred
    else:
        out["y_true"] = te.meta["y_reg"].to_numpy(); out["y_pred"] = np.expm1(pred)
    for sub in ["predictions", "params", "tables", "importance"]:
        os.makedirs(os.path.join(results_dir, sub), exist_ok=True)
    out.to_csv(os.path.join(results_dir, "predictions", job_id + ".csv.gz"), index=False)
    with open(os.path.join(results_dir, "params", job_id + ".json"), "w") as f:
        json.dump({"job_id": job_id, "params": params, "params_source": params_source, "max_prefix_len": L,
                   "gap_train": gap_train, "sample_frac": sample_frac, "train_frac": train_frac,
                   "n_features": [len(m[1]) for m in models]}, f, indent=1, default=str)
    evaluate.metrics_by_prefix(out, task_type).to_csv(
        os.path.join(results_dir, "tables", job_id + "__by_prefix.csv"), index=False)
    if learner == "xgboost" and models:
        gains = {}
        for model, names, _ in models:
            sc = model.get_booster().get_score(importance_type="total_gain")
            for k, v in sc.items():
                name = names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k
                gains[name] = gains.get(name, 0.0) + v
        srcs = {}
        for _, names, sources in models:
            srcs.update(dict(zip(names, sources)))
        evaluate.importance_by_view(gains, srcs, spec.col_view).to_csv(
            os.path.join(results_dir, "importance", job_id + ".csv"), index=False)
    row = dict(j, job_id=job_id, status="done", runtime_s=round(time.time() - t0, 1),
               n_train_prefixes=len(tr.meta), n_test_prefixes=len(te.meta),
               n_train_segments=tr.meta["segment_id"].nunique(), n_test_segments=te.meta["segment_id"].nunique(),
               max_prefix_len=L, split_ts=str(split_ts), params_source=params_source, flags=flags,
               finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    row.update(evaluate.overall_metrics(out, task_type))
    if task_type == "regression":
        bl = evaluate.regression_baselines(tr.meta, te.meta)
        row["baseline_mae_overall"] = bl["baseline_overall"]["mae"]
        row["baseline_mae_per_prefix"] = bl["baseline_per_prefix"]["mae"]
    row.pop("n", None)
    append_run(results_dir, row)
    return row


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--flags", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.job_id.startswith("tune__"):
        from court import tune
        tune.run_tuning(a.job_id, cfg, a.data_dir, a.results_dir, a.sample_frac, a.train_frac, a.flags)
    else:
        run_job(a.job_id, cfg, a.data_dir, a.results_dir, a.sample_frac, a.train_frac, a.flags)


if __name__ == "__main__":
    main()
