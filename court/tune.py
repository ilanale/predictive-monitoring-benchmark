"""Hyperparameter tuning (mirrors upstream ``optimize_params.py``; C3/C4/C5).

* hyperopt TPE, upstream search space (xgboost: learning_rate U(0,1), subsample U(0.5,1),
  max_depth qU(4,30,1), colsample_bytree U(0.5,1), min_child_weight qU(1,6,1); rf: max_features
  U(0,1)); ``n_iter`` from config.
* 3-fold CV over *segments* of the training part of the strict split: StratifiedKFold by label
  (classification) / KFold (regression), shuffle, random_state 22, as upstream.
* loss: -mean AUC (pooled over buckets) / +mean MAE in days.
* sample cap: <= ``max_train_segments`` training segments, evenly spaced in time order.
* tuned on ``tuned_fs`` (FS-ALL) and reused for every FS of the same (task, method).
* prefix_agg: per-bucket best params = the trial with the best mean per-bucket score (what the
  upstream extraction notebook does with the all-trials file); stored under ``per_bucket``.

Deviation (logged): CV prefixes use the training gap (config ``tuning.gap``, default 3)
instead of upstream gap 1, to fit the 20-min cap.
"""
import json
import os
import time

import numpy as np

from court import data_io, evaluate, runner
from court.fast_pipeline import FastEncoder, prepare_log
from court.feature_sets import get_spec


def search_space(learner):
    from hyperopt import hp
    from hyperopt.pyll.base import scope
    if learner == "rf":
        return {"max_features": hp.uniform("max_features", 0, 1)}
    return {"learning_rate": hp.uniform("learning_rate", 0, 1),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "max_depth": scope.int(hp.quniform("max_depth", 4, 30, 1)),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "min_child_weight": scope.int(hp.quniform("min_child_weight", 1, 6, 1))}


def cv_folds(seg_ids, seg_labels, task_type, n_splits=3, seed=22):
    from sklearn.model_selection import KFold, StratifiedKFold
    if task_type == "classification":
        spl = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(seg_ids, seg_labels)
    else:
        spl = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(seg_ids)
    return [(set(seg_ids[a]), set(seg_ids[b])) for a, b in spl]


def run_tuning(job_id, cfg, data_dir, results_dir, sample_frac=None, train_frac=None, flags=""):
    from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
    t0 = time.time()
    _, task, method, learner = job_id.split("__")
    _, task_type = data_io.parse_task(task)
    b = cfg.get("budget", {})
    tcfg = b.get("tuning", {})
    n_iter = int(tcfg.get("n_iter", 10))
    n_splits = int(tcfg.get("cv_folds", 3))
    cap = int(tcfg.get("max_train_segments", 20000))
    if train_frac:
        cap = int(cap * train_frac)
        flags = (flags + ";" if flags else "") + "train_frac=%s" % train_frac
    gap = int(tcfg.get("gap", b.get("gap_train", 3)))
    n_jobs = int(b.get("xgb_n_jobs", 4))
    fs = cfg.get("tuned_fs", "FS-ALL")

    df = data_io.load_task_log(task, data_dir, sample_frac=sample_frac)
    cols = data_io.load_columns(data_dir)
    spec = get_spec(fs, cols, df.columns)
    from court import upstream
    dm = upstream.dataset_manager(task, fs, data_dir)
    L = min(int(b.get("max_prefix_cap", 40)), dm.get_pos_case_length_quantile(df, 0.90))  # C7
    train, test = dm.split_data_strict(df, 0.8, split="temporal")
    split_ts = data_io.segment_start_times(test).min() if len(test) else None
    del test
    start = data_io.segment_start_times(train).sort_values(kind="mergesort")
    keep = runner.systematic_segments(start.index.to_numpy(), cap=cap)
    train = prepare_log(train[train["segment_id"].isin(set(keep))])
    seg_lab = train.groupby("segment_id", sort=True)["label"].first()
    folds = cv_folds(seg_lab.index.to_numpy(), seg_lab.to_numpy(), task_type, n_splits)
    enc = FastEncoder(spec)
    enc.build_vocab(train)
    allp = enc.encode(train, L, gap, prepared=True)
    seg_of = allp.meta["segment_id"].to_numpy()
    fold_data = []
    for tr_ids, te_ids in folds:
        tr = allp.subset_rows(np.isin(seg_of, list(tr_ids)))
        te = allp.subset_rows(np.isin(seg_of, list(te_ids)))
        tr_events = train[train["segment_id"].isin(tr_ids)]
        if task_type == "classification":
            fb = float((tr_events["label"] == "deviant").mean())
        else:
            fb = float(np.log1p(np.median(tr.meta["y_reg"])))
        fold_data.append((tr, te, fb))
    trials_log = []

    def objective(params):
        scores, bucket_scores = [], {}
        for tr, te, fb in fold_data:
            pred, _ = runner.fit_predict_buckets(tr, te, method, learner, task_type, params, n_jobs, fb)
            if task_type == "classification":
                y = te.meta["y_clf"].to_numpy()
                scores.append(evaluate._auc(y, pred))
            else:
                y = te.meta["y_reg"].to_numpy(); pred = np.expm1(pred)
                scores.append(float(np.mean(np.abs(y - pred))))
            if method.startswith("prefix"):
                _, b_te = runner.bucket_of(method, tr.meta, te.meta)
                for bk in np.unique(b_te):
                    m = b_te == bk
                    if task_type == "classification":
                        s = evaluate._auc(y[m], pred[m]); s = 0.5 if np.isnan(s) else s
                    else:
                        s = float(np.mean(np.abs(y[m] - pred[m])))
                    bucket_scores.setdefault(int(bk), []).append(s)
        score = float(np.nanmean(scores))
        loss = -score if task_type == "classification" else score
        trials_log.append({"params": {k: (int(v) if k in ("max_depth", "min_child_weight") else float(v))
                                      for k, v in params.items()},
                           "score": score, "bucket_scores": {k: float(np.mean(v)) for k, v in bucket_scores.items()}})
        return {"loss": loss, "status": STATUS_OK}

    space = search_space(learner)
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=n_iter, trials=trials,
                rstate=np.random.default_rng(22), show_progressbar=False)
    best_params = space_eval(space, best)
    best_params = {k: (int(v) if k in ("max_depth", "min_child_weight") else float(v)) for k, v in best_params.items()}
    out = {"job_id": job_id, "tuned_fs": fs, "best": best_params, "max_prefix_len": L, "gap": gap,
           "n_train_segments": int(len(keep)), "n_iter": n_iter, "trials": trials_log}
    if method.startswith("prefix"):
        better = max if task_type == "classification" else min
        per = {}
        for bk in sorted({k for t in trials_log for k in t["bucket_scores"]}):
            cand = [t for t in trials_log if bk in t["bucket_scores"]]
            per[str(bk)] = better(cand, key=lambda t: t["bucket_scores"][bk])["params"]
        out["per_bucket"] = per
    os.makedirs(os.path.join(results_dir, "params"), exist_ok=True)
    with open(os.path.join(results_dir, "params", job_id + ".json"), "w") as f:
        json.dump(out, f, indent=1)
    metric = "auc" if task_type == "classification" else "mae"
    row = {"job_id": job_id, "tier": "tune", "task": task, "method": method, "fs": fs, "learner": learner,
           "status": "done", "runtime_s": round(time.time() - t0, 1), "n_train_prefixes": len(allp.meta),
           "n_train_segments": int(len(keep)), "max_prefix_len": L, "split_ts": str(split_ts),
           metric: max(t["score"] for t in trials_log) if metric == "auc" else min(t["score"] for t in trials_log),
           "params_source": "cv%d" % n_splits, "flags": flags, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    runner.append_run(results_dir, row)
    return out
