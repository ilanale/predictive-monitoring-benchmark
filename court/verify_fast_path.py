"""C9 / gate G3: the fast path equals the (adapted) upstream path on a segment sample.

For each task and feature set: write a sample of N segments in the benchmark CSV format, read it
with upstream ``DatasetManager.read_dataset``, then compare upstream (split_data_strict ->
generate_prefix_data -> BucketFactory -> FeatureUnion(static, agg[, lastG]) -> XGBoost) with the
court fast path (court.runner.build_matrices -> bucket_of -> fit_columns -> XGBoost):
split sets, L, prefix ids/order, labels, feature names, feature values (max abs diff) and
predictions with the same hyperparameters (max abs diff; 0 = identical).

    PYTHONPATH=$PWD:$PWD/experiments python -m court.verify_fast_path --data-dir data/processed \
        --n-segments 1000 --out results/verify_fast_path.json
Must run in its own process (upstream modules read PPM_COURT_DATA_DIR at import time).
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXED = {"learning_rate": 0.1, "subsample": 0.8, "max_depth": 6, "colsample_bytree": 0.8, "min_child_weight": 2}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--tasks", default="court_p1_clf,court_p1_reg,court_p2_clf,court_p2_reg,court_p3_clf,court_p3_reg")
    ap.add_argument("--fs", default="FS-ALL")
    ap.add_argument("--methods", default="single_agg,prefix_agg")
    ap.add_argument("--n-segments", type=int, default=1000)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--max-buckets", type=int, default=3, help="prefix_agg: compare this many buckets")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    tmp = tempfile.mkdtemp(prefix="verify_fast_")
    data_dir = os.path.abspath(a.data_dir)
    # sample segments of every phase file into tmp (benchmark CSV format), same columns json
    from court import data_io
    tasks = a.tasks.split(",")
    for phase in sorted({data_io.parse_task(t)[0] for t in tasks}):
        df = data_io.load_task_log("court_p%d_clf" % phase, data_dir)
        segs = df["segment_id"].drop_duplicates()
        keep = segs.sample(n=min(a.n_segments, len(segs)), random_state=22)
        sub = df[df["segment_id"].isin(set(keep))].copy()
        sub["timestamp"] = sub["timestamp"].dt.strftime("%Y-%m-%d")
        sub.to_csv(os.path.join(tmp, "court_phase%d.csv" % phase), sep=";", index=False)
    with open(os.path.join(data_dir, "court_columns.json")) as f:
        cols_json = json.load(f)
    with open(os.path.join(tmp, "court_columns.json"), "w") as f:
        json.dump(cols_json, f)
    os.environ["PPM_COURT_DATA_DIR"] = tmp
    for p in (ROOT, os.path.join(ROOT, "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import xgboost as xgb  # noqa: F401
    from sklearn.pipeline import FeatureUnion
    import BucketFactory
    import EncoderFactory
    from DatasetManager import DatasetManager
    from court import runner
    from court.fast_pipeline import fit_columns
    from court.feature_sets import get_spec

    report = []
    for task in tasks:
        _, task_type = data_io.parse_task(task)
        for fs in a.fs.split(","):
            ds = "%s__%s" % (task, fs)
            t0 = time.time()
            dm = DatasetManager(ds)
            data = dm.read_dataset()
            rec = {"task": task, "fs": fs, "n_segments": int(data["segment_id"].nunique())}
            # upstream path
            L_up = min(40, dm.get_pos_case_length_quantile(data, 0.90))
            train, test = dm.split_data_strict(data, 0.8, split="temporal")
            dt_train = dm.generate_prefix_data(train, 1, L_up, 3)
            dt_test = dm.generate_prefix_data(test, 1, L_up)
            # fast path on the very same parsed frame
            spec = get_spec(fs, cols_json, data.columns)
            L_fast = data_io.max_prefix_length(data, task_type)
            tr, te, split_ts, _ = runner.build_matrices(data, task_type, spec, L_fast, 3, dtype=np.float64)
            rec["L_equal"] = bool(L_up == L_fast); rec["L"] = int(L_up)
            f_tr, f_te, _ = data_io.split_data_strict(data)
            rec["split_equal"] = bool(set(train.segment_id) == set(f_tr.segment_id) and
                                      set(test.segment_id) == set(f_te.segment_id) and len(train) == len(f_tr))
            methods = ["static", "agg"] + (["lastG"] if dm.global_cols else [])
            if not (dm.dynamic_cat_cols or dm.dynamic_num_cols):
                methods = ["static"]
            enc_args = {"case_id_col": dm.case_id_col, "static_cat_cols": dm.static_cat_cols,
                        "static_num_cols": dm.static_num_cols, "dynamic_cat_cols": dm.dynamic_cat_cols,
                        "dynamic_num_cols": dm.dynamic_num_cols, "fillna": True, "global_cols": dm.global_cols}
            rec["methods"] = {}
            for method in a.methods.split(","):
                bm = "single" if method.startswith("single") else "prefix"
                bucketer = BucketFactory.get_bucketer(bm, encoding_method="agg", case_id_col=dm.case_id_col,
                                                      cat_cols=[dm.activity_col], num_cols=[], random_state=22)
                b_train = bucketer.fit_predict(dt_train)
                b_test = bucketer.predict(dt_test)
                fb_tr, fb_te = runner.bucket_of(method, tr.meta, te.meta)
                m = {"bucket_assign_equal": bool(np.array_equal(np.asarray(b_test), fb_te) and
                                                 np.array_equal(np.asarray(b_train), fb_tr)),
                     "buckets": []}
                for bucket in sorted(set(b_test))[: a.max_buckets]:
                    tr_ids = dm.get_indexes(dt_train)[b_train == bucket]
                    te_ids = dm.get_indexes(dt_test)[b_test == bucket]
                    if len(tr_ids) == 0:
                        continue
                    dt_tr = dm.get_relevant_data_by_indexes(dt_train, tr_ids)
                    dt_te = dm.get_relevant_data_by_indexes(dt_test, te_ids)
                    y_up = np.asarray(dm.get_label_numeric(dt_tr))
                    union = FeatureUnion([(mm, EncoderFactory.get_encoder(mm, **enc_args)) for mm in methods])
                    X_up_tr = union.fit_transform(dt_tr)
                    X_up_te = union.transform(dt_te)
                    names_up = [c for _, t in union.transformer_list for c in t.columns]
                    trb = tr.subset_rows(np.flatnonzero(fb_tr == bucket))
                    teb = te.subset_rows(np.flatnonzero(fb_te == bucket))
                    keep = fit_columns(trb)
                    trb, teb = trb.subset_cols(keep), teb.subset_cols(keep)
                    y_fast = runner.target(trb.meta, task_type)
                    b = {"bucket": int(bucket), "n_train": int(len(y_up)), "n_test": int(len(te_ids)),
                         "names_equal": names_up == trb.names,
                         "train_ids_equal": list(dt_tr.groupby(dm.case_id_col).first().index) ==
                                            list(trb.meta["prefix_case_id"]),
                         "test_ids_equal": list(dt_te.groupby(dm.case_id_col).first().index) ==
                                           list(teb.meta["prefix_case_id"]),
                         "labels_equal": bool(np.allclose(y_up, y_fast))}
                    if b["names_equal"] and b["train_ids_equal"] and b["test_ids_equal"]:
                        b["max_abs_diff_train"] = float(np.abs(np.asarray(X_up_tr, float) - trb.X.astype(float)).max())
                        b["max_abs_diff_test"] = float(np.abs(np.asarray(X_up_te, float) - teb.X.astype(float)).max())
                        b["max_rel_diff_train"] = float((np.abs(np.asarray(X_up_tr, float) - trb.X.astype(float)) /
                                                         np.maximum(1, np.abs(np.asarray(X_up_tr, float)))).max())
                        if len(set(y_up)) > 1:
                            p = dict(FIXED)
                            m_up = runner.make_learner("xgboost", task_type, p, a.n_jobs)
                            m_up.set_params(n_estimators=a.n_estimators)
                            m_up.fit(np.asarray(X_up_tr, dtype=np.float32), y_up)
                            m_f = runner.make_learner("xgboost", task_type, p, a.n_jobs)
                            m_f.set_params(n_estimators=a.n_estimators)
                            m_f.fit(trb.X.astype(np.float32), y_fast)
                            if task_type == "classification":
                                pu = m_up.predict_proba(np.asarray(X_up_te, dtype=np.float32))[:, 1]
                                pf = m_f.predict_proba(teb.X.astype(np.float32))[:, 1]
                            else:
                                pu = m_up.predict(np.asarray(X_up_te, dtype=np.float32)); pf = m_f.predict(teb.X.astype(np.float32))
                            b["max_abs_pred_diff"] = float(np.abs(pu - pf).max()) if len(pu) else 0.0
                    m["buckets"].append(b)
                rec["methods"][method] = m
            rec["seconds"] = round(time.time() - t0, 1)
            print(json.dumps(rec), flush=True)
            report.append(rec)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    main()
