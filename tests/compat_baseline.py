"""Compute the compat baseline on the synthetic log (fixed hyperparameters, fixed seeds, no tuning).

Run as a script (the upstream modules read PPM_LOGS_DIR at import time, so this runs in its own process):
    python tests/compat_baseline.py <out_dir> [--logs-dir DIR]
Writes into <out_dir>:
  performance_results_xgboost_synthetic_{single,prefix}_agg.csv  - upstream experiments.py output (AUC rows are compared)
  features_single_agg.csv            - encoded test features of the first 200 test prefixes (single bucket)
  features_prefix_agg_b{1,5,10}.csv  - encoded test features of the first 70 test prefixes of buckets 1, 5, 10
  predictions_{single,prefix}_agg.csv - predicted P(deviant) for every test prefix
To refresh the committed fixture: python tests/compat_baseline.py tests/fixtures/compat_baseline
"""
import argparse
import os
import pickle
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

FIXED_XGB_PARAMS = {"learning_rate": 0.1, "subsample": 0.8, "max_depth": 6,
                    "colsample_bytree": 0.8, "min_child_weight": 2}
FLOAT_FMT = "%.17g"


def _env(logs_dir):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([REPO, os.path.join(REPO, "experiments")])
    env["PPM_LOGS_DIR"] = logs_dir
    env["PPM_N_JOBS"] = "4"
    env["PYTHONHASHSEED"] = "0"
    return env


def write_params(params_dir):
    os.makedirs(params_dir, exist_ok=True)
    with open(os.path.join(params_dir, "optimal_params_xgboost_synthetic_single_agg.pickle"), "wb") as f:
        pickle.dump(dict(FIXED_XGB_PARAMS), f)
    with open(os.path.join(params_dir, "optimal_params_xgboost_synthetic_prefix_agg.pickle"), "wb") as f:
        pickle.dump({k: dict(FIXED_XGB_PARAMS) for k in range(1, 41)}, f)


def run_upstream_experiments(params_dir, out_dir, logs_dir):
    for bucket in ["single", "prefix"]:
        subprocess.run([PY, os.path.join(REPO, "experiments", "experiments.py"), "synthetic", params_dir, out_dir,
                        bucket, "agg", "xgboost", "1", "1"], check=True, env=_env(logs_dir), cwd=REPO)


def encode_and_predict(out_dir):
    """Mirrors experiments.py (split, prefixes, bucketing, FeatureUnion + XGBClassifier) and dumps
    the encoded test features and the predictions. Must run with PPM_LOGS_DIR set (see main)."""
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from sklearn.pipeline import FeatureUnion, Pipeline
    import EncoderFactory
    import BucketFactory
    from DatasetManager import DatasetManager

    dm = DatasetManager("synthetic")
    data = dm.read_dataset()
    enc_args = {"case_id_col": dm.case_id_col, "static_cat_cols": dm.static_cat_cols,
                "static_num_cols": dm.static_num_cols, "dynamic_cat_cols": dm.dynamic_cat_cols,
                "dynamic_num_cols": dm.dynamic_num_cols, "fillna": True}
    max_len = min(40, dm.get_pos_case_length_quantile(data, 0.90))
    train, test = dm.split_data_strict(data, 0.8, split="temporal")
    dt_test = dm.generate_prefix_data(test, 1, max_len)
    dt_train = dm.generate_prefix_data(train, 1, max_len, 1)

    for bucket_method in ["single", "prefix"]:
        bucketer = BucketFactory.get_bucketer(bucket_method, encoding_method="agg", case_id_col=dm.case_id_col,
                                              cat_cols=[dm.activity_col], num_cols=[], random_state=22)
        b_train = bucketer.fit_predict(dt_train)
        b_test = bucketer.predict(dt_test)
        pred_rows = []
        for bucket in sorted(set(b_test)):
            tr_ids = dm.get_indexes(dt_train)[b_train == bucket]
            te_ids = dm.get_indexes(dt_test)[b_test == bucket]
            dt_tr = dm.get_relevant_data_by_indexes(dt_train, tr_ids)
            dt_te = dm.get_relevant_data_by_indexes(dt_test, te_ids)
            y_tr = dm.get_label_numeric(dt_tr)
            union = FeatureUnion([(m, EncoderFactory.get_encoder(m, **enc_args)) for m in ["static", "agg"]])
            cls = xgb.XGBClassifier(objective="binary:logistic", n_estimators=500, random_state=22, n_jobs=4,
                                    learning_rate=FIXED_XGB_PARAMS["learning_rate"],
                                    subsample=FIXED_XGB_PARAMS["subsample"],
                                    max_depth=int(FIXED_XGB_PARAMS["max_depth"]),
                                    colsample_bytree=FIXED_XGB_PARAMS["colsample_bytree"],
                                    min_child_weight=int(FIXED_XGB_PARAMS["min_child_weight"]))
            pipe = Pipeline([("encoder", union), ("cls", cls)])
            pipe.fit(dt_tr, y_tr)
            case_ids = list(dt_te.groupby(dm.case_id_col).first().index)
            pos = np.where(cls.classes_ == 1)[0][0]
            preds = pipe.predict_proba(dt_te)[:, pos]
            y_te = dm.get_label_numeric(dt_te)
            nr = list(dm.get_prefix_lengths(dt_te))
            for cid, n, y, p in zip(case_ids, nr, y_te, preds):
                pred_rows.append((bucket_method, int(bucket), cid, int(n), int(y), float(p)))

            names = [n_ for m, t in union.transformer_list for n_ in ["%s__%s" % (m, c) for c in t.columns]]
            if bucket_method == "single" or int(bucket) in (1, 5, 10):
                n_rows = 200 if bucket_method == "single" else 70
                X = union.transform(dt_te)
                feats = pd.DataFrame(np.asarray(X, dtype=float), columns=names)
                feats.insert(0, dm.case_id_col, case_ids)
                fname = "features_single_agg.csv" if bucket_method == "single" \
                    else "features_prefix_agg_b%d.csv" % int(bucket)
                feats.head(n_rows).to_csv(os.path.join(out_dir, fname), sep=";", index=False, float_format=FLOAT_FMT)
        pd.DataFrame(pred_rows, columns=["method", "bucket", "case_id", "nr_events", "actual", "predicted"]).to_csv(
            os.path.join(out_dir, "predictions_%s_agg.csv" % bucket_method), sep=";", index=False,
            float_format=FLOAT_FMT)


def main(out_dir, logs_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        if logs_dir is None:
            logs_dir = os.path.join(tmp, "logs")
            sys.path.insert(0, os.path.join(REPO, "tests", "fixtures"))
            import make_synthetic_log
            make_synthetic_log.main(logs_dir)
        params_dir = os.path.join(tmp, "params")
        write_params(params_dir)
        run_upstream_experiments(params_dir, out_dir, logs_dir)
        subprocess.run([PY, os.path.abspath(__file__), out_dir, "--logs-dir", logs_dir, "--encode-only"],
                       check=True, env=_env(logs_dir), cwd=REPO)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--logs-dir", default=None)
    ap.add_argument("--encode-only", action="store_true")
    a = ap.parse_args()
    if a.encode_only:
        encode_and_predict(a.out_dir)
    else:
        main(os.path.abspath(a.out_dir), a.logs_dir)
