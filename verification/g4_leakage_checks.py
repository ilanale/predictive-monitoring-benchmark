"""G4 (A4): leakage and alignment checks on small real samples (light load: 2 threads, < 4 GB).

    PYTHONPATH=$PWD:$PWD/experiments python verification/g4_leakage_checks.py \
        --data-dir <repo>/data/processed --results-dir <repo>/results --out verification/g4_results.json

Checks
  1 feature matrix columns per FS (forbidden names/sources, last_ only with G+)
  2 D7 perturbation: changing every event after a cut point (all columns, incl. labels) leaves
    the features of every prefix ending at or before the cut bit-identical
  3 targets never reach X: randomising label / remaining_in_phase leaves X identical
  4 strict temporal split on the full logs (upstream split_data_strict, C6-adapted)
  5 saved predictions: (segment_id, prefix_len) == test prefixes of the split; y_true recomputed
    from the raw log; y_pred range (expm1 once)
  6 G drift: train vs test distribution of G columns (in-phase rows)
"""
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from court import data_io
from court.fast_pipeline import FastEncoder, prepare_log
from court.feature_sets import ALL_FS, forbidden_set, get_spec, is_forbidden

RNG = np.random.default_rng(4)
BAD_TOKENS = ["remaining_in_phase", "label", "in_phase", "n_hist", "entry_date", "exit_date", "filing_year",
              "ucid", "segment_id", "timestamp"]


def sample_log(data_dir, phase, n_seg, seed=4):
    path = os.path.join(data_dir, "court_phase%d.parquet" % phase)
    segs = pq.read_table(path, columns=["segment_id"]).column(0).to_pandas().drop_duplicates()
    keep = segs.sample(n=min(n_seg, len(segs)), random_state=seed).tolist()
    df = pq.read_table(path, filters=[("segment_id", "in", keep)]).to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def check_columns(df, cols):
    out = {}
    forb = forbidden_set(cols)
    for fs in ALL_FS + ["FS-ALL+S"]:
        spec = get_spec(fs, cols, df.columns)
        enc = FastEncoder(spec)
        pm = enc.encode(df, 40, 1)
        bad_src = sorted({s for s in pm.sources if s is not None and is_forbidden(s, forb)})
        bad_name = sorted(n for n in pm.names if any(n == t or n.startswith(t + "_") or n.endswith("_" + t)
                                                      for t in BAD_TOKENS))
        n_last = sum(n.startswith("last_") for n in pm.names)
        has_g = "G" in spec.col_view.values()
        last_ok = (n_last > 0) == has_g and all(spec.col_view[s] == "G" for n, s in zip(pm.names, pm.sources)
                                                 if n.startswith("last_"))
        out[fs] = {"n_cols": len(pm.names), "bad_sources": bad_src, "bad_names": bad_name, "n_last": n_last,
                   "views": sorted(set(spec.col_view.values())), "last_only_G": bool(last_ok),
                   "S_cols": sum(1 for s in pm.sources if s and s.startswith("S_")),
                   "pass": not bad_src and not bad_name and last_ok and (fs.endswith("+S") or
                                                                         not any(s and s.startswith("S_") for s in pm.sources))}
    return out


def check_perturbation(df, cols, fs="FS-ALL+S"):
    spec = get_spec(fs, cols, df.columns)
    d0 = prepare_log(df)
    starts, ends = data_io_bounds(d0)
    h = d0["n_hist"].to_numpy()[starts].astype(int)
    cut = np.empty(len(starts), int)  # last kept row (absolute) per segment
    for i, (s, e) in enumerate(zip(starts, ends)):
        lo, hi = s + h[i], e - 1
        cut[i] = RNG.integers(lo, hi + 1) if hi >= lo else e - 1
    d1 = d0.copy()
    after = np.zeros(len(d0), bool)
    for i, (s, e) in enumerate(zip(starts, ends)):
        after[cut[i] + 1:e] = True
    idx = np.flatnonzero(after)
    for c in spec.dynamic_num + spec.static_num + ["remaining_in_phase"]:
        d1[c] = d1[c].astype(float)
        d1.loc[idx, c] = d1.loc[idx, c].to_numpy(float) * RNG.uniform(0.1, 10, len(idx)) + RNG.normal(0, 50, len(idx))
    for c in spec.dynamic_cat + spec.static_cat + ["label"]:
        vals = d0[c].dropna().unique()
        d1.loc[idx, c] = RNG.choice(vals, len(idx))
    enc = FastEncoder(spec); enc.build_vocab(d0)
    X0 = enc.encode(d0, 40, 1, prepared=True)
    X1 = enc.encode(d1, 40, 1, prepared=True)
    seg_cut = pd.Series(cut - starts + 1, index=d0["segment_id"].to_numpy()[starts])  # events kept
    kept = X0.meta["prefix_len"].to_numpy() <= seg_cut.reindex(X0.meta["segment_id"]).to_numpy()
    assert (X0.meta["prefix_case_id"].to_numpy() == X1.meta["prefix_case_id"].to_numpy()).all()
    diff = np.abs(X0.X.astype(float) - X1.X.astype(float)).max(axis=1)
    # targets only (labels randomised everywhere) -> X must be identical
    d2 = d0.copy()
    d2["label"] = RNG.choice(["deviant", "regular"], len(d2))
    d2["remaining_in_phase"] = RNG.integers(0, 5000, len(d2))
    X2 = enc.encode(d2, 40, 1, prepared=True)
    return {"fs": fs, "n_segments": int(len(starts)), "n_prefixes": int(len(diff)),
            "n_prefixes_before_cut": int(kept.sum()),
            "max_abs_diff_before_cut": float(diff[kept].max()) if kept.any() else None,
            "share_changed_after_cut": float((diff[~kept] > 0).mean()) if (~kept).any() else None,
            "targets_randomised_max_abs_diff": float(np.abs(X0.X - X2.X).max()),
            "y_reg_changed_by_target_randomisation": bool((X0.meta["y_reg"] != X2.meta["y_reg"]).any()),
            "pass": bool(kept.any() and diff[kept].max() == 0 and np.abs(X0.X - X2.X).max() == 0
                         and (diff[~kept] > 0).mean() > 0.5)}


def data_io_bounds(d):
    from court.fast_pipeline import segment_bounds
    return segment_bounds(d)


def load_min(data_dir, phase, extra=()):
    cols = ["segment_id", "timestamp", "Activity", "in_phase", "n_hist", "label", "remaining_in_phase"] + list(extra)
    df = pd.read_parquet(os.path.join(data_dir, "court_phase%d.parquet" % phase), columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def check_split_and_predictions(data_dir, results_dir, phase, gcols):
    from court import upstream
    df = load_min(data_dir, phase, gcols)
    res = {}
    for kind in ["clf", "reg"]:
        task = "court_p%d_%s" % (phase, kind)
        dm = upstream.dataset_manager(task, "FS-B0", data_dir)
        train, test = dm.split_data_strict(df, 0.8, split="temporal")
        st_test = data_io.segment_start_times(test)
        st_train = data_io.segment_start_times(train)
        split_ts = st_test.min()
        r = {"n_train_segments": int(train["segment_id"].nunique()), "n_test_segments": int(test["segment_id"].nunique()),
             "overlap_segments": len(set(train["segment_id"]) & set(test["segment_id"])),
             "split_ts": str(split_ts), "max_train_event_ts": str(train["timestamp"].max()),
             "train_event_on_or_after_split": int((train["timestamp"] >= split_ts).sum()),
             "test_segments_starting_before_split": int((st_test < split_ts).sum()),
             "train_inphase_starts_after_split": int((st_train > split_ts).sum())}
        # same split as the court re-implementation
        tr2, te2, ts2 = data_io.split_data_strict(df)
        r["court_split_identical"] = bool(set(tr2["segment_id"]) == set(train["segment_id"]) and
                                          set(te2["segment_id"]) == set(test["segment_id"]) and ts2 == split_ts)
        # expected test prefixes and targets from the raw log
        tp = prepare_log(test)
        from court.fast_pipeline import prefix_ends
        L = min(40, dm.get_pos_case_length_quantile(df, 0.90))
        rows, j, h = prefix_ends(tp, L, 1)
        exp = pd.DataFrame({"segment_id": tp["segment_id"].to_numpy()[rows], "prefix_len": j,
                            "y_exp": (tp["label"].to_numpy()[rows] == "deviant").astype(float) if kind == "clf"
                            else tp["remaining_in_phase"].to_numpy()[rows].astype(float),
                            "in_phase_at_end": tp["in_phase"].to_numpy()[rows],
                            "ts_end": tp["timestamp"].to_numpy()[rows]})
        r["max_prefix_len"] = int(L)
        r["prefix_ends_all_in_phase"] = bool((exp["in_phase_at_end"] == 1).all())
        for fs in ["FS-B0", "FS-ALL"]:
            f = os.path.join(results_dir, "predictions", "tier1a__%s__single_agg__%s__xgboost.csv.gz" % (task, fs))
            if not os.path.exists(f):
                continue
            p = pd.read_csv(f)
            m = p.merge(exp, on=["segment_id", "prefix_len"], how="outer", indicator=True)
            q = {"rows": len(p), "expected_rows": len(exp), "only_in_pred": int((m["_merge"] == "left_only").sum()),
                 "only_expected": int((m["_merge"] == "right_only").sum()),
                 "y_true_mismatch": int((np.abs(m["y_true"] - m["y_exp"]) > 1e-6).sum()),
                 "y_pred_min": float(p["y_pred"].min()), "y_pred_median": float(p["y_pred"].median()),
                 "y_pred_max": float(p["y_pred"].max()), "y_true_median": float(p["y_true"].median())}
            if kind == "clf":
                from sklearn.metrics import roc_auc_score
                q["auc"] = float(roc_auc_score(p["y_true"], p["y_pred"]))
                q["pos_rate"] = float(p["y_true"].mean())
                q["mean_pred"] = float(p["y_pred"].mean())
            else:
                q["mae"] = float(np.mean(np.abs(p["y_true"] - p["y_pred"])))
                q["mae_if_pred_were_log_scale"] = float(np.mean(np.abs(p["y_true"] - np.log1p(p["y_pred"].clip(lower=0)))))
                q["mean_pred"] = float(p["y_pred"].mean()); q["mean_true"] = float(p["y_true"].mean())
                # MAE-optimal constant vs mean
                q["mae_const_test_median"] = float(np.mean(np.abs(p["y_true"] - p["y_true"].median())))
            r[fs] = q
        res[kind] = r
        if kind == "clf" and gcols:
            res["g_drift"] = g_drift(train, test, gcols, split_ts)
    return res


def g_drift(train, test, gcols, split_ts):
    tr = train[train["in_phase"] == 1]
    te = test[test["in_phase"] == 1]
    rows = []
    for c in gcols:
        a, b = tr[c].to_numpy(float), te[c].to_numpy(float)
        sd = np.nanstd(a) or 1.0
        rows.append({"col": c, "train_mean": float(np.nanmean(a)), "test_mean": float(np.nanmean(b)),
                     "smd": float((np.nanmean(b) - np.nanmean(a)) / sd),
                     "test_above_train_max": float(np.mean(b > np.nanmax(a))),
                     "test_below_train_min": float(np.mean(b < np.nanmin(a)))})
    rows.sort(key=lambda r: -abs(r["smd"]))
    # yearly mean of the most shifted columns (all in-phase rows, train+test)
    allr = pd.concat([tr, te])
    yr = allr["timestamp"].dt.year
    top = [r["col"] for r in rows[:4]] + ["open_cases"]
    by_year = allr.groupby(yr)[top].mean().round(1)
    return {"top": rows[:10], "by_year": {str(k): v for k, v in by_year.to_dict(orient="index").items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out", default="verification/g4_results.json")
    ap.add_argument("--n-seg", type=int, default=400)
    a = ap.parse_args()
    cols = data_io.load_columns(a.data_dir)
    out = {}
    for phase in (1, 2):
        df = sample_log(a.data_dir, phase, a.n_seg)
        out["columns_p%d" % phase] = check_columns(df, cols)
        out["perturbation_p%d" % phase] = check_perturbation(df, cols)
        del df
    for phase in (1, 2, 3):
        out["split_pred_p%d" % phase] = check_split_and_predictions(
            a.data_dir, a.results_dir, phase, cols["views"]["G"] + ["open_cases"] if phase < 3 else [])
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
