"""Metrics (C4, D10), paired bootstrap (C11), importance by view (C12), baselines (C13), drift.

Prediction frames follow C10: columns ``segment_id, prefix_len, y_true, y_pred`` where
``prefix_len`` is the in-phase prefix length j (prefix_nr). For regression ``y_true`` and
``y_pred`` are in days (``y_pred`` already back-transformed with ``expm1``).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error, r2_score,
                             roc_auc_score)

N_BOOT = 1000
BOOT_SEED = 22


def _auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def clf_metrics(y, p):
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    return {"auc": _auc(y, p),
            "f1": f1_score(y, (p >= 0.5).astype(int), zero_division=0) if len(y) else np.nan,
            "pr_auc": average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan,
            "n": len(y)}


def reg_metrics(y_days, p_days):
    y = np.asarray(y_days, float); p = np.asarray(p_days, float)
    if not len(y):
        return {"mae": np.nan, "median_ae": np.nan, "r2_log": np.nan, "n": 0}
    return {"mae": mean_absolute_error(y, p),
            "median_ae": float(np.median(np.abs(y - p))),
            "r2_log": r2_score(np.log1p(np.maximum(y, 0)), np.log1p(np.maximum(p, 0))) if len(y) > 1 else np.nan,
            "n": len(y)}


def overall_metrics(pred, task_type):
    """Overall = pooled over all test prefixes (prefix-weighted, as in the paper)."""
    f = clf_metrics if task_type == "classification" else reg_metrics
    return f(pred["y_true"], pred["y_pred"])


def metrics_by_prefix(pred, task_type):
    f = clf_metrics if task_type == "classification" else reg_metrics
    rows = []
    for j, g in pred.groupby("prefix_len"):
        r = f(g["y_true"], g["y_pred"]); r["prefix_len"] = j; rows.append(r)
    return pd.DataFrame(rows)


def metrics_by_year(pred, year_of_segment, task_type):
    """Drift check: metrics by filing_year of the segment (Tier 3)."""
    f = clf_metrics if task_type == "classification" else reg_metrics
    yr = pred["segment_id"].map(year_of_segment)
    rows = []
    for y, g in pred.groupby(yr):
        r = f(g["y_true"], g["y_pred"]); r["filing_year"] = y; rows.append(r)
    return pd.DataFrame(rows)


def primary_metric(task_type):
    return "auc" if task_type == "classification" else "mae"


def paired_bootstrap(pred_x, pred_base, task_type, n_boot=N_BOOT, seed=BOOT_SEED):
    """Δ = metric(FS_x) − metric(base) with a 95% percentile CI; resampling unit = test segment.

    Both frames must hold the same (segment_id, prefix_len) rows (same test set)."""
    key = ["segment_id", "prefix_len"]
    m = pred_x[key + ["y_true", "y_pred"]].merge(pred_base[key + ["y_pred"]], on=key, suffixes=("_x", "_b"))
    if len(m) != len(pred_x) or len(m) != len(pred_base):
        raise ValueError("prediction frames do not cover the same test prefixes")
    metric = primary_metric(task_type)
    f = clf_metrics if task_type == "classification" else reg_metrics
    point = f(m["y_true"], m["y_pred_x"])[metric] - f(m["y_true"], m["y_pred_b"])[metric]
    segs, inv = np.unique(m["segment_id"].to_numpy(), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(segs) + 1))
    y = m["y_true"].to_numpy()[order]; px = m["y_pred_x"].to_numpy()[order]; pb = m["y_pred_b"].to_numpy()[order]
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(segs), len(segs))
        st = bounds[pick]; ln = bounds[pick + 1] - st
        idx = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln) + np.repeat(st, ln)
        if metric == "auc":
            deltas[b] = _auc(y[idx], px[idx]) - _auc(y[idx], pb[idx])
        else:
            deltas[b] = np.mean(np.abs(y[idx] - px[idx])) - np.mean(np.abs(y[idx] - pb[idx]))
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return {"metric": metric, "delta": point, "ci_low": lo, "ci_high": hi, "n_boot": n_boot,
            "n_segments": len(segs)}


def importance_by_view(gain_by_feature, feature_sources, col_view):
    """C12: XGBoost gain summed per view. ``gain_by_feature``: {feature name: total gain}."""
    from court.feature_sets import view_label
    src = dict(feature_sources)
    out = {}
    for feat, gain in gain_by_feature.items():
        view = view_label(col_view.get(src.get(feat), "?"), src.get(feat))
        out[view] = out.get(view, 0.0) + float(gain)
    tot = sum(out.values()) or 1.0
    return pd.DataFrame([{"view": v, "gain": g, "share": g / tot} for v, g in sorted(out.items())])


def regression_baselines(train_meta, test_meta):
    """C13: training median of remaining days, (i) overall and (ii) per in-phase prefix length.

    ``*_meta``: frames with ``prefix_nr`` and ``y_reg`` (days). Lengths unseen in train fall back
    to the largest train length below (as prefix bucketing does), else the overall median."""
    overall = float(np.median(train_meta["y_reg"]))
    per = train_meta.groupby("prefix_nr")["y_reg"].median()
    known = np.sort(per.index.to_numpy())

    def per_len(j):
        k = known[known <= j]
        return float(per[k[-1]]) if len(k) else overall
    y = test_meta["y_reg"].to_numpy()
    p_overall = np.full(len(y), overall)
    p_per = np.array([per_len(j) for j in test_meta["prefix_nr"]])
    return {"baseline_overall": reg_metrics(y, p_overall), "baseline_per_prefix": reg_metrics(y, p_per),
            "pred_overall": p_overall, "pred_per_prefix": p_per}
