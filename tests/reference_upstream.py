"""Literal re-statement of the upstream prefix log + encoders, used only to test the fast path.

Follows ``DatasetManager.split_data_strict`` sorting, ``generate_prefix_data`` (with the C6
offset n_hist: lengths n_hist + j, prefix_nr = j), ``StaticTransformer`` and
``AggregateTransformer`` (+ the C8 last value of G columns), FeatureUnion order static | agg | last.
Deliberately slow and simple: head(n) per segment, then groupby.
"""
import numpy as np
import pandas as pd

from court.fast_pipeline import STATS, agg_feature_names


def generate_prefix_data(data, max_length, gap=1, case_col="segment_id"):
    data = data.sort_values(["timestamp", "Activity"], ascending=True, kind="mergesort").copy()
    data["case_length"] = data.groupby(case_col)["Activity"].transform(len)
    data["pos"] = data.groupby(case_col).cumcount()  # 0-based position in the segment
    parts = []
    for j in range(1, max_length + 1, gap):
        n_events = data["n_hist"] + j
        tmp = data[(data["case_length"] >= n_events) & (data["pos"] < n_events)].copy()
        tmp["orig_case_id"] = tmp[case_col]
        if j > 1:
            tmp[case_col] = tmp[case_col] + "_" + (tmp["n_hist"] + j).astype(str)
        tmp["prefix_nr"] = j
        parts.append(tmp)
    return pd.concat(parts, axis=0)


def encode(prefixes, spec, layout="stat_major", case_col="segment_id"):
    X = prefixes
    # StaticTransformer
    first = X.groupby(case_col).first()
    st = first[spec.static_num]
    if spec.static_cat:
        st = pd.concat([st, pd.get_dummies(first[spec.static_cat])], axis=1)
    st = st.fillna(0)
    # AggregateTransformer (boolean=False)
    # FS-M has no dynamic columns: upstream would use the "static" encoder alone
    if spec.dynamic_cat:
        dummies = pd.get_dummies(X[spec.dynamic_cat])
        dummies[case_col] = X[case_col]
        ag = dummies.groupby(case_col).sum()
    else:
        ag = pd.DataFrame(index=st.index)
    if spec.dynamic_num:
        g = X.groupby(case_col)[spec.dynamic_num]
        stats = {"mean": g.mean(), "max": g.max(), "min": g.min(), "sum": g.sum(), "std": g.std()}
        num = pd.concat([stats[s][c].rename(name) for name, c, s in agg_feature_names(spec.dynamic_num, layout)],
                        axis=1)
        ag = pd.concat([ag, num], axis=1)
    ag = ag.fillna(0)
    parts = [st, ag]
    if spec.last_cols:
        last = X.groupby(case_col)[spec.last_cols].last().fillna(0)
        last.columns = ["last_%s" % c for c in spec.last_cols]
        parts.append(last)
    out = pd.concat(parts, axis=1).astype(float)
    labels_reg = X.groupby(case_col)["remaining_in_phase"].last()
    prefix_nr = X.groupby(case_col)["prefix_nr"].last()
    return out, labels_reg, prefix_nr


_ = (np, STATS)
