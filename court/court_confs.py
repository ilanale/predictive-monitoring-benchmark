"""Register the court tasks (3 phases x {clf, reg}) x 6 feature sets into the upstream
`experiments/dataset_confs` dictionaries, from data/processed/court_columns.json.

Names: court_p{k}_{clf|reg}__{FS}, e.g. court_p1_clf__FS-B0, court_p1_reg__FS-ALL.
Optionally the D8 sensitivity variants court_p{k}_{task}__{FS}+S.

Usage (A3 wires this in; upstream code is not edited here):
    import dataset_confs
    from court.court_confs import register
    register(dataset_confs, "data/processed/court_columns.json", "data/processed")
"""
import json
import os

FEATURE_SETS = {
    "FS-M": ["M"],
    "FS-MP": ["M", "P", "PP"],
    "FS-B0": ["M", "P", "B0G"],
    "FS-B0+P": ["M", "P", "PP", "B0G"],
    "FS-B0+G": ["M", "P", "B0G", "G"],
    "FS-ALL": ["M", "P", "PP", "B0G", "G"],
}
TASKS = {"clf": "classification", "reg": "regression"}


def feature_columns(spec, views):
    cols = [c for v in views for c in spec["views"][v]]
    seen = set(); cols = [c for c in cols if not (c in seen or seen.add(c))]
    out = {k: [c for c in cols if c in set(spec[k])]
           for k in ("dynamic_cat", "static_cat", "dynamic_num", "static_num")}
    assert sum(len(v) for v in out.values()) == len(cols), "column without a role"
    return out


def dataset_names(with_sensitivity=False):
    names = []
    for k in (1, 2, 3):
        for t in TASKS:
            for fs in FEATURE_SETS:
                names.append(f"court_p{k}_{t}__{fs}")
                if with_sensitivity:
                    names.append(f"court_p{k}_{t}__{fs}+S")
    return names


def register(dc, columns_json="data/processed/court_columns.json", data_dir="data/processed",
             with_sensitivity=False, fmt="csv"):
    """Fill dc.<dict>[dataset] for every court dataset; returns the list of names.

    Adds (creating if absent) the extra dicts `task_type`, `regression_label_col`,
    `global_cols` (G-view columns, for the last-value encoder C8) and `feature_set`.
    """
    with open(columns_json) as f:
        spec = json.load(f)
    from .data_prep.export import assert_no_leaky_features
    assert_no_leaky_features(spec)
    for extra in ("task_type", "regression_label_col", "global_cols", "feature_set"):
        if not hasattr(dc, extra):
            setattr(dc, extra, {})
    names = []
    for k in (1, 2, 3):
        for t, ttype in TASKS.items():
            for fs, views in FEATURE_SETS.items():
                variants = [(fs, views)] + ([(fs + "+S", views + ["S"])] if with_sensitivity else [])
                for fs_name, vv in variants:
                    ds = f"court_p{k}_{t}__{fs_name}"
                    cols = feature_columns(spec, vv)
                    dc.filename[ds] = os.path.join(data_dir, f"court_phase{k}.{fmt}")
                    dc.case_id_col[ds] = spec["case_id_col"]
                    dc.activity_col[ds] = spec["activity_col"]
                    dc.resource_col[ds] = spec["resource_col"]
                    dc.timestamp_col[ds] = spec["timestamp_col"]
                    dc.label_col[ds] = spec["label_col"]
                    dc.pos_label[ds] = "deviant"
                    dc.neg_label[ds] = "regular"
                    dc.dynamic_cat_cols[ds] = cols["dynamic_cat"]
                    dc.static_cat_cols[ds] = cols["static_cat"]
                    dc.dynamic_num_cols[ds] = cols["dynamic_num"]
                    dc.static_num_cols[ds] = cols["static_num"]
                    dc.task_type[ds] = ttype
                    dc.regression_label_col[ds] = spec["regression_label_col"]
                    dc.global_cols[ds] = [c for c in spec["views"]["G"] if "G" in vv]
                    dc.feature_set[ds] = fs_name
                    names.append(ds)
    return names
