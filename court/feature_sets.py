"""Feature-set (FS) definitions for the ablation (PLAN.md §1.2, court/SCHEMA.md).

A feature set is a union of *views*; each view is a list of raw columns taken from
``data/processed/court_columns.json``. Columns are then split by kind
(static/dynamic x cat/num) in the order they appear in the json kind lists, which
fixes the column order of the feature matrix (see fast_pipeline).

``+S`` suffix (e.g. ``FS-ALL+S``) adds the D8 sensitivity view S.
"""
import json
import os
from dataclasses import dataclass, field

FS_VIEWS = {
    "FS-M": ["M"],
    "FS-MP": ["M", "P", "PP"],          # RD-02
    "FS-B0": ["M", "P", "B0G"],
    "FS-B0+P": ["M", "P", "PP", "B0G"],
    "FS-B0+G": ["M", "P", "B0G", "G"],
    "FS-ALL": ["M", "P", "PP", "B0G", "G"],
}
ALL_FS = list(FS_VIEWS)
TIME_FEATURES = ["timesincelastevent", "timesincecasestart", "event_nr", "month", "weekday"]
# Columns that must never be features, whatever the json says (SCHEMA "never_features").
HARD_NEVER = ["segment_id", "ucid", "in_phase", "n_hist", "entry_date", "exit_date", "filing_year",
              "remaining_in_phase", "label", "timestamp"]


def load_columns(path):
    with open(path) as f:
        return json.load(f)


def views_of(fs):
    sens = fs.endswith("+S")
    base = fs[:-2] if sens else fs
    if base not in FS_VIEWS:
        raise KeyError("unknown feature set %r" % fs)
    return FS_VIEWS[base] + (["S"] if sens else [])


@dataclass
class FeatureSpec:
    fs: str
    static_cat: list = field(default_factory=list)
    static_num: list = field(default_factory=list)
    dynamic_cat: list = field(default_factory=list)
    dynamic_num: list = field(default_factory=list)
    last_cols: list = field(default_factory=list)      # C8: G columns also encoded by last value
    col_view: dict = field(default_factory=dict)        # raw column -> view name

    @property
    def all_cols(self):
        return self.static_cat + self.static_num + self.dynamic_cat + self.dynamic_num


def _view_cols(cols_json, view):
    cols = list(cols_json["views"].get(view, []))
    if view == "P":  # SCHEMA: Activity, judge and the five time features belong to P
        extra = [cols_json["activity_col"], cols_json["resource_col"]] + TIME_FEATURES
        cols = cols + [c for c in extra if c not in cols]
    if view == "B0G" and "open_cases" not in cols:
        cols = ["open_cases"] + cols
    return cols


def is_forbidden(col, forbidden):
    """Exact name or glob pattern (``is_case_*``) from never_features / excluded."""
    import fnmatch
    return any(col == f or (any(ch in f for ch in "*?[") and fnmatch.fnmatchcase(col, f)) for f in forbidden)


def forbidden_set(cols_json):
    return set(HARD_NEVER) | set(cols_json.get("never_features", [])) | set(cols_json.get("excluded", {}))


def get_spec(fs, cols_json, available=None):
    """Build the FeatureSpec of feature set ``fs``.

    ``available``: optional iterable of columns actually present in the log; columns
    listed in the json but absent from the data are dropped (with no error) so that
    small fixtures work.
    """
    views = views_of(fs)
    forbidden = forbidden_set(cols_json)
    col_view = {}
    for v in views:
        for c in _view_cols(cols_json, v):
            col_view.setdefault(c, v)
    bad = sorted(c for c in col_view if is_forbidden(c, forbidden))
    if bad:
        raise ValueError("never-feature/excluded columns in %s: %s" % (fs, bad))
    if available is not None:
        available = set(available)
        col_view = {c: v for c, v in col_view.items() if c in available}
    kinds = {}
    for kind in ["static_cat", "static_num", "dynamic_cat", "dynamic_num"]:
        order = list(cols_json.get(kind, []))
        # activity/resource are dynamic categoricals even if the json lists forgot them
        if kind == "dynamic_cat":
            for c in [cols_json["activity_col"], cols_json["resource_col"]]:
                if c not in order:
                    order.insert(0 if c == cols_json["activity_col"] else 1, c)
        if kind == "dynamic_num":
            for c in TIME_FEATURES + ["open_cases"]:
                if c not in order:
                    order.append(c)
        # column order = view order (M, P, PP, B0G, G, S), as court/court_confs.feature_columns
        # registers them in dataset_confs -> same FeatureUnion column order as upstream
        members = set(order)
        kinds[kind] = [c for c in col_view if c in members]
    unassigned = set(col_view) - set(sum(kinds.values(), []))
    if unassigned:
        raise ValueError("columns of %s with no kind in court_columns.json: %s" % (fs, sorted(unassigned)))
    last_cols = [c for c in kinds["dynamic_num"] if col_view[c] == "G"]
    return FeatureSpec(fs=fs, last_cols=last_cols, col_view=col_view, **kinds)


def view_label(view, col):
    """Importance-by-view label (C12): M / P / P+ / G / G+ / S."""
    return {"M": "M", "P": "P", "PP": "P+", "B0G": "G", "G": "G+", "S": "S"}.get(view, view)


def default_columns_path(data_dir="data/processed"):
    return os.path.join(data_dir, "court_columns.json")
