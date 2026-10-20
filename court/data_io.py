"""Task loading, max prefix length (C7) and the strict temporal split for court tasks.

``split_data_strict`` re-implements upstream ``DatasetManager.split_data_strict`` in court
code for phase 1 (upstream files are being ported by A1). Court adaptation, as in PLAN.md
§4 "Split": segments are ordered by their *first in-phase event* (not the first trace event,
which for D9 traces is the case filing date), and train events on/after the test start
(= first in-phase event of the earliest test segment) are cut. Phase 2 switches to the
(adapted) upstream function.
"""
import json
import os

import numpy as np
import pandas as pd

TASKS = ["court_p1_clf", "court_p1_reg", "court_p2_clf", "court_p2_reg", "court_p3_clf", "court_p3_reg"]


def parse_task(task):
    _, p, kind = task.split("_")
    return int(p[1:]), ("classification" if kind == "clf" else "regression")


def load_columns(data_dir):
    with open(os.path.join(data_dir, "court_columns.json")) as f:
        return json.load(f)


def load_task_log(task, data_dir, sample_frac=None, seed=22):
    phase, _ = parse_task(task)
    pq = os.path.join(data_dir, "court_phase%d.parquet" % phase)
    csv = os.path.join(data_dir, "court_phase%d.csv" % phase)
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
    else:
        df = pd.read_csv(csv, sep=";", low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if sample_frac and sample_frac < 1:
        segs = df["segment_id"].drop_duplicates()
        keep = segs.sample(frac=sample_frac, random_state=seed)
        df = df[df["segment_id"].isin(set(keep))]
    return df.reset_index(drop=True)


def in_phase_lengths(df, case_col="segment_id"):
    g = df.groupby(case_col)
    return g.size() - g["n_hist"].first()


def max_prefix_length(df, task_type, cap=40, quantile=0.90, label_col="label", pos_label="deviant",
                      case_col="segment_id"):
    """C7: min(cap, ceil(q90 of in-phase length)); positive class only for classification."""
    lens = in_phase_lengths(df, case_col)
    if task_type == "classification":
        pos = df.groupby(case_col)[label_col].first() == pos_label
        lens = lens[pos[lens.index]]
    return int(min(cap, int(np.ceil(lens.quantile(quantile)))))


def segment_start_times(df, case_col="segment_id", ts_col="timestamp"):
    """First in-phase event time per segment."""
    ip = df[df["in_phase"] == 1]
    return ip.groupby(case_col)[ts_col].min()


def split_data_strict(df, train_ratio=0.8, case_col="segment_id", ts_col="timestamp", act_col="Activity"):
    data = df.sort_values([ts_col, act_col], ascending=True, kind="mergesort")
    start = segment_start_times(data, case_col, ts_col).reset_index()
    start = start.sort_values(ts_col, ascending=True, kind="mergesort")
    train_ids = set(start[case_col].iloc[:int(train_ratio * len(start))])
    test_ids = set(start[case_col]) - train_ids
    train = data[data[case_col].isin(train_ids)]
    test = data[data[case_col].isin(test_ids)]
    split_ts = None
    if test_ids:
        split_ts = start.loc[start[case_col].isin(test_ids), ts_col].min()
        train = train[train[ts_col] < split_ts]
    return train, test, split_ts
