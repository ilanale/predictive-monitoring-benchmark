"""Per-bucket best parameters for prefix-length bucketing (port of the "prefix" cell of upstream
experiments/extracting_best_params_revision.ipynb, which is left untouched).

optimize_params.py writes one flat dict for prefix_* methods, but experiments.py reads args[bucket].
Upstream bridged this with the notebook: for each nr_events, take the trial with the highest score
(first such trial on ties) from param_optim_all_trials_*.csv.

Usage: python tests/fixtures/extract_best_params_prefix.py <params_dir> <cls_method> <dataset> <method_name>
Overwrites <params_dir>/optimal_params_<cls>_<dataset>_<method>.pickle with {nr_events: params}.
"""
import os
import pickle
import sys

import pandas as pd


def extract(params_dir, cls_method, dataset_name, method_name):
    trials_file = os.path.join(params_dir, "param_optim_all_trials_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    data = pd.read_csv(trials_file, sep=";")
    data = data[data.param != "processing_time"]
    best_params = {}
    for nr_events, group in data.groupby("nr_events"):
        vals = {val.iloc[0]: val.iloc[1] for _, val in list(group[group.score == group.score.max()].groupby("iter"))[0][1][["param", "value"]].iterrows()}
        best_params[nr_events] = vals
    out = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (cls_method, dataset_name, method_name))
    with open(out, "wb") as fout:
        pickle.dump(best_params, fout)
    return out


if __name__ == "__main__":
    print(extract(*sys.argv[1:5]))
