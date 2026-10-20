# Modified by ilanale/predictive-monitoring-benchmark for court-case adaptation — see CHANGES_FROM_UPSTREAM.md
import sys

import dataset_confs

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold


class DatasetManager:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.activity_col = dataset_confs.activity_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        self.label_col = dataset_confs.label_col[self.dataset_name]
        self.pos_label = dataset_confs.pos_label[self.dataset_name]

        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name]
        self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]
        
        self.sorting_cols = [self.timestamp_col, self.activity_col]

        # C1: task type (classification | regression); datasets not registered by court/court_confs.py
        # keep the upstream behaviour (classification, no in-phase offsets)
        self.task_type = getattr(dataset_confs, "task_type", {}).get(self.dataset_name, "classification")
        self.regression_label_col = getattr(dataset_confs, "regression_label_col", {}).get(self.dataset_name)
        self.global_cols = getattr(dataset_confs, "global_cols", {}).get(self.dataset_name, [])
        # C6: court segments carry `n_hist` (events before phase entry) and `in_phase`
        self.n_hist_col = "n_hist" if self.dataset_name in getattr(dataset_confs, "task_type", {}) else None
        
    
    def read_dataset(self):
        # read dataset
        dtypes = {col:"object" for col in self.dynamic_cat_cols+self.static_cat_cols+[self.case_id_col, self.label_col, self.timestamp_col]}
        for col in self.dynamic_num_cols + self.static_num_cols:
            dtypes[col] = "float"
        if self.regression_label_col is not None:
            dtypes[self.regression_label_col] = "float"  # C2

        data = pd.read_csv(dataset_confs.filename[self.dataset_name], sep=";", dtype=dtypes)
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])

        return data


    def split_data(self, data, train_ratio, split="temporal", seed=22):  
        # split into train and test using temporal split

        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')

        return (train, test)
    
    def split_data_strict(self, data, train_ratio, split="temporal"):  
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        if self.n_hist_col is not None:
            # C6/D9: a segment's trace starts at case filing, so segments are ordered by their first
            # *in-phase* event and the test period starts at the earliest test segment's phase entry
            grouped = data[data["in_phase"] == 1].groupby(self.case_id_col)
        else:
            grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        if self.n_hist_col is not None:
            test = test[test[self.case_id_col].isin(set(start_timestamps[self.case_id_col]))]
            split_ts = start_timestamps[self.timestamp_col].iloc[int(train_ratio*len(start_timestamps))]
        else:
            split_ts = test[self.timestamp_col].min()
        train = train[train[self.timestamp_col] < split_ts]
        return (train, test)
    
    def split_data_discard(self, data, train_ratio, split="temporal"):  
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        split_ts = test[self.timestamp_col].min()
        overlapping_cases = train[train[self.timestamp_col] >= split_ts][self.case_id_col].unique()
        train = train[~train[self.case_id_col].isin(overlapping_cases)]
        return (train, test)
    
    
    def split_val(self, data, val_ratio, split="random", seed=22):  
        # split into train and test using temporal split
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
        val_ids = list(start_timestamps[self.case_id_col])[-int(val_ratio*len(start_timestamps)):]
        val = data[data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        train = data[~data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        return (train, val)


    def generate_prefix_data(self, data, min_length, max_length, gap=1):
        if self.n_hist_col is not None:
            return self._generate_in_phase_prefix_data(data, min_length, max_length, gap)
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
        dt_prefixes["prefix_nr"] = 1
        dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id_col]
        for nr_events in range(min_length+gap, max_length+1, gap):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            tmp["orig_case_id"] = tmp[self.case_id_col]
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            tmp["prefix_nr"] = nr_events
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
        dt_prefixes['case_length'] = dt_prefixes['case_length'].apply(lambda x: min(max_length, x))
        
        return dt_prefixes


    def _generate_in_phase_prefix_data(self, data, min_length, max_length, gap=1):
        # C6 (D9): prefixes are cut only inside the phase. Prefix lengths are n_hist + j for
        # j = min_length, min_length+gap, ... <= max_length; prefix_nr := j (in-phase length).
        # Row order and case-id naming follow the upstream loop above (first prefix keeps the
        # case id, later ones get "<case>_<nr_events>").
        data = data.copy()
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)
        pos = data.groupby(self.case_id_col).cumcount().to_numpy()
        n_hist = data[self.n_hist_col].to_numpy().astype(int)
        case_length = data['case_length'].to_numpy()
        parts = []
        for j in range(min_length, max_length+1, gap):
            nr_events = n_hist + j
            mask = (case_length >= nr_events) & (pos < nr_events)
            tmp = data[mask].copy()
            tmp["orig_case_id"] = tmp[self.case_id_col]
            if j > min_length:
                tmp[self.case_id_col] = tmp[self.case_id_col].astype(str) + "_" + nr_events[mask].astype(str)
            tmp["prefix_nr"] = j
            parts.append(tmp)
        dt_prefixes = pd.concat(parts, axis=0)
        dt_prefixes['case_length'] = np.minimum(max_length, dt_prefixes['case_length'] - dt_prefixes[self.n_hist_col].astype(int))
        return dt_prefixes

    def get_pos_case_length_quantile(self, data, quantile=0.90):
        if self.n_hist_col is not None:
            # C7: quantile of the in-phase length; positive class (classification) / all segments (regression)
            if self.task_type == "classification":
                data = data[data[self.label_col]==self.pos_label]
            g = data.groupby(self.case_id_col)
            return int(np.ceil((g.size() - g[self.n_hist_col].first()).quantile(quantile)))
        return int(np.ceil(data[data[self.label_col]==self.pos_label].groupby(self.case_id_col).size().quantile(quantile)))

    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data):
        if self.task_type == "regression":
            # C2: the target changes with every prefix: remaining days at the last event of the prefix
            return data.groupby(self.case_id_col).last()[self.regression_label_col]
        return data.groupby(self.case_id_col).first()[self.label_col]
    
    def get_prefix_lengths(self, data):
        return data.groupby(self.case_id_col).last()["prefix_nr"]
    
    def get_case_ids(self, data, nr_events=1):
        case_ids = pd.Series(data.groupby(self.case_id_col).first().index)
        if nr_events > 1:
            case_ids = case_ids.apply(lambda x: "_".join(x.split("_")[:-1]))
        return case_ids
    
    def get_label_numeric(self, data):
        y = self.get_label(data) # one row per case
        if self.task_type == "regression":
            return list(np.log1p(y.to_numpy(dtype=float)))  # D10: models are trained on log1p(days)
        return [1 if label == self.pos_label else 0 for label in y]
    
    def get_class_ratio(self, data):
        if self.task_type == "regression":
            # C5: small-bucket fallback = training median (log1p scale, like the model output)
            return float(np.log1p(np.median(data[self.regression_label_col].to_numpy(dtype=float))))
        class_freqs = data[self.label_col].value_counts()
        return class_freqs[self.pos_label] / class_freqs.sum()
    
    def get_stratified_split_generator(self, data, n_splits=5, shuffle=True, random_state=22):
        grouped_firsts = data.groupby(self.case_id_col, as_index=False).first()
        if self.task_type == "regression":
            skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)  # C5
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        for train_index, test_index in skf.split(grouped_firsts, grouped_firsts[self.label_col]):
            current_train_names = grouped_firsts[self.case_id_col][train_index]
            train_chunk = data[data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            test_chunk = data[~data[self.case_id_col].isin(current_train_names)].sort_values(self.timestamp_col, ascending=True, kind='mergesort')
            yield (train_chunk, test_chunk)
            
    def get_idx_split_generator(self, dt_for_splitting, n_splits=5, shuffle=True, random_state=22):
        if self.task_type == "regression":
            skf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)  # C5
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        for train_index, test_index in skf.split(dt_for_splitting, dt_for_splitting[self.label_col]):
            current_train_names = dt_for_splitting[self.case_id_col][train_index]
            current_test_names = dt_for_splitting[self.case_id_col][test_index]
            yield (current_train_names, current_test_names)
            