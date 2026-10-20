# Modified by ilanale/predictive-monitoring-benchmark for court-case adaptation — see CHANGES_FROM_UPSTREAM.md
import pandas as pd
import numpy as np
from time import time
import sys

class PrefixLengthBucketer(object):
    
    def __init__(self, case_id_col):
        self.n_states = 0
        self.case_id_col = case_id_col
        
    
    def _lengths(self, X):
        # C6: court prefixes (with an `n_hist` offset) are bucketed by their in-phase length
        # `prefix_nr`; upstream logs (no `n_hist`) keep the number of events in the prefix
        if "n_hist" in X.columns and "prefix_nr" in X.columns:
            return X.groupby(self.case_id_col)["prefix_nr"].last()
        return X.groupby(self.case_id_col).size()

    def fit(self, X, y=None):
        
        sizes = self._lengths(X)
        self.n_states = sizes.unique()
        
        return self
    
    
    def predict(self, X, y=None):
        
        bucket_assignments = self._lengths(X)
        while sum(~bucket_assignments.isin(self.n_states)) > 0:
            bucket_assignments[~bucket_assignments.isin(self.n_states)] -= 1
        return bucket_assignments.to_numpy()  # compat: as_matrix removed
    
    
    def fit_predict(self, X, y=None):
        
        self.fit(X)
        return self.predict(X)