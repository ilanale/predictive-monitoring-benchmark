# Modified by ilanale/predictive-monitoring-benchmark for court-case adaptation — see CHANGES_FROM_UPSTREAM.md
from sklearn.base import TransformerMixin
import pandas as pd
import numpy as np
from time import time
import sys

class AggregateTransformer(TransformerMixin):
    
    def __init__(self, case_id_col, cat_cols, num_cols, boolean=False, fillna=True):
        self.case_id_col = case_id_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        
        self.boolean = boolean
        self.fillna = fillna
        
        self.columns = None
        
        self.fit_time = 0
        self.transform_time = 0
    
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        start = time()
        
        # transform numeric cols
        if len(self.num_cols) > 0:
            # compat: pandas>=1.0 removed dict-renaming agg. Named aggregation reproduces the old
            # output: function-major column order (mean, max, min, sum, std), names "<func>_<col>",
            # std with ddof=1 (np.std was dispatched to the cythonized groupby std, ddof=1).
            named_aggs = {"%s_%s" % (func, col): pd.NamedAgg(column=col, aggfunc=func)
                          for func in ['mean', 'max', 'min', 'sum', 'std'] for col in self.num_cols}
            dt_numeric = X.groupby(self.case_id_col)[self.num_cols].agg(**named_aggs)
            
        # transform cat cols
        dt_transformed = pd.get_dummies(X[self.cat_cols], dtype=np.uint8)  # compat: old default dtype (pandas>=2 gives bool)
        dt_transformed[self.case_id_col] = X[self.case_id_col]
        del X
        if self.boolean:
            dt_transformed = dt_transformed.groupby(self.case_id_col).max()
        else:
            dt_transformed = dt_transformed.groupby(self.case_id_col).sum()
        
        # concatenate
        if len(self.num_cols) > 0:
            dt_transformed = pd.concat([dt_transformed, dt_numeric], axis=1)
            del dt_numeric
        
        # fill missing values with 0-s
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)
            
        # add missing columns if necessary
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            missing_cols = [col for col in self.columns if col not in dt_transformed.columns]
            for col in missing_cols:
                dt_transformed[col] = 0
            dt_transformed = dt_transformed[self.columns]
        
        self.transform_time = time() - start
        return dt_transformed
    
    def get_feature_names(self):
        return self.columns