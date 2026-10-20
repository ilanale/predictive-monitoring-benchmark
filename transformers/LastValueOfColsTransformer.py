# Added by ilanale/predictive-monitoring-benchmark for court-case adaptation (C8) — see CHANGES_FROM_UPSTREAM.md
from sklearn.base import TransformerMixin
import pandas as pd
from time import time


class LastValueOfColsTransformer(TransformerMixin):
    """C8: last (non-null) value of selected dynamic numeric columns (the G+ global features),
    i.e. the state at the prediction point. Combined with the aggregation encoding ("agg+lastG"),
    so G columns get mean/max/min/sum/std *and* their last value. Columns are named last_<col>.
    Same conventions as LastStateTransformer (groupby(case).last(), fillna(0), fixed columns)."""

    def __init__(self, case_id_col, cols, fillna=True):
        self.case_id_col = case_id_col
        self.cols = cols
        self.fillna = fillna
        self.columns = None
        self.fit_time = 0
        self.transform_time = 0

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        start = time()
        dt_transformed = X.groupby(self.case_id_col)[self.cols].last()
        dt_transformed.columns = ["last_%s" % col for col in self.cols]
        if self.fillna:
            dt_transformed = dt_transformed.fillna(0)
        if self.columns is None:
            self.columns = dt_transformed.columns
        else:
            dt_transformed = dt_transformed[self.columns]
        self.transform_time = time() - start
        return dt_transformed

    def get_feature_names(self):
        return self.columns
