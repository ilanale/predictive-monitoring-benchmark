import unittest

import numpy as np
import pandas as pd

from court import fast_pipeline
from court.fast_pipeline import FastEncoder, apply_columns, fit_columns, prepare_log
from court.feature_sets import ALL_FS, get_spec
from tests import reference_upstream as ref
from tests.fixtures.make_court_like_log import columns_json, make_log


def load(n=60, phase=1, seed=0):
    df = make_log(n, phase, seed)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def fast(df, spec, L, gap, layout=None, dtype=np.float64, max_chunk_cells=4e7):
    enc = FastEncoder(spec, dtype=dtype, layout=layout, max_chunk_cells=max_chunk_cells)
    pm = enc.encode(df, L, gap)
    return pm.subset_cols(fit_columns(pm))


class TestFastEqualsUpstream(unittest.TestCase):
    """C9: fast path == literal upstream prefix log + encoders (max abs diff < 1e-6)."""

    def check(self, df, fs, L, gap, layout="stat_major", **kw):
        spec = get_spec(fs, columns_json(), df.columns)
        pm = fast(df, spec, L, gap, layout, **kw)
        R, yreg, pnr = ref.encode(ref.generate_prefix_data(df, L, gap), spec, layout)
        self.assertEqual(list(R.columns), pm.names)
        self.assertEqual(list(R.index), list(pm.meta["prefix_case_id"]))
        self.assertEqual(np.abs(R.to_numpy() - pm.X).max(), 0.0)  # bit-identical in float64
        np.testing.assert_allclose(yreg.to_numpy(), pm.meta["y_reg"].to_numpy())
        np.testing.assert_array_equal(pnr.to_numpy(), pm.meta["prefix_nr"].to_numpy())

    def test_all_feature_sets_gap1_and_gap3(self):
        df = load()
        for fs in ALL_FS + ["FS-ALL+S"]:
            for gap in (1, 3):
                with self.subTest(fs=fs, gap=gap):
                    self.check(df, fs, 8, gap)

    def test_identical_xgboost_predictions(self):
        """Same params -> identical predictions from the fast matrix and the upstream-style matrix."""
        import xgboost as xgb
        df = load(80, 1, 3)
        spec = get_spec("FS-ALL", columns_json(), df.columns)
        pm = fast(df, spec, 10, 3)
        R, _, _ = ref.encode(ref.generate_prefix_data(df, 10, 3), spec)
        y_ref = (ref.generate_prefix_data(df, 10, 3).groupby("segment_id")["label"].first() == "deviant").astype(int)
        np.testing.assert_array_equal(y_ref.to_numpy(), pm.meta["y_clf"].to_numpy())
        kw = dict(n_estimators=50, learning_rate=0.3, subsample=0.7, max_depth=5, colsample_bytree=0.6,
                  min_child_weight=1, random_state=22, n_jobs=2)
        p1 = xgb.XGBClassifier(**kw).fit(pm.X, pm.meta["y_clf"]).predict_proba(pm.X)[:, 1]
        p2 = xgb.XGBClassifier(**kw).fit(R.to_numpy(np.float32), y_ref.to_numpy()).predict_proba(
            R.to_numpy(np.float32))[:, 1]
        np.testing.assert_array_equal(p1, p2)

    def test_col_major_layout_and_small_chunks(self):
        df = load(40, 2, 1)
        self.check(df, "FS-ALL", 40, 1, layout="col_major", max_chunk_cells=50)

    def test_prefix_lengths_in_phase_only(self):
        df = load()
        spec = get_spec("FS-B0", columns_json(), df.columns)
        pm = fast(df, spec, 5, 3)
        seg = df.groupby("segment_id").agg(n=("Activity", "size"), h=("n_hist", "first"))
        for _, r in pm.meta.iterrows():
            s = seg.loc[r.segment_id]
            self.assertEqual(r.prefix_len, s.h + r.prefix_nr)
            self.assertIn(r.prefix_nr, (1, 4))
            self.assertLessEqual(r.prefix_nr, s.n - s.h)

    def test_float32_close(self):
        df = load()
        spec = get_spec("FS-ALL", columns_json(), df.columns)
        a, b = fast(df, spec, 8, 1), fast(df, spec, 8, 1, dtype=np.float32)
        np.testing.assert_allclose(a.X, b.X, rtol=1e-6, atol=1e-3)

    def test_apply_columns_aligns_test_to_train(self):
        df = load()
        spec = get_spec("FS-B0", columns_json(), df.columns)
        enc = FastEncoder(spec, dtype=np.float64)
        tr = enc.encode(df.iloc[: len(df) // 2], 8, 1)
        tr = tr.subset_cols(fit_columns(tr))
        te = apply_columns(enc.encode(df, 8, 1), tr.names + ["Activity_NEVER"])
        self.assertEqual(te.names[-1], "Activity_NEVER")
        self.assertTrue((te.X[:, -1] == 0).all())


class TestNoFutureLeakage(unittest.TestCase):
    """D7: perturbing events after a prefix end leaves that prefix's features unchanged."""

    def test_perturb_future_events(self):
        df = prepare_log(load())
        spec = get_spec("FS-ALL+S", columns_json(), df.columns)
        enc = FastEncoder(spec, dtype=np.float64)
        enc.build_vocab(df)
        base = enc.encode(df, 40, 1, prepared=True)
        rng = np.random.default_rng(5)
        pos = df.groupby("segment_id").cumcount().to_numpy()
        cut = df["n_hist"].to_numpy() + 2  # prefix j = 2 ends at position n_hist + 1 (0-based)
        end_ts = df.assign(pos=pos).loc[pos == cut - 1].set_index("segment_id")["timestamp"]
        seg_end = df["segment_id"].map(end_ts)
        fut = (pos >= cut) & seg_end.notna().to_numpy()
        later_day = fut & (df["timestamp"] > seg_end).to_numpy()
        p = df.copy()
        num = [c for c in spec.dynamic_num]
        p.loc[fut, num] = p.loc[fut, num].to_numpy() * 7 + rng.normal(size=(fut.sum(), len(num)))
        p.loc[fut, "judge"] = "J99"
        p.loc[later_day, "Activity"] = "Zzz"  # same-day events keep their activity (sort key)
        pert = enc.encode(p, 40, 1, prepared=True)
        k = base.meta["prefix_nr"].to_numpy() <= 2
        np.testing.assert_array_equal(base.meta["prefix_case_id"], pert.meta["prefix_case_id"])
        np.testing.assert_allclose(base.X[k], pert.X[k], atol=1e-9)
        self.assertGreater(np.abs(base.X[~k] - pert.X[~k]).max(), 0)  # the perturbation did bite


class TestRegressionLabels(unittest.TestCase):
    def test_label_is_remaining_of_last_event(self):
        df = load()
        spec = get_spec("FS-B0", columns_json(), df.columns)
        pm = fast(df, spec, 40, 1)
        exit_ = pd.to_datetime(df.groupby("segment_id")["exit_date"].first())
        expect = (pm.meta["segment_id"].map(exit_) - pm.meta["last_ts"]).dt.days.to_numpy()
        np.testing.assert_array_equal(pm.meta["y_reg"].to_numpy(), expect)
        varies = pm.meta.groupby("segment_id")["y_reg"].nunique()
        self.assertTrue((varies > 1).any())


if __name__ == "__main__":
    unittest.main()

_ = fast_pipeline
