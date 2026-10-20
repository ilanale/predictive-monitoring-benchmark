"""C1-C8 changes in the upstream files, and fast path == adapted upstream path (C9, G3)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from tests.fixtures.make_court_like_log import columns_json, make_log, write_fixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestAdaptedDatasetManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="court_up_")
        cls.data = write_fixture(os.path.join(cls.tmp, "data"), n_cases=60)
        from court import upstream
        cls.upstream = upstream
        cls.dm_clf = upstream.dataset_manager("court_p1_clf", "FS-ALL", cls.data)
        cls.dm_reg = upstream.dataset_manager("court_p1_reg", "FS-ALL", cls.data)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def df(self, max_hist=6, seed=0):
        d = make_log(60, 1, seed, max_hist=max_hist)
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        return d

    def test_registration(self):
        import dataset_confs
        self.assertEqual(dataset_confs.task_type["court_p1_reg__FS-B0"], "regression")
        self.assertEqual(self.dm_clf.global_cols, ["G_court_open", "G_judge_load"])
        self.assertEqual(self.dm_clf.n_hist_col, "n_hist")
        self.assertIn("court_p2_clf__FS-ALL+S", dataset_confs.task_type)

    def test_n_hist_zero_equals_original_prefix_generation(self):
        """DoD-4 style: with n_hist = 0 the in-phase prefixes are the upstream prefixes."""
        d = self.df(max_hist=0).sort_values(["timestamp", "Activity"], kind="mergesort")
        new = self.dm_clf.generate_prefix_data(d.copy(), 1, 8, 3)
        self.dm_clf.n_hist_col = None
        try:
            old = self.dm_clf.generate_prefix_data(d.copy(), 1, 8, 3)
        finally:
            self.dm_clf.n_hist_col = "n_hist"
        cols = ["segment_id", "Activity", "timestamp", "prefix_nr", "orig_case_id", "case_length"]
        a = new[cols].sort_values(["segment_id", "timestamp"], kind="mergesort").reset_index(drop=True)
        b = old[cols].sort_values(["segment_id", "timestamp"], kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)

    def test_in_phase_prefixes(self):
        d = self.df().sort_values(["timestamp", "Activity"], kind="mergesort")
        p = self.dm_clf.generate_prefix_data(d, 1, 8, 3)
        g = p.groupby("segment_id")
        size, nh, nr = g.size(), g["n_hist"].first(), g["prefix_nr"].last()
        np.testing.assert_array_equal(size.to_numpy(), (nh + nr).to_numpy())
        self.assertTrue(set(nr.unique()) <= {1, 4, 7})
        last_in_phase = g["in_phase"].last()
        self.assertTrue((last_in_phase == 1).all())

    def test_regression_labels_and_folds(self):
        d = self.df().sort_values(["timestamp", "Activity"], kind="mergesort")
        p = self.dm_reg.generate_prefix_data(d, 1, 40, 1)
        lab = self.dm_reg.get_label(p)
        last = p.groupby("segment_id")["remaining_in_phase"].last()
        pd.testing.assert_series_equal(lab, last)
        np.testing.assert_allclose(self.dm_reg.get_label_numeric(p), np.log1p(last.to_numpy()))
        self.assertAlmostEqual(self.dm_reg.get_class_ratio(d), np.log1p(np.median(d.remaining_in_phase)))
        folds = list(self.dm_reg.get_stratified_split_generator(d, n_splits=3))
        self.assertEqual(len(folds), 3)

    def test_split_matches_court_reimplementation(self):
        from court import data_io
        d = self.df()
        tr, te = self.dm_clf.split_data_strict(d, 0.8)
        tr2, te2, split_ts = data_io.split_data_strict(d)
        self.assertEqual(set(tr.segment_id), set(tr2.segment_id))
        self.assertEqual(set(te.segment_id), set(te2.segment_id))
        self.assertEqual(len(tr), len(tr2))
        self.assertTrue((tr.timestamp < split_ts).all())

    def test_quantile_c7(self):
        from court import data_io
        d = self.df()
        self.assertEqual(self.dm_clf.get_pos_case_length_quantile(d), int(np.ceil(
            data_io.in_phase_lengths(d)[d.groupby("segment_id").label.first() == "deviant"].quantile(0.9))))
        self.assertEqual(min(40, self.dm_reg.get_pos_case_length_quantile(d)), data_io.max_prefix_length(d, "regression"))

    def test_last_value_transformer_and_prefix_bucketer(self):
        import BucketFactory
        import EncoderFactory
        d = self.df().sort_values(["timestamp", "Activity"], kind="mergesort")
        p = self.dm_clf.generate_prefix_data(d, 1, 8, 1)
        enc = EncoderFactory.get_encoder("lastG", case_id_col="segment_id", global_cols=["G_court_open"])
        out = enc.fit_transform(p)
        exp = p.groupby("segment_id")["G_court_open"].last().fillna(0)
        np.testing.assert_array_equal(out["last_G_court_open"].to_numpy(), exp.to_numpy())
        b = BucketFactory.get_bucketer("prefix", case_id_col="segment_id")
        np.testing.assert_array_equal(b.fit_predict(p), p.groupby("segment_id")["prefix_nr"].last().to_numpy())


class TestFastEqualsAdaptedUpstream(unittest.TestCase):
    """C9: court.verify_fast_path on the court-like fixture: same split, L, buckets, labels, feature
    names/values and identical XGBoost predictions."""

    def test_verify_fast_path(self):
        tmp = tempfile.mkdtemp(prefix="court_verify_")
        try:
            data = write_fixture(os.path.join(tmp, "data"), n_cases=70, seed=4)
            out = os.path.join(tmp, "verify.json")
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([ROOT, os.path.join(ROOT, "experiments")]),
                       OMP_NUM_THREADS="2", PYTHONWARNINGS="ignore")
            subprocess.run([sys.executable, "-m", "court.verify_fast_path", "--data-dir", data, "--n-segments", "70",
                            "--fs", "FS-ALL,FS-M,FS-B0+S", "--n-estimators", "40", "--n-jobs", "2", "--out", out],
                           cwd=ROOT, env=env, check=True, capture_output=True)
            with open(out) as f:
                rep = json.load(f)
            self.assertEqual(len(rep), 6 * 3)
            for r in rep:
                self.assertTrue(r["L_equal"] and r["split_equal"], r["task"])
                for m, v in r["methods"].items():
                    self.assertTrue(v["bucket_assign_equal"], (r["task"], m))
                    self.assertTrue(v["buckets"])
                    for b in v["buckets"]:
                        ctx = (r["task"], r["fs"], m, b["bucket"])
                        self.assertTrue(b["names_equal"] and b["train_ids_equal"] and b["test_ids_equal"], ctx)
                        self.assertTrue(b["labels_equal"], ctx)
                        self.assertLessEqual(b["max_abs_diff_train"], 1e-12, ctx)
                        self.assertLessEqual(b["max_abs_diff_test"], 1e-12, ctx)
                        if "max_abs_pred_diff" in b:
                            self.assertEqual(b["max_abs_pred_diff"], 0.0, ctx)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

_ = columns_json
