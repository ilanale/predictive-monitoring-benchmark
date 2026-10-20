"""Compat regression test (DoD-4): the current code must reproduce tests/fixtures/compat_baseline exactly.

Run: python -m unittest discover -s tests      (from the repo root; takes ~40 s)
Regenerates the synthetic log (seed 22) in a temp dir, reruns upstream experiments.py for single_agg and
prefix_agg with fixed XGBoost hyperparameters, re-encodes the test prefixes and compares:
  - per-prefix-length and overall AUC rows of the results files: identical
  - encoded feature matrices: same columns/rows, max abs diff < 1e-9
  - predicted probabilities for every test prefix: identical
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "compat_baseline")
sys.path.insert(0, HERE)
import compat_baseline  # noqa: E402


def _auc_rows(path):
    with open(path) as f:
        return [line.rstrip("\n") for line in f if ";auc;" in line]


class TestCompatSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = cls._tmp.name
        compat_baseline.main(cls.out)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_auc_results_identical(self):
        for m in ["single_agg", "prefix_agg"]:
            fname = "performance_results_xgboost_synthetic_%s.csv" % m
            expected = _auc_rows(os.path.join(FIXTURE, fname))
            self.assertGreater(len(expected), 10)
            self.assertEqual(_auc_rows(os.path.join(self.out, fname)), expected, m)

    def test_features_match(self):
        files = sorted(f for f in os.listdir(FIXTURE) if f.startswith("features_"))
        self.assertEqual(len(files), 4)
        for fname in files:
            exp = pd.read_csv(os.path.join(FIXTURE, fname), sep=";")
            got = pd.read_csv(os.path.join(self.out, fname), sep=";")
            self.assertEqual(list(got.columns), list(exp.columns), fname)
            self.assertEqual(list(got.iloc[:, 0]), list(exp.iloc[:, 0]), fname)
            diff = np.abs(got.iloc[:, 1:].to_numpy(dtype=float) - exp.iloc[:, 1:].to_numpy(dtype=float))
            self.assertLess(float(np.nanmax(diff)) if diff.size else 0.0, 1e-9, fname)
            self.assertTrue((np.isnan(got.iloc[:, 1:].to_numpy(dtype=float)) ==
                             np.isnan(exp.iloc[:, 1:].to_numpy(dtype=float))).all(), fname)

    def test_predictions_identical(self):
        for m in ["single_agg", "prefix_agg"]:
            fname = "predictions_%s.csv" % m
            exp = pd.read_csv(os.path.join(FIXTURE, fname), sep=";", dtype={"predicted": str})
            got = pd.read_csv(os.path.join(self.out, fname), sep=";", dtype={"predicted": str})
            self.assertGreater(len(exp), 100)
            pd.testing.assert_frame_equal(got, exp, check_exact=True)


if __name__ == "__main__":
    unittest.main()
