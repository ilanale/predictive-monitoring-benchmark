import contextlib
import io
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from court import data_io, evaluate, run_queue, runner
from court.fast_pipeline import FastEncoder, fit_columns
from court.feature_sets import ALL_FS, HARD_NEVER, get_spec
from tests.fixtures.make_court_like_log import columns_json, make_log, write_fixture


def load(n=80, phase=1, seed=0):
    df = make_log(n, phase, seed)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


class TestSplit(unittest.TestCase):
    def test_strict_split(self):
        df = load(120)
        train, test, split_ts = data_io.split_data_strict(df)
        self.assertFalse(set(train.segment_id) & set(test.segment_id))
        self.assertTrue((train.timestamp < split_ts).all())
        starts = data_io.segment_start_times(df)
        self.assertEqual(split_ts, starts[test.segment_id.unique()].min())
        self.assertEqual(test.segment_id.nunique(), len(starts) - int(0.8 * len(starts)))
        self.assertTrue((starts[train.segment_id.unique()] <= split_ts).all())

    def test_max_prefix_length(self):
        df = load(120)
        L = data_io.max_prefix_length(df, "classification")
        lens = data_io.in_phase_lengths(df)
        pos = df.groupby("segment_id").label.first() == "deviant"
        self.assertEqual(L, min(40, int(np.ceil(lens[pos].quantile(0.9)))))


class TestExcludedColumns(unittest.TestCase):
    """D8: never-feature / excluded columns are in no feature matrix."""

    def test_never_in_any_matrix(self):
        df = load()
        cols = columns_json()
        banned = set(HARD_NEVER) | set(cols["never_features"]) | set(cols["excluded"])
        for fs in ALL_FS + [f + "+S" for f in ALL_FS]:
            spec = get_spec(fs, cols, df.columns)
            pm = FastEncoder(spec).encode(df, 10, 3)
            self.assertFalse(banned & set(pm.sources), fs)
            self.assertFalse([n for n in pm.names if any(n == b or n.startswith(b + "_") for b in banned)], fs)
            if not fs.endswith("+S"):
                self.assertNotIn("S_counsel_count", pm.sources)

    def test_json_putting_excluded_in_a_view_raises(self):
        cols = columns_json()
        cols["views"]["M"].append("resolution_type")
        with self.assertRaises(ValueError):
            get_spec("FS-M", cols)

    def test_fs_views(self):
        cols = columns_json()
        self.assertNotIn("open_cases", get_spec("FS-MP", cols).col_view)
        b0 = get_spec("FS-B0", cols)
        self.assertIn("open_cases", b0.col_view)
        self.assertFalse([c for c in b0.col_view if c.startswith(("PP_", "G_"))])
        self.assertEqual(get_spec("FS-ALL", cols).last_cols, ["G_court_open", "G_judge_load"])
        self.assertEqual(get_spec("FS-B0", cols).last_cols, [])


class TestEvaluate(unittest.TestCase):
    def preds(self, noise, seed=0, n_seg=300):
        rng = np.random.default_rng(seed)
        seg = np.repeat(np.arange(n_seg), 3).astype(str)
        y = np.repeat(rng.integers(0, 2, n_seg), 3)
        p = np.clip(y * 0.6 + 0.2 + rng.normal(0, noise, len(y)), 0, 1)
        return pd.DataFrame({"segment_id": seg, "prefix_len": np.tile([1, 2, 3], n_seg), "y_true": y, "y_pred": p})

    def test_bootstrap_identical_is_zero(self):
        a = self.preds(0.3)
        r = evaluate.paired_bootstrap(a, a.copy(), "classification", n_boot=200)
        self.assertEqual((r["delta"], r["ci_low"], r["ci_high"]), (0, 0, 0))

    def test_bootstrap_better_model_positive_and_reproducible(self):
        good, bad = self.preds(0.1), self.preds(0.6)
        bad = good[["segment_id", "prefix_len", "y_true"]].assign(y_pred=bad["y_pred"].to_numpy())
        r1 = evaluate.paired_bootstrap(good, bad, "classification", n_boot=300)
        r2 = evaluate.paired_bootstrap(good, bad, "classification", n_boot=300)
        self.assertGreater(r1["ci_low"], 0)
        self.assertLessEqual(r1["ci_low"], r1["delta"]); self.assertLessEqual(r1["delta"], r1["ci_high"])
        self.assertEqual(r1, r2)

    def test_regression_metrics_and_baselines(self):
        m = evaluate.reg_metrics([10, 20, 30], [10, 25, 20])
        self.assertAlmostEqual(m["mae"], 5.0); self.assertAlmostEqual(m["median_ae"], 5.0)
        tr = pd.DataFrame({"prefix_nr": [1, 1, 4, 4], "y_reg": [10.0, 30.0, 5.0, 7.0]})
        te = pd.DataFrame({"prefix_nr": [1, 2, 5], "y_reg": [20.0, 20.0, 6.0]})
        bl = evaluate.regression_baselines(tr, te)
        np.testing.assert_allclose(bl["pred_overall"], [8.5] * 3)
        np.testing.assert_allclose(bl["pred_per_prefix"], [20, 20, 6])

    def test_importance_by_view(self):
        imp = evaluate.importance_by_view({"mean_open_cases": 2.0, "last_G_court_open": 1.0, "Activity_Order": 1.0},
                                          {"mean_open_cases": "open_cases", "last_G_court_open": "G_court_open",
                                           "Activity_Order": "Activity"},
                                          {"open_cases": "B0G", "G_court_open": "G", "Activity": "P"})
        self.assertEqual(dict(zip(imp.view, imp.share)), {"G": 0.5, "G+": 0.25, "P": 0.25})


class TestRunnerAndQueue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="court_test_")
        cls.data = write_fixture(os.path.join(cls.tmp, "data"), n_cases=80)
        cls.cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "configs", "experiments.yaml")))
        cls.cfg["budget"]["xgb_n_jobs"] = 2
        cls.cfg["budget"]["tuning"]["n_iter"] = 2
        cls.cfg_path = os.path.join(cls.tmp, "cfg.yaml")
        yaml.safe_dump(cls.cfg, open(cls.cfg_path, "w"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_run_job_end_to_end(self):
        res = os.path.join(self.tmp, "res_run")
        for task in ["court_p1_clf", "court_p1_reg"]:
            for method in ["single_agg", "prefix_agg"]:
                from court import tune
                tune.run_tuning(runner.tune_id(task, method, "xgboost"), self.cfg, self.data, res)
                jid = runner.make_job_id("tier1a", task, method, "FS-ALL", "xgboost")
                row = runner.run_job(jid, self.cfg, self.data, res)
                pred = pd.read_csv(os.path.join(res, "predictions", jid + ".csv.gz"))
                self.assertEqual(list(pred.columns), ["segment_id", "prefix_len", "y_true", "y_pred"])
                self.assertEqual(len(pred), row["n_test_prefixes"])
                self.assertFalse(pred.y_pred.isna().any())
                self.assertTrue(row["params_source"].startswith("tuned"))
                if task.endswith("reg"):
                    self.assertTrue((pred.y_pred >= 0).all())
                    self.assertTrue(np.isfinite(row["baseline_mae_overall"]))
        runs = runner.read_runs(res)
        self.assertEqual((runs.status == "done").sum(), 8)

    def test_queue_order_and_resume(self):
        jobs = run_queue.expand_jobs(self.cfg, ["tier1a"])
        tasks = [j["job_id"].split("__")[1] for j in jobs]
        order = list(dict.fromkeys(tasks))
        self.assertEqual(order, ["court_p1_clf", "court_p1_reg", "court_p2_clf", "court_p2_reg",
                                 "court_p3_clf", "court_p3_reg"])
        self.assertEqual(len(jobs), 6 * (1 + 6))
        res = os.path.join(self.tmp, "res_queue")
        done_id = runner.make_job_id("tier1a", "court_p1_clf", "single_agg", "FS-M", "xgboost")
        runner.append_run(res, dict(runner.parse_job_id(done_id), job_id=done_id, status="done"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_queue.main(["--config", self.cfg_path, "--tier", "tier1a", "--dry-run", "--results-dir", res,
                            "--data-dir", self.data])
        out = buf.getvalue()
        self.assertNotIn("DRY " + done_id, out)
        self.assertIn("DRY " + runner.make_job_id("tier1a", "court_p1_clf", "single_agg", "FS-MP", "xgboost"), out)
        self.assertIn("1 done", out)

    def test_queue_runs_subprocess_and_skips_done(self):
        res = os.path.join(self.tmp, "res_sub")
        args = ["--config", self.cfg_path, "--tier", "tier1a", "--only", "tier1a__court_p3_clf__single_agg__FS-M__",
                "--results-dir", res, "--data-dir", self.data]
        with contextlib.redirect_stdout(io.StringIO()):
            run_queue.main(args)
        runs = runner.read_runs(res)
        self.assertEqual(list(runs.status), ["done"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_queue.main(args)
        self.assertNotIn("START", buf.getvalue())
        self.assertEqual(len(runner.read_runs(res)), 1)


if __name__ == "__main__":
    unittest.main()

_ = (FastEncoder, fit_columns)
