"""court/report.py on a tiny synthetic results dir (DoD-7): it must build every section, mark missing /
failed jobs, and its vectorised bootstrap must equal evaluate.paired_bootstrap."""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from court import evaluate, report, runner


def _make_data(data_dir, n_seg=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2013-01-01")
    for s in range(n_seg):
        sid = "case%02d|p1" % s
        t0 = start + pd.Timedelta(days=45 * s)
        dates = [t0 + pd.Timedelta(days=int(d)) for d in np.cumsum(rng.integers(1, 60, 5))]
        exit_ = dates[-1] + pd.Timedelta(days=30)
        for i, d in enumerate(dates):  # 1 history event, then 4 in-phase events
            rows.append(dict(segment_id=sid, Activity="a%d" % (i % 3), timestamp=d.strftime("%Y-%m-%d"),
                             in_phase=int(i >= 1), n_hist=1, remaining_in_phase=(exit_ - d).days,
                             filing_year=t0.year, label="deviant" if s % 2 else "regular",
                             open_cases=float(100 + 10 * s + i), G_judge_open=float(5 * s + i)))
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(data_dir, "court_phase1.parquet"), index=False)
    with open(os.path.join(data_dir, "court_columns.json"), "w") as f:
        json.dump({"views": {"B0G": ["open_cases"], "G": ["G_judge_open"]}}, f)
    return df


def _pred_for(test_segs, task_type, rng):
    rows = []
    for k, sid in enumerate(test_segs):
        for j in range(1, 5):
            y = (k % 2) if task_type == "classification" else float(rng.integers(10, 400))
            rows.append(dict(segment_id=sid, prefix_len=j, y_true=y))
    p = pd.DataFrame(rows)
    if task_type == "classification":
        p["y_pred"] = np.clip(0.5 * p["y_true"] + rng.uniform(0, 0.6, len(p)), 0, 1)
    else:
        p["y_pred"] = p["y_true"] * rng.uniform(0.6, 1.4, len(p))
    return p


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="ppm_report_test_")
        cls.root = os.path.join(cls.tmp, "repo"); os.makedirs(os.path.join(cls.root, "docs"))
        cls.data = os.path.join(cls.tmp, "data"); os.makedirs(cls.data)
        cls.res = os.path.join(cls.tmp, "results")
        for sub in ("predictions", "importance", "params", "tables"):
            os.makedirs(os.path.join(cls.res, sub))
        df = _make_data(cls.data)
        with open(os.path.join(cls.root, "RUN_DECISIONS.md"), "w") as f:
            f.write("| Time (IDT) | ID | Who | Decision | Options | Choice | Why |\n|---|---|---|---|---|---|---|\n"
                    "| 01:00 | RD-01 | test | a decision | x / y | x | because |\n")
        with open(os.path.join(cls.root, "docs", "data_report.md"), "w") as f:
            f.write("# report\n\n## Limitations\n\n- synthetic limitation line\n\n## Next\n")
        cfg = {"budget": {"tuning": {"n_iter": 2, "cv_folds": 3, "max_train_segments": 100, "cap_minutes": 1,
                                     "gap": 3}, "final_fit": {"cap_minutes": 1}, "xgb_n_jobs": 1, "gap_train": 3,
                          "max_prefix_cap": 40},
               "data_dir": cls.data, "results_dir": cls.res, "tuned_fs": "FS-ALL",
               "tasks": ["court_p1_clf", "court_p1_reg"], "feature_sets": ["FS-B0", "FS-B0+G", "FS-ALL"],
               "tier_order": ["tier1a"], "tiers": {"tier1a": {"methods": ["single_agg"], "learners": ["xgboost"]}}}
        cls.cfg_path = os.path.join(cls.root, "experiments.yaml")
        with open(cls.cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        segs = sorted(df["segment_id"].unique())
        test_segs = segs[-10:]
        rng = np.random.default_rng(1)
        runner.append_run(cls.res, dict(job_id="tune__court_p1_clf__single_agg__xgboost", tier="tune",
                                        task="court_p1_clf", method="single_agg", fs="FS-ALL", learner="xgboost",
                                        status="done", runtime_s=3.0, flags=""))
        runner.append_run(cls.res, dict(job_id="tune__court_p1_reg__single_agg__xgboost", status="timeout",
                                        runtime_s=60.0, flags="timeout_halved"))
        for task in cfg["tasks"]:
            tt = report.task_type_of(task)
            for fs in cfg["feature_sets"]:
                jid = runner.make_job_id("tier1a", task, "single_agg", fs, "xgboost")
                if task == "court_p1_reg" and fs == "FS-ALL":
                    continue  # missing job -> "not run"
                if task == "court_p1_clf" and fs == "FS-B0+G":  # failed job
                    runner.append_run(cls.res, dict(runner.parse_job_id(jid), job_id=jid, status="failed",
                                                    runtime_s=1.0, flags="MemoryError"))
                    continue
                p = _pred_for(test_segs, tt, rng)
                p.to_csv(report.pred_path(cls.res, jid), index=False)
                evaluate.metrics_by_prefix(p, tt).to_csv(os.path.join(cls.res, "tables", jid + "__by_prefix.csv"),
                                                         index=False)
                pd.DataFrame({"view": ["M", "P", "G+"], "gain": [3.0, 2.0, 1.0],
                              "share": [0.5, 1 / 3, 1 / 6]}).to_csv(os.path.join(cls.res, "importance", jid + ".csv"),
                                                                    index=False)
                row = dict(runner.parse_job_id(jid), job_id=jid, status="done", runtime_s=2.5,
                           n_test_prefixes=len(p), n_test_segments=len(test_segs), max_prefix_len=4,
                           params_source="default" if tt == "regression" else "tuned",
                           flags="default_params" if tt == "regression" else "")
                row.update(evaluate.overall_metrics(p, tt))
                if tt == "regression":
                    row.update(baseline_mae_overall=150.0, baseline_mae_per_prefix=140.0)
                row.pop("n", None)
                runner.append_run(cls.res, row)
        cls.out = os.path.join(cls.tmp, "results.md")
        report.main(["--results-dir", cls.res, "--out", cls.out, "--config", cls.cfg_path, "--data-dir", cls.data,
                     "--root", cls.root, "--n-boot", "50"])
        with open(cls.out) as f:
            cls.md = f.read()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_sections_in_order(self):
        heads = ["## 1. Summary", "## 2. Data", "## 3. Protocol", "## 4. RQ1", "## 5. RQ2", "## 6. RQ3",
                 "## 7. Tier 2", "## 8. Drift", "## 9. Runtime", "## 10. Limitations", "## 11. Commands"]
        pos = [self.md.find(h) for h in heads]
        self.assertTrue(all(p >= 0 for p in pos), pos)
        self.assertEqual(pos, sorted(pos))
        summary = self.md.split("## 2.")[0]
        self.assertLessEqual(summary.count("\n- "), 8)

    def test_missing_and_failed_jobs_marked(self):
        self.assertIn("not run: not reached by the queue", self.md)
        self.assertIn("failed: MemoryError", self.md)
        self.assertIn("timeout_halved", self.md)
        self.assertIn("default_params", self.md)
        self.assertIn("RD-01", self.md)
        self.assertIn("synthetic limitation line", self.md)

    def test_outputs_written(self):
        for f in ["tables/bootstrap_delta_vs_FS-B0.csv", "tables/job_status.csv", "tables/runtime.csv",
                  "tables/drift_by_year.csv", "tables/restricted_le_2016.csv",
                  "tables/bootstrap_delta_vs_FS-B0__restricted_le2016.csv", "tables/g_drift_smd__p1.csv",
                  "tables/baseline_by_prefix__court_p1_reg.csv", "figures/auc_by_prefix_court_p1_clf.png",
                  "figures/mae_by_prefix_court_p1_reg.png"]:
            self.assertTrue(os.path.exists(os.path.join(self.res, f)), f)
        st = pd.read_csv(os.path.join(self.res, "tables", "job_status.csv"), keep_default_na=False)
        self.assertEqual(st.set_index("job_id").loc["tier1a__court_p1_reg__single_agg__FS-ALL__xgboost", "status"],
                         "not run")

    def test_numbers_come_from_files(self):
        runs = report.load_runs(self.res)
        auc = runs.set_index("job_id").loc["tier1a__court_p1_clf__single_agg__FS-B0__xgboost", "auc"]
        self.assertIn(report.fmt(auc), self.md)
        b = pd.read_csv(os.path.join(self.res, "tables", "bootstrap_delta_vs_FS-B0.csv"))
        self.assertEqual(len(b), 2)  # clf FS-ALL and reg FS-B0+G vs FS-B0

    def test_rerun_uses_cache(self):
        info = report.build(self.res, self.out, report.runner.load_config(self.cfg_path), self.cfg_path, self.data,
                            self.root, do_bootstrap=False, n_boot=50)
        self.assertEqual(info["n_boot_pairs"], 2)

    def test_fast_bootstrap_equals_reference(self):
        rng = np.random.default_rng(5)
        for tt in ("classification", "regression"):
            segs = ["s%d" % i for i in range(60)]
            a = _pred_for(segs, tt, rng); b = _pred_for(segs, tt, rng)
            b["y_true"] = a["y_true"]
            ref = evaluate.paired_bootstrap(a, b, tt, n_boot=100)
            got = report.fast_paired_bootstrap(a, b, tt, n_boot=100)
            for k in ("delta", "ci_low", "ci_high"):
                self.assertAlmostEqual(ref[k], got[k], places=9)


if __name__ == "__main__":
    unittest.main()
