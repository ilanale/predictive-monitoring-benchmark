"""WS1 tests (stdlib unittest): D1-D4/D6 labels, D7, regression label, G brute force, D8 assert."""
import types
import unittest

import numpy as np
import pandas as pd

from court import court_confs
from court.data_prep import build, export, global_features, phase_fixes
from court.data_prep.segments import DAY0

NaT = pd.NaT


def case(ucid, filing, term, p2=None, p2x=None, p3=None, p3x=None, p2_entered=None, p3_completed=False,
         p1_completed=None, judge="J1", mag=None, ctype="cv"):
    p2_entered = (p2 is not None) if p2_entered is None else p2_entered
    p3_entered = p3 is not None
    p1x = p2 or p3 or term
    return {
        "ucid": ucid, "filing_date": filing, "terminating_date": term,
        "phase1_entered": True, "phase1_completed": p2_entered if p1_completed is None else p1_completed,
        "phase1_entry_date": filing, "phase1_exit_date": p1x,
        "phase2_entered": p2_entered, "phase2_completed": p3_entered,
        "phase2_entry_date": p2, "phase2_exit_date": (p2x or (p3 or term)) if (p2 or p2_entered) else None,
        "phase3_entered": p3_entered, "phase3_completed": p3_completed,
        "phase3_entry_date": p3, "phase3_exit_date": p3x or (term if p3 else None),
        "case_type": ctype, "city": "Chicago", "nature_suit": "x", "District_Judge": judge,
        "Magistrate_Judge": mag, "Defendants_highest_offense_opening": "Not Applicable",
        **{c: 0 for c in export.M_NUM}, **{c: 1 for c in export.S_NUM},
        "resolution_type": "Court decision", "total_activity_count": 0,
    }


def make_cases(rows):
    c = pd.DataFrame(rows)
    for col in [x for x in c.columns if x.endswith("_date")]:
        c[col] = pd.to_datetime(c[col])
    return c


def ev(ucid, date, act="motion", judge="(('SJ000001', 'District Judge'),)", flag=False):
    return {"ucid": ucid, "date_filed": date, "Activity": act, "event_judge": judge,
            "attribute_duplicates": 1, "attribute_scheduling": flag, "attribute_opening": False}


def make_events(rows):
    e = pd.DataFrame(rows)
    e["row"] = np.arange(len(e))
    return e


def run(cases, events):
    rep = {}
    old = export.RARE_MIN
    export.RARE_MIN = 0
    try:
        tables, spec = build.build_tables(cases, events, rep, log=lambda *a: None)
    finally:
        export.RARE_MIN = old
    return tables, spec, rep


def base_fixture():
    rows = [
        # normal: p1 -> p2 -> p3, completes
        case("A", "2005-01-01", "2007-01-01", p2="2005-03-01", p3="2006-01-01", p3_completed=True),
        # D1: flagged phase 2, no entry date, no trial
        case("D1", "2005-01-01", "2005-06-01", p2=None, p2_entered=True),
        # D2: trial with no answer
        case("D2", "2005-01-01", "2006-06-01", p2=None, p2_entered=True, p3="2005-08-01", p3_completed=False),
        # D3: inverted phase 2
        case("D3", "2005-01-01", "2006-01-01", p2="2005-06-01", p2x="2005-05-01"),
        # D4: answered on termination day: phase1_completed but phase2_entered False
        case("D4", "2005-01-01", "2005-04-01", p2_entered=False, p1_completed=True),
        # out of window
        case("OLD", "2001-01-01", "2005-04-01"),
    ]
    events = []
    for u in ("A", "D1", "D2", "D3", "D4", "OLD"):
        events.append(ev(u, "2005-01-01", "complaint"))
    events += [ev("A", "2005-02-01"), ev("A", "2005-03-01", "answer"), ev("A", "2005-03-01"),
               ev("A", "2005-06-01"), ev("A", "2006-01-01", "trial"), ev("A", "2006-06-01"),
               ev("A", "2007-01-01", "verdict")]
    events += [ev("D1", "2005-02-01"), ev("D1", "2005-06-01", "judgment")]
    events += [ev("D2", "2005-03-01"), ev("D2", "2005-08-01", "trial"), ev("D2", "2005-09-01")]
    events += [ev("D4", "2005-02-01"), ev("D4", "2005-04-01", "answer")]
    return make_cases(rows), make_events(events)


def seg_labels(tables):
    out = {}
    for k, df in tables.items():
        for sid, lab in df.groupby("segment_id").label.first().items():
            out[sid] = lab
    return out


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.cases, self.events = base_fixture()
        self.tables, self.spec, self.rep = run(self.cases, self.events)
        self.labels = seg_labels(self.tables)

    def test_normal_case(self):
        self.assertEqual(self.labels["A|p1"], "deviant")
        self.assertEqual(self.labels["A|p2"], "deviant")
        self.assertEqual(self.labels["A|p3"], "deviant")

    def test_D1(self):
        self.assertEqual(self.labels["D1|p1"], "regular")
        self.assertNotIn("D1|p2", self.labels)

    def test_D2(self):
        self.assertEqual(self.labels["D2|p1"], "deviant")
        self.assertNotIn("D2|p2", self.labels)
        self.assertEqual(self.labels["D2|p3"], "regular")
        p3 = self.tables[3][self.tables[3].segment_id == "D2|p3"]
        self.assertEqual(p3.PP_prev_phase_days_2.iloc[0], 0)

    def test_D3_dropped(self):
        self.assertFalse(any(s.startswith("D3|") for s in self.labels))

    def test_D4(self):
        self.assertEqual(self.labels["D4|p1"], "regular")
        self.assertNotIn("D4|p2", self.labels)

    def test_D5_window(self):
        self.assertFalse(any(s.startswith("OLD|") for s in self.labels))

    def test_D7_exit_day_excluded(self):
        p1 = self.tables[1][self.tables[1].segment_id == "A|p1"]
        # phase-1 exit = 2005-03-01: both events on that day are excluded
        self.assertEqual(list(p1.timestamp), ["2005-01-01", "2005-02-01"])
        self.assertTrue((p1.timestamp < p1.exit_date).all())
        for df in self.tables.values():
            self.assertTrue((df.remaining_in_phase >= 1).all())
        # phase 2 trace holds history (D9) and the entry-day events are in phase
        p2 = self.tables[2][self.tables[2].segment_id == "A|p2"]
        self.assertEqual(p2.n_hist.iloc[0], 2)
        self.assertEqual(list(p2.in_phase), [0, 0, 1, 1, 1])

    def test_regression_label(self):
        for df in self.tables.values():
            exp = (pd.to_datetime(df.exit_date) - pd.to_datetime(df.timestamp)).dt.days
            self.assertTrue((df.remaining_in_phase.values == exp.values).all())
        p2 = self.tables[2][self.tables[2].segment_id == "A|p2"]
        # exit of phase 2 = 2006-01-01; events 2005-01-01, 02-01, 03-01, 03-01, 06-01
        self.assertEqual(list(p2.remaining_in_phase), [365, 334, 306, 306, 214])
        self.assertGreater(p2.remaining_in_phase.nunique(), 1)
        self.assertEqual(p2.remaining_in_phase.iloc[-1], (pd.Timestamp("2006-01-01") - pd.Timestamp("2005-06-01")).days)

    def test_D7_perturb_future(self):
        """Changing events after the prefix leaves that prefix's feature values unchanged."""
        e2 = self.events.copy()
        late = (e2.ucid == "A") & (e2.date_filed > "2005-03-01")
        e2.loc[late, "Activity"] = "order"
        e2.loc[late, "event_judge"] = "(('SJ000009', 'Magistrate Judge'),)"
        e2.loc[late, "attribute_scheduling"] = True
        extra = make_events([ev("A", "2005-07-01", "notice"), ev("A", "2005-03-02", "notice")])
        e2 = pd.concat([e2, extra.assign(row=extra.row + 1000)], ignore_index=True)
        t2, _, _ = run(self.cases, e2)
        feats = [c for v in ("M", "P", "PP", "B0G", "G", "S") for c in self.spec["views"][v]]
        for k in (1, 2):
            a = self.tables[k][self.tables[k].ucid == "A"]
            b = t2[k][t2[k].ucid == "A"]
            a = a[a.timestamp <= "2005-03-01"]; b = b[b.timestamp <= "2005-03-01"]
            pd.testing.assert_frame_equal(a[feats].reset_index(drop=True), b[feats].reset_index(drop=True))


class TestGlobalBruteForce(unittest.TestCase):
    def test_global_matches_brute_force(self):
        rng = np.random.default_rng(1)
        rows = []
        for i in range(40):
            f = pd.Timestamp("2004-01-01") + pd.Timedelta(days=int(rng.integers(0, 400)))
            L = int(rng.integers(1, 300))
            term = f + pd.Timedelta(days=L)
            p2 = f + pd.Timedelta(days=int(rng.integers(0, L))) if rng.random() < 0.6 else None
            p3 = None
            if p2 is not None and rng.random() < 0.4 and (term - p2).days > 1:
                p3 = p2 + pd.Timedelta(days=int(rng.integers(1, (term - p2).days)))
            rows.append(case(f"c{i}", f, term, p2=p2, p3=p3,
                             judge=[None, "J1", "J2", "J3"][i % 4], mag=[None, "M1"][i % 2],
                             ctype=["cv", "cr"][int(rng.integers(0, 2))]))
        c = make_cases(rows)
        n = 300
        qc = rng.integers(0, len(c), n)
        qd = (pd.Timestamp("2004-01-01") - DAY0).days + rng.integers(-10, 800, n)
        qp = rng.integers(1, 4, n)
        G = global_features.global_features(c, qc, qd, qp)
        day = lambda x: np.nan if pd.isna(x) else (x - DAY0).days
        C = [{k: (day(v) if isinstance(v, pd.Timestamp) or v is NaT else v) for k, v in r.items()} for r in c.to_dict("records")]

        def isopen(r, t, a="filing_date", b="terminating_date"):
            s, x = r[a], r[b]
            return (not np.isnan(s)) and (not np.isnan(x)) and x > s and s <= t < x

        for j in range(n):
            me, t, k = C[qc[j]], qd[j], qp[j]
            others = [r for i, r in enumerate(C) if i != qc[j]]
            jm = me["District_Judge"] is None
            mine = [r for r in others if r["District_Judge"] == me["District_Judge"]] if not jm else []
            exp = {
                "open_cases": sum(isopen(r, t) for r in others),
                "G_court_open_same_type": sum(isopen(r, t) for r in others if r["case_type"] == me["case_type"]),
                "G_court_filed_last30": sum(t - 30 < r["filing_date"] <= t for r in C),
                "G_court_closed_last90": sum(t - 90 < r["terminating_date"] <= t for r in C),
                "G_judge_open": sum(isopen(r, t) for r in mine),
                "G_judge_open_same_type": sum(isopen(r, t) for r in mine if r["case_type"] == me["case_type"]),
                "G_judge_filed_last90": 0 if jm else sum(t - 90 < r["filing_date"] <= t for r in C if r["District_Judge"] == me["District_Judge"]),
                "G_mag_open": 0 if me["Magistrate_Judge"] is None else sum(isopen(r, t) for r in others if r["Magistrate_Judge"] == me["Magistrate_Judge"]),
            }
            for p in (1, 2, 3):
                exp[f"G_court_open_phase{p}"] = sum(isopen(r, t, f"phase{p}_entry_date", f"phase{p}_exit_date") for r in others)
                exp[f"G_judge_open_phase{p}"] = sum(isopen(r, t, f"phase{p}_entry_date", f"phase{p}_exit_date") for r in mine)
            exp["G_judge_open_same_phase"] = exp[f"G_judge_open_phase{k}"]
            closed = [r["terminating_date"] - r["filing_date"] for r in C
                      if not jm and r["District_Judge"] == me["District_Judge"] and r["terminating_date"] < t]
            exp["G_judge_hist_n_closed"] = len(closed)
            exp["G_judge_hist_median_los"] = float(np.median(closed)) if closed else 0.0
            for col, v in exp.items():
                self.assertAlmostEqual(float(G[col].iloc[j]), float(v), msg=f"{col} query {j}")


class TestD8(unittest.TestCase):
    def _spec(self):
        _, spec, _ = run(*base_fixture())
        return spec

    def test_real_spec_passes_and_has_no_outcomes(self):
        spec = self._spec()
        export.assert_no_leaky_features(spec)
        allf = [c for v in spec["views"].values() for c in v]
        for bad in ("resolution_type", "disposition_group", "total_activity_count", "case_status"):
            self.assertFalse(any(bad in c for c in allf))
        self.assertTrue(all(c.startswith("S_") for c in spec["views"]["S"]))

    def test_outcome_column_rejected(self):
        for bad in ("M_resolution_type", "M_is_case_trial_only", "M_activity_count_motion",
                    "M_Defendants_terminated_counts", "M_phase1_los", "M_total_case_los"):
            spec = self._spec()
            spec["views"]["M"].append(bad); spec["static_num"].append(bad)
            with self.assertRaises(AssertionError):
                export.assert_no_leaky_features(spec)

    def test_suspect_only_in_S(self):
        spec = self._spec()
        spec["views"]["M"].append("M_Defendants_counsels_count")
        with self.assertRaises(AssertionError):
            export.assert_no_leaky_features(spec)


class TestConfs(unittest.TestCase):
    def test_register(self):
        import json, os, tempfile
        _, spec, _ = run(*base_fixture())
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cols.json")
            with open(p, "w") as f:
                json.dump(spec, f)
            dc = types.SimpleNamespace(**{k: {} for k in (
                "filename", "case_id_col", "activity_col", "resource_col", "timestamp_col", "label_col",
                "pos_label", "neg_label", "dynamic_cat_cols", "static_cat_cols", "dynamic_num_cols", "static_num_cols")})
            names = court_confs.register(dc, p, d)
        self.assertEqual(len(names), 36)
        self.assertIn("court_p1_clf__FS-B0", names)
        b0 = dc.dynamic_num_cols["court_p1_clf__FS-B0"]
        self.assertIn("open_cases", b0)
        self.assertFalse(any(c.startswith(("G_", "PP_")) for c in b0))
        self.assertEqual(dc.task_type["court_p2_reg__FS-ALL"], "regression")
        self.assertFalse(any(c.startswith("S_") for c in dc.static_num_cols["court_p1_clf__FS-ALL"]))


if __name__ == "__main__":
    unittest.main()
