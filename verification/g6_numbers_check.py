"""G6 (A4): recompute every number in results.md from the raw result files, with independent code.

    python verification/g6_numbers_check.py --results-md results.md --results-dir results \
        --pred-dir <main repo>/results/predictions --data-dir data/processed --out verification/g6_results.json

Independent of court/report.py and court/evaluate.py: metrics via sklearn on the saved predictions;
paired bootstrap by segment with integer resampling weights (own weighted-AUC kernel, own seed);
split, train prefixes (C13 baselines) and prefix-end dates rebuilt from the parquet logs with pandas.
Every check compares the *printed* value in results.md with the recomputed one at the printed precision.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, r2_score, roc_auc_score

TASK = {"T1": "court_p1_clf", "R1": "court_p1_reg", "T2": "court_p2_clf", "R2": "court_p2_reg",
        "T3": "court_p3_clf", "R3": "court_p3_reg"}
TIER = {"single_agg": "tier1a", "prefix_agg": "tier1b"}
MISMATCHES = []
N_CHECKED = {"n": 0}


def num(s):
    s = s.strip().replace(",", "").replace("*", "").replace("−", "-")
    return float(s)


def dec(s):
    s = s.strip().replace(",", "").replace("*", "")
    return len(s.split(".")[1]) if "." in s else 0


def check(where, printed, value, tol=None):
    """printed: string from results.md; value: recomputed float. Default tolerance = half a unit of
    the printed precision (+ float slack)."""
    N_CHECKED["n"] += 1
    p = num(printed)
    t = (0.5 * 10 ** -dec(printed) + 1e-9) if tol is None else tol
    if value is None or not np.isfinite(value) or abs(p - value) > t:
        MISMATCHES.append({"where": where, "printed": printed, "recomputed": value, "tol": t})
        return False
    return True


def md_sections(md):
    out, cur, buf = {}, None, []
    for line in md.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            if cur:
                out[cur] = "\n".join(buf)
            cur, buf = line.strip("# ").strip(), []
        else:
            buf.append(line)
    out[cur] = "\n".join(buf)
    return out


def md_tables(text):
    tabs, cur = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif cur:
            hdr, rows = cur[0], [r for r in cur[2:]]
            tabs.append(pd.DataFrame(rows, columns=hdr))
            cur = []
    return tabs


# ---------------------------------------------------------------- data rebuild (own code)
def load_phase(data_dir, phase, extra=()):
    cols = ["segment_id", "timestamp", "in_phase", "n_hist", "label", "remaining_in_phase", "filing_year"] + list(extra)
    df = pd.read_parquet(os.path.join(data_dir, "court_phase%d.parquet" % phase), columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def rebuild(df, task_type, gap_train=3, cap=40):
    """Own split + prefix rebuild. Returns dict with split info, train prefix targets, test prefix
    (segment, j, end date) table."""
    g = df.groupby("segment_id")
    inlen = g["in_phase"].sum()
    if task_type == "classification":
        pos = g["label"].first() == "deviant"
        q = inlen[pos].quantile(0.9)
    else:
        q = inlen.quantile(0.9)
    L = int(min(cap, int(np.ceil(q))))
    start = df[df["in_phase"] == 1].groupby("segment_id")["timestamp"].min().sort_values(kind="mergesort")
    ntr = int(0.8 * len(start))
    tr_ids, te_ids = set(start.index[:ntr]), set(start.index[ntr:])
    split_ts = start.iloc[ntr]
    d = df.sort_values(["segment_id", "timestamp"], kind="mergesort")
    tr = d[d["segment_id"].isin(tr_ids) & (d["timestamp"] < split_ts)]
    te = d[d["segment_id"].isin(te_ids)]

    def prefixes(x, gap):
        x = x.copy()
        x["pos"] = x.groupby("segment_id").cumcount() + 1           # 1-based position in the trace
        x["j"] = x["pos"] - x["n_hist"].astype(int)
        x = x[(x["j"] >= 1) & (x["j"] <= L) & ((x["j"] - 1) % gap == 0)]
        return x[["segment_id", "j", "timestamp", "remaining_in_phase", "label"]]
    return {"L": L, "split_ts": split_ts, "train": prefixes(tr, gap_train), "test": prefixes(te, 1),
            "n_train_seg": len(tr_ids), "n_test_seg": len(te_ids)}


# ---------------------------------------------------------------- metrics (own code)
def clf(y, p):
    return {"auc": roc_auc_score(y, p), "f1": f1_score(y, (p >= 0.5).astype(int)),
            "pr_auc": average_precision_score(y, p), "n": len(y)}


def reg(y, p):
    return {"mae": float(np.mean(np.abs(y - p))), "median_ae": float(np.median(np.abs(y - p))),
            "r2_log": r2_score(np.log1p(np.clip(y, 0, None)), np.log1p(np.clip(p, 0, None))), "n": len(y)}


class Boot:
    """Paired bootstrap by segment with multinomial integer weights (= resampling whole segments)."""

    def __init__(self, seg, y, px, pb, kind, n_boot=1000, seed=20261021):
        self.kind = kind
        u, inv = np.unique(seg, return_inverse=True)
        self.inv, self.nseg = inv, len(u)
        rng = np.random.default_rng(seed)
        self.W = np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(n_boot)])
        self.y = y.astype(float)
        if kind == "auc":
            self.ax = self._prep(px); self.ab = self._prep(pb)
        else:
            self.ex = np.bincount(inv, np.abs(y - px)); self.eb = np.bincount(inv, np.abs(y - pb))
            self.cnt = np.bincount(inv)

    def _prep(self, p):
        _, sc = np.unique(p, return_inverse=True)
        return sc

    def _wauc(self, sc, w):
        pos = np.bincount(sc, w * self.y); neg = np.bincount(sc, w * (1 - self.y))
        below = np.cumsum(neg) - neg
        return float(np.sum(pos * (below + 0.5 * neg)) / (pos.sum() * neg.sum()))

    def run(self):
        out = np.empty(len(self.W))
        for b, ws in enumerate(self.W):
            if self.kind == "auc":
                w = ws[self.inv].astype(float)
                out[b] = self._wauc(self.ax, w) - self._wauc(self.ab, w)
            else:
                n = ws @ self.cnt
                out[b] = (ws @ self.ex - ws @ self.eb) / n
        return np.percentile(out, [2.5, 97.5])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-md", default="results.md")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="verification/g6_results.json")
    a = ap.parse_args()
    md = open(a.results_md).read()
    S = md_sections(md)
    runs = pd.read_csv(os.path.join(a.results_dir, "runs.csv"), dtype=str, keep_default_na=False)
    runs_last = runs.drop_duplicates("job_id", keep="last").set_index("job_id")
    report = {}

    # ---- rebuild per phase: split, train-prefix targets (C13), prefix-end dates, filing year
    reb, fy = {}, {}
    for ph in (1, 2, 3):
        df = load_phase(a.data_dir, ph)
        assert df.loc[df["in_phase"] == 1, "remaining_in_phase"].notna().all()
        fy.update(df.groupby("segment_id")["filing_year"].first().to_dict())
        for kind, tt in (("clf", "classification"), ("reg", "regression")):
            reb[TASK_KEY(ph, kind)] = rebuild(df, tt)
        del df

    preds, met = {}, {}

    def pred(job):
        if job not in preds:
            f = os.path.join(a.pred_dir, job + ".csv.gz")
            preds[job] = pd.read_csv(f) if os.path.exists(f) else None
        return preds[job]

    def job_of(t, method, fs):
        return "%s__%s__%s__%s__xgboost" % (TIER[method], TASK[t], method, fs)

    # ---- prediction files vs rebuilt test prefixes
    align = {}
    for f in sorted(os.listdir(a.pred_dir)):
        job = f[:-len(".csv.gz")]
        p = pred(job)
        task = job.split("__")[1]
        r = reb[task]
        te = r["test"]
        m = p.merge(te, left_on=["segment_id", "prefix_len"], right_on=["segment_id", "j"], how="outer", indicator=True)
        yexp = (m["label"] == "deviant").astype(float) if task.endswith("clf") else m["remaining_in_phase"].astype(float)
        align[job] = {"rows": len(p), "expected": len(te), "unmatched": int((m["_merge"] != "both").sum()),
                      "y_true_mismatch": int((np.abs(m["y_true"] - yexp) > 1e-6).sum())}
        if align[job]["unmatched"] or align[job]["y_true_mismatch"]:
            MISMATCHES.append({"where": "predictions file " + job, "printed": "-", "recomputed": align[job]})
    report["prediction_alignment"] = align

    # ---- section 4 (RQ1) and 5 (RQ2) tables
    def metrics(job, kind):
        if job not in met:
            p = pred(job)
            met[job] = clf(p["y_true"].to_numpy(), p["y_pred"].to_numpy()) if kind == "clf" else \
                reg(p["y_true"].to_numpy(float), p["y_pred"].to_numpy(float))
        return met[job]

    sec4 = [k for k in S if k.startswith("4.")][0]
    t4 = md_tables(S[sec4])[0]
    for _, row in t4.iterrows():
        job = job_of(row["Task"], row["Method"], row["FS"])
        if row["AUC"] == "–":
            if pred(job) is not None:
                MISMATCHES.append({"where": "RQ1 %s: shown as not run but predictions exist" % job})
            continue
        mm = metrics(job, "clf")
        w = "RQ1 %s %s %s" % (row["Task"], row["Method"], row["FS"])
        check(w + " AUC", row["AUC"], mm["auc"]); check(w + " F1", row["F1@0.5"], mm["f1"])
        check(w + " PR-AUC", row["PR-AUC"], mm["pr_auc"]); check(w + " n", row["Test prefixes"], mm["n"])

    # C13 baselines (own): overall and per-length train medians
    base = {}
    for t in ("R1", "R2", "R3"):
        r = reb[TASK[t]]
        tr, te = r["train"], r["test"]
        med = float(np.median(tr["remaining_in_phase"]))
        per = tr.groupby("j")["remaining_in_phase"].median()
        known = np.sort(per.index.to_numpy())
        pj = np.array([per[known[known <= j][-1]] if (known <= j).any() else med for j in te["j"]])
        y = te["remaining_in_phase"].to_numpy(float)
        base[t] = {"overall": float(np.mean(np.abs(y - med))), "per_len": float(np.mean(np.abs(y - pj))),
                   "train_median": med, "n_test": len(te), "L": r["L"]}
    report["baselines"] = base

    sec5 = [k for k in S if k.startswith("5.")][0]
    t5, t5b = md_tables(S[sec5])[:2]
    for _, row in t5.iterrows():
        job = job_of(row["Task"], row["Method"], row["FS"])
        if row["MAE"] == "–":
            if pred(job) is not None:
                MISMATCHES.append({"where": "RQ2 %s: shown as not run but predictions exist" % job})
            continue
        mm = metrics(job, "reg")
        w = "RQ2 %s %s %s" % (row["Task"], row["Method"], row["FS"])
        check(w + " MAE", row["MAE"], mm["mae"]); check(w + " MedAE", row["MedAE"], mm["median_ae"])
        check(w + " R2log", row["R² (log)"], mm["r2_log"]); check(w + " n", row["Test prefixes"], mm["n"])
        check(w + " baseline overall", row["Baseline MAE (overall median)"], base[row["Task"]]["overall"])
        check(w + " baseline per-length", row["Baseline MAE (per-length median)"], base[row["Task"]]["per_len"])
        rr = runs_last.loc[job]
        check(w + " runs.csv MAE", row["MAE"], float(rr["mae"]))
        check(w + " runs.csv baseline", row["Baseline MAE (overall median)"], float(rr["baseline_mae_overall"]))
        check(w + " runs.csv baseline per-length", row["Baseline MAE (per-length median)"], float(rr["baseline_mae_per_prefix"]))
    for _, row in t5b.iterrows():
        t = row["Task"]
        check("C13 rebuilt %s overall" % t, row["Rebuilt MAE (overall median)"], base[t]["overall"])
        check("C13 runs.csv %s overall" % t, row["runs.csv MAE (overall median)"], base[t]["overall"])
        check("C13 rebuilt %s per-length" % t, row["Rebuilt MAE (per-length median)"], base[t]["per_len"])
        check("C13 %s n" % t, row["Test prefixes"], base[t]["n_test"])
    # RQ1 also against runs.csv
    for _, row in t4.iterrows():
        if row["AUC"] != "–":
            check("RQ1 runs.csv %s %s %s AUC" % (row["Task"], row["Method"], row["FS"]), row["AUC"],
                  float(runs_last.loc[job_of(row["Task"], row["Method"], row["FS"])]["auc"]))

    # ---- RQ3 bootstrap (section 6) and section 8.1 restricted
    end_date = {}
    for t, task in TASK.items():
        te = reb[task]["test"]
        end_date[task] = te.set_index(["segment_id", "j"])["timestamp"]

    def delta(t, method, fs, restricted=False):
        kind = "auc" if t.startswith("T") else "mae"
        px, pb = pred(job_of(t, method, fs)), pred(job_of(t, method, "FS-B0"))
        m = px.merge(pb[["segment_id", "prefix_len", "y_pred"]], on=["segment_id", "prefix_len"], suffixes=("_x", "_b"))
        assert len(m) == len(px) == len(pb)
        if restricted:
            ed = end_date[TASK[t]].reindex(pd.MultiIndex.from_arrays([m["segment_id"], m["prefix_len"]])).to_numpy()
            m = m[ed <= np.datetime64("2016-12-31")]
        y, a_, b_ = m["y_true"].to_numpy(float), m["y_pred_x"].to_numpy(float), m["y_pred_b"].to_numpy(float)
        if kind == "auc":
            pt = roc_auc_score(y, a_) - roc_auc_score(y, b_)
        else:
            pt = np.mean(np.abs(y - a_)) - np.mean(np.abs(y - b_))
        lo, hi = Boot(m["segment_id"].to_numpy(), y, a_, b_, kind).run()
        return pt, lo, hi, m["segment_id"].nunique(), len(m)

    cache = {}

    def D(t, method, fs, restricted=False):
        k = (t, method, fs, restricted)
        if k not in cache:
            cache[k] = delta(*k)
        return cache[k]

    def check_ci(w, txt, d, t):
        # txt like "-0.011* [-0.014, -0.008]" or separate
        mm = re.match(r"\s*([-0-9.]+)\*?\s*\[([-0-9.]+),\s*([-0-9.]+)\]", txt)
        tol = 0.005 if t.startswith("T") else 1.0
        check(w + " point", mm.group(1), d[0])
        check(w + " CI low", mm.group(2), d[1], tol + 0.5 * 10 ** -dec(mm.group(2)))
        check(w + " CI high", mm.group(3), d[2], tol + 0.5 * 10 ** -dec(mm.group(3)))
        sig = "*" in txt.split("[")[0]
        if sig != (d[1] > 0 or d[2] < 0) and not (abs(d[1]) < tol or abs(d[2]) < tol):
            MISMATCHES.append({"where": w + " significance star", "printed": txt, "recomputed": d[:3]})

    sec6 = [k for k in S if k.startswith("6.")][0]
    t6 = md_tables(S[sec6])[0]
    boot_rows = []
    for _, row in t6.iterrows():
        d = D(row["Task"], row["Method"], row["FS"])
        w = "RQ3 %s %s %s" % (row["Task"], row["Method"], row["FS"])
        check_ci(w, row["Δ vs FS-B0"] + " " + row["95% CI"], d, row["Task"])
        check(w + " segments", row["Test segments"], d[3])
        boot_rows.append({"row": w, "printed": row["Δ vs FS-B0"] + " " + row["95% CI"],
                          "recomputed": [round(x, 4) for x in d[:3]]})
    report["bootstrap"] = boot_rows

    sec81 = [k for k in S if k.startswith("8.1")][0]
    t81 = md_tables(S[sec81])[0]
    restr_rows = []
    for _, row in t81.iterrows():
        t, method, fs = row["Task"], row["Method"], row["FS"]
        job = job_of(t, method, fs)
        p = pred(job)
        ed = end_date[TASK[t]].reindex(pd.MultiIndex.from_arrays([p["segment_id"], p["prefix_len"]])).to_numpy()
        keep = ed <= np.datetime64("2016-12-31")
        kind = "clf" if t.startswith("T") else "reg"
        f = clf if kind == "clf" else reg
        mall = metrics(job, kind)
        mres = f(p["y_true"].to_numpy(float)[keep], p["y_pred"].to_numpy(float)[keep])
        key = "auc" if kind == "clf" else "mae"
        w = "8.1 %s %s %s" % (t, method, fs)
        check(w + " all", row["All test prefixes"], mall[key])
        check(w + " restricted", row["Prefixes ≤ 2016-12-31"], mres[key])
        kept, alln = row["Prefixes kept / all"].split("/")
        check(w + " kept", kept, keep.sum()); check(w + " all n", alln, len(p))
        if fs != "FS-B0":
            check_ci(w + " Δ all", row["Δ vs FS-B0, all [95% CI]"], D(t, method, fs), t)
            dr = D(t, method, fs, True)
            check_ci(w + " Δ restricted", row["Δ vs FS-B0, ≤ 2016 [95% CI]"], dr, t)
            restr_rows.append({"row": w, "printed": row["Δ vs FS-B0, ≤ 2016 [95% CI]"],
                               "recomputed": [round(x, 4) for x in dr[:3]]})
    report["restricted_bootstrap"] = restr_rows

    # ---- section 1 summary bullets
    s1 = S["1. Summary"]
    bul = {}
    for t in ("T1", "T2"):
        cand = {(r["Method"], r["FS"]): num(r["AUC"]) for _, r in t4[t4["Task"] == t].iterrows() if r["AUC"] != "–"}
        exact = {k: metrics(job_of(t, *k), "clf")["auc"] for k in cand}
        best = max(exact, key=exact.get)
        mm = re.search(r"RQ1 %s: best AUC ([0-9.]+) \((\S+), (\S+)\); paper baseline FS-B0 with the same method: ([0-9]+\.[0-9]+)" % t, s1)
        check("summary %s best AUC" % t, mm.group(1), exact[best])
        if (mm.group(3), mm.group(2)) != best:
            MISMATCHES.append({"where": "summary %s best config" % t, "printed": mm.groups()[1:3], "recomputed": best})
        check("summary %s FS-B0 same method" % t, mm.group(4), exact[(best[0], "FS-B0")])
        bul[t] = {"best": best, "auc": exact[best]}
    for t in ("R1", "R2"):
        cand = {(r["Method"], r["FS"]) for _, r in t5[t5["Task"] == t].iterrows() if r["MAE"] != "–"}
        exact = {k: metrics(job_of(t, *k), "reg")["mae"] for k in cand}
        best = min(exact, key=exact.get)
        mm = re.search(r"RQ2 %s: best MAE ([0-9.]+) days \((\S+), (\S+)\) vs\. ([0-9.]+) days for the per-length median baseline and ([0-9]+\.[0-9]+) for the overall median" % t, s1)
        check("summary %s best MAE" % t, mm.group(1), exact[best])
        if (mm.group(3), mm.group(2)) != best:
            MISMATCHES.append({"where": "summary %s best config" % t, "printed": mm.groups()[1:3], "recomputed": best})
        check("summary %s per-length baseline" % t, mm.group(4), base[t]["per_len"])
        check("summary %s overall baseline" % t, mm.group(5), base[t]["overall"])
        bul[t] = {"best": best, "mae": exact[best]}
    for label, restricted in (("RQ3, global extras", False), ("Same comparison restricted", True)):
        line = [l for l in s1.splitlines() if label in l][0]
        for t in ("T1", "R1", "T2", "R2", "T3", "R3"):
            mm = re.search(r"%s Δ(?:AUC|MAE) ([-0-9.]+ \[[-0-9.]+, [-0-9.]+\])" % t, line)
            check_ci("summary %s %s" % (label, t), mm.group(1), D(t, "single_agg", "FS-B0+G", restricted), t)
    jobs = re.search(r"Jobs: (\d+) of (\d+) Tier-1 fits done; (\d+) of (\d+) Tier-1 tuning jobs done; (\d+) jobs failed or timed out; (\d+) Tier-1 fits not run", s1)
    fits = runs_last[runs_last["tier"].isin(["tier1a", "tier1b"])]
    tunes = runs_last[runs_last.index.str.startswith("tune__") & ~runs_last.index.str.contains("state_agg|cluster_agg|laststate")]
    got = [int((fits["status"] == "done").sum()), 72, int((tunes["status"] == "done").sum()), 12,
           int(runs_last["status"].isin(["failed", "timeout"]).sum()), 72 - len(fits)]
    for i, g in enumerate(got):
        check("summary jobs field %d" % (i + 1), jobs.group(i + 1), g)
    report["summary"] = {"best": {k: str(v) for k, v in bul.items()}, "jobs": got}

    # ---- section 2 vs data_report.json and the parquet files
    dr = json.load(open(os.path.join(a.data_dir, "data_report.json")))
    s2 = md_tables(S["2. Data and cleaning"])
    cnt = s2[1].set_index("Step")["Count"]
    pf = dr["phase_fixes"]
    exp2 = {"Event-log rows (full log)": [dr["n_events_full_log"]],
            "Event rows of case-table cases": [dr["n_events_case_table_cases"]],
            "Cases in case table": [dr["n_cases_table"]],
            "Cases whose duplicate sum ≠ total_activity_count": [dr["dup_sum_mismatch_cases"]],
            "D1 cases (all / in window)": [pf["flags_all"]["D1"], pf["flags_window"]["D1"]],
            "D2 cases (all / in window)": [pf["flags_all"]["D2"], pf["flags_window"]["D2"]],
            "D3 cases dropped (all / in window)": [pf["flags_all"]["D3"], pf["flags_window"]["D3"]],
            "D4 cases (all / in window)": [pf["flags_all"]["D4"], pf["flags_window"]["D4"]],
            "Cases after D3 (used for global features)": [pf["after_D3_all_cases"]],
            "D5 window cases, before → after D3": [pf["window_before_fixes"]["cases"], pf["window_after_fixes"]["cases"]],
            "D2 phase-1 exit moved to trial start (RD-18), window": [pf["D2_phase1_exit_changed_window"]],
            "Phase-1 segments built / dropped empty (D7)": [dr["segments"]["built"]["1"], dr["traces"]["p1_empty_dropped"]],
            "Phase-2 segments built / dropped empty (D7)": [dr["segments"]["built"]["2"], dr["traces"]["p2_empty_dropped"]],
            "Phase-3 segments built / dropped empty (D7)": [dr["segments"]["built"]["3"], dr["traces"]["p3_empty_dropped"]]}
    for k, vals in exp2.items():
        parts = re.split(r"/|→", cnt[k])
        for pv, ev in zip(parts, vals):
            check("section 2 " + k, pv, ev)
    inp = s2[0].set_index("File")
    for f, v in dr["inputs"].items():
        key = f if f in inp.index else [i for i in inp.index if i in f or f in i][0]
        check("section 2 input bytes " + f, inp.loc[key, "Bytes"], v["size"])
        check("section 2 input lines " + f, inp.loc[key, "Lines (wc -l)"], v["lines"])
        if inp.loc[key, "MD5"].strip("`") != v["md5"]:
            MISMATCHES.append({"where": "section 2 md5 " + f})
    seg = s2[2].set_index("Phase")
    for ph in (1, 2, 3):
        jp = dr["phases"]["p%d" % ph]
        df = load_phase(a.data_dir, ph)
        g = df.groupby("segment_id")
        own = {"Segments": g.ngroups, "Trace rows": len(df), "In-phase rows (prefixes)": int(df["in_phase"].sum()),
               "Label rate (moved on)": float((g["label"].first() == "deviant").mean()),
               "In-phase length q50": g["in_phase"].sum().quantile(0.5), "q90": g["in_phase"].sum().quantile(0.9),
               "q90 positives": g["in_phase"].sum()[g["label"].first() == "deviant"].quantile(0.9)}
        jv = {"Segments": jp["segments"], "Trace rows": jp["rows"], "In-phase rows (prefixes)": jp["in_phase_rows"],
              "Label rate (moved on)": jp["label_rate_deviant"], "In-phase length q50": jp["in_phase_len_quantiles"]["0.5"],
              "q90": jp["in_phase_len_quantiles"]["0.9"], "q90 positives": jp["in_phase_len_q90_positive"]}
        for c in own:
            check("section 2 p%d %s (json)" % (ph, c), seg.loc["p%d" % ph, c], jv[c])
            check("section 2 p%d %s (parquet)" % (ph, c), seg.loc["p%d" % ph, c], own[c])
        del df

    # ---- section 8 drift by filing year
    t8 = md_tables(S["8. Drift check — test performance by filing year"])[0]
    for _, row in t8.iterrows():
        t, yr = row["Task"], int(row["Filing year"])
        for fs in ("FS-B0", "FS-B0+G", "FS-ALL"):
            p = pred(job_of(t, "single_agg", fs))
            sel = p["segment_id"].map(fy) == yr
            q = p[sel]
            if t.startswith("T"):
                v = roc_auc_score(q["y_true"], q["y_pred"]) if q["y_true"].nunique() == 2 else np.nan
            else:
                v = float(np.mean(np.abs(q["y_true"] - q["y_pred"])))
            if row[fs] in ("–", "nan"):
                continue
            check("8 drift %s %d %s" % (t, yr, fs), row[fs], v)
        check("8 drift %s %d segments" % (t, yr), row["Test segments"], q["segment_id"].nunique())

    # ---- importance by view (section 6) vs results/importance
    ti = md_tables(S[sec6])[1]
    for _, row in ti.iterrows():
        f = os.path.join(a.results_dir, "importance", job_of(row["Task"], row["Method"], row["FS"]) + ".csv")
        imp = pd.read_csv(f).set_index("view")["share"]
        tot = imp.sum()
        for v in ("M", "P", "P+", "G", "G+"):
            if row[v] == "–":
                if v in imp.index and imp[v] > 0:
                    MISMATCHES.append({"where": "importance %s %s" % (f, v)})
                continue
            check("importance %s %s %s %s" % (row["Task"], row["Method"], row["FS"], v), row[v], imp.get(v, 0) / tot)

    report["n_checked"] = N_CHECKED["n"]
    report["mismatches"] = MISMATCHES
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("checked", N_CHECKED["n"], "values;", len(MISMATCHES), "mismatches")
    for m in MISMATCHES[:60]:
        print(m)


def TASK_KEY(ph, kind):
    return "court_p%d_%s" % (ph, kind)


if __name__ == "__main__":
    main()
