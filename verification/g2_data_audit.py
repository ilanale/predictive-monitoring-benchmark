"""G2 data audit (A4 Verifier) -- independent re-derivation from raw inputs.

Written without reading court/data_prep/, court/court_confs.py or tests/test_data_prep.py.
Everything here is re-derived from data/raw/* using DECISIONS.md D1-D9, court/SCHEMA.md
and RUN_DECISIONS.md RD-06..RD-14.

Usage: python verification/g2_data_audit.py [--cache DIR]
Writes verification/g2_results.json (machine-readable findings) and prints a summary.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
WIN = ("2003-01-01", "2016-12-31")
SEED = 7                                     # Check-1 sample seed
FSEED = int(os.environ.get("G2_FSEED", "7"))  # feature-cell seed (re-audit: 2026)
T0 = time.time()
RES = {}


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- raw inputs
def load_log(cache):
    path = os.path.join(cache, "g2_log.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    hdr = pd.read_csv(os.path.join(RAW, "df_event_log.csv"), nrows=0).columns.tolist()
    use = ["ucid", "date_filed", "Activity", "event_judge"] + [c for c in hdr if c.startswith("attribute_")]
    ids = set(pd.read_csv(os.path.join(RAW, "case_level_df.csv"), usecols=["ucid"])["ucid"])
    parts, off = [], 0
    for ch in pd.read_csv(os.path.join(RAW, "df_event_log.csv"), usecols=use, chunksize=500_000,
                          dtype={"event_judge": str}):
        ch["rowno"] = np.arange(off, off + len(ch))
        off += len(ch)
        parts.append(ch[ch.ucid.isin(ids)])
    df = pd.concat(parts, ignore_index=True)
    os.makedirs(cache, exist_ok=True)
    df.to_parquet(path)
    return df


JUDGE_RE = re.compile(r"\('([^']*)', '([^']*)'\)")


def parse_judges(s):
    """list of (id, role) from the event_judge tuple string."""
    if not isinstance(s, str):
        return []
    return JUDGE_RE.findall(s)


def fixed_cases(c):
    """Apply D1-D4 (own reading of DECISIONS.md). Returns per-case fixed phase intervals."""
    d = lambda col: pd.to_datetime(c[col])
    c = c.copy()
    c["D1"] = c.phase1_exit_type.eq("summary_judgment")          # p2 entered, no p2 entry, no p3
    c["D2"] = c.phase1_exit_type.eq("trial")                     # p3 entered, no p2 entry
    c["D3"] = d("phase2_exit_date") < d("phase2_entry_date")
    c["D4"] = c.phase1_completed & ~c.phase2_entered
    # sanity: definitions match DECISIONS.md wording/counts
    assert (c.D1 == (c.phase2_entered & c.phase2_entry_date.isna() & ~c.phase3_entered)).all()
    assert (c.D2 == (c.phase3_entered & c.phase2_entry_date.isna())).all()
    RES["D_counts_all"] = {k: int(c[k].sum()) for k in ["D1", "D2", "D3", "D4"]}
    c["filing"] = d("filing_date")
    c["term"] = d("terminating_date")
    c["e1"] = d("phase1_entry_date")
    c["x1"] = d("phase1_exit_date")
    # D2: phase 2 has zero length -> phase 1 ends when phase 3 starts
    c.loc[c.D2, "x1"] = d("phase3_entry_date")[c.D2]
    c["x1_raw"] = d("phase1_exit_date")
    has2 = c.phase2_entered & ~c.D1 & ~c.D2
    c["e2"] = d("phase2_entry_date").where(has2)
    c["x2"] = d("phase2_exit_date").where(has2)
    c["e3"] = d("phase3_entry_date").where(c.phase3_entered)
    c["x3"] = d("phase3_exit_date").where(c.phase3_entered)
    c["lab1"] = (c.phase2_entered & ~c.D1).astype(int)          # D1 ->0, D2 ->1, D4 ->0
    c["lab2"] = c.phase3_entered.astype(int)
    c["lab3"] = c.phase3_completed.astype(int)
    c["has1"], c["has2"], c["has3"] = True, has2, c.phase3_entered
    c["inwin"] = c.filing.between(pd.Timestamp(WIN[0]), pd.Timestamp(WIN[1]))
    return c


# ---------------------------------------------------------------- outputs
SUMCOLS = ["segment_id", "ucid", "timestamp", "label", "remaining_in_phase", "in_phase", "n_hist",
           "entry_date", "exit_date", "filing_year", "event_nr", "Activity"]


def out_summary(k):
    o = pd.read_parquet(os.path.join(PROC, f"court_phase{k}.parquet"), columns=SUMCOLS)
    o["pos"] = o.groupby("segment_id", sort=False).cumcount()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("G2_CACHE", os.path.join(ROOT, "verification", ".cache")))
    args = ap.parse_args()

    cols = json.load(open(os.path.join(PROC, "court_columns.json")))
    case = pd.read_csv(os.path.join(RAW, "case_level_df.csv"), index_col=0)
    assert len(case) == 155_525
    c = fixed_cases(case)
    log("case table", len(c), RES["D_counts_all"])
    L = load_log(args.cache)
    L["date"] = pd.to_datetime(L.date_filed)
    L = L.sort_values(["ucid", "date", "rowno"], kind="stable").reset_index(drop=True)
    log("log", len(L), L.ucid.nunique())
    cw = c[c.inwin & ~c.D3]
    RES["window_cases_after_D3"] = int(len(cw))
    RES["D_counts_window"] = {k: int(c[c.inwin][k].sum()) for k in ["D1", "D2", "D3", "D4"]}

    # ------------------------------------------------ my segments (all, before D7)
    segs = []
    for k in (1, 2, 3):
        s = cw[cw[f"has{k}"]][["ucid", f"e{k}", f"x{k}", f"lab{k}", "D1", "D2", "D4"]].copy()
        s.columns = ["ucid", "entry", "exit", "lab", "D1", "D2", "D4"]
        s["phase"] = k
        s["segment_id"] = s.ucid + f"|p{k}"
        segs.append(s)
    segs = pd.concat(segs, ignore_index=True)
    # traces: all events with date < exit (D7/D9)
    ev = L[["ucid", "date", "rowno"]].merge(segs[["segment_id", "ucid", "entry", "exit"]], on="ucid")
    ev = ev[ev.date < ev.exit]
    g = ev.groupby("segment_id")
    tr = pd.DataFrame({"n": g.size(), "last": g.date.max(),
                       "n_in": ev.assign(i=(ev.date >= ev.entry)).groupby("segment_id").i.sum()})
    segs = segs.merge(tr, left_on="segment_id", right_index=True, how="left").fillna({"n": 0, "n_in": 0})
    segs["n_hist"] = segs.n - segs.n_in
    segs["kept"] = segs.n_in > 0                                  # D7: drop empty
    RES["my_segments_built"] = segs.groupby("phase").size().to_dict()
    RES["my_D7_dropped"] = {int(k): {int(a): int(b) for a, b in v.items()} for k, v in
                            segs[~segs.kept].groupby("phase").lab.value_counts().unstack(fill_value=0).iterrows()}
    mine = segs[segs.kept].set_index("segment_id")
    log("my segments", mine.groupby("phase").size().to_dict(), "D7 dropped", RES["my_D7_dropped"])

    # ------------------------------------------------ output summaries
    outs = {k: out_summary(k) for k in (1, 2, 3)}
    osum = []
    for k, o in outs.items():
        g = o.groupby("segment_id", sort=False)
        s = g.agg(ucid=("ucid", "first"), label=("label", "first"), entry=("entry_date", "first"),
                  exit=("exit_date", "first"), n=("timestamp", "size"), last=("timestamp", "max"),
                  n_hist=("n_hist", "first"), n_in=("in_phase", "sum"),
                  nlab=("label", "nunique"), nent=("entry_date", "nunique"), nexit=("exit_date", "nunique"),
                  nnh=("n_hist", "nunique"))
        s["phase"] = k
        osum.append(s)
    osum = pd.concat(osum)
    assert (osum[["nlab", "nent", "nexit", "nnh"]] == 1).all().all(), "segment-static columns vary"
    osum["lab"] = osum.label.map({"deviant": 1, "regular": 0})
    assert osum.lab.notna().all()
    log("output segments", osum.groupby("phase").size().to_dict())

    # ------------------------------------------------ CHECK 1a: full population segment comparison
    j = mine.join(osum, how="outer", lsuffix="_m", rsuffix="_o", )
    only_m = j[j.ucid_o.isna()].index.tolist()
    only_o = j[j.ucid_m.isna()].index.tolist()
    b = j.dropna(subset=["ucid_m", "ucid_o"]).copy()
    b["entry_o"] = pd.to_datetime(b.entry_o)
    b["exit_o"] = pd.to_datetime(b.exit_o)
    b["last_o"] = pd.to_datetime(b.last_o)
    mm = {
        "label": b.lab_m != b.lab_o, "entry_date": b.entry_m != b.entry_o, "exit_date": b.exit_m != b.exit_o,
        "n_events": b.n_m != b.n_o, "last_event_date": b.last_m != b.last_o, "n_hist": b.n_hist_m != b.n_hist_o,
    }
    full = {k: int(v.sum()) for k, v in mm.items()}
    full["only_in_mine"] = len(only_m)
    full["only_in_output"] = len(only_o)
    anym = pd.concat(mm, axis=1).any(axis=1)
    bad = b[anym]
    full["segments_any_mismatch"] = int(anym.sum())
    full["mismatch_by_phase_and_flag"] = {
        f"p{p}": {"total": int((bad.phase_m == p).sum()), "D2": int(((bad.phase_m == p) & bad.D2).sum()),
                  "D1": int(((bad.phase_m == p) & bad.D1).sum()), "D4": int(((bad.phase_m == p) & bad.D4).sum())}
        for p in (1, 2, 3)}
    full["examples"] = [
        {"segment_id": i, "label_mine": int(r.lab_m), "label_out": int(r.lab_o),
         "exit_mine": str(r.exit_m.date()), "exit_out": str(r.exit_o.date()),
         "n_mine": int(r.n_m), "n_out": int(r.n_o), "last_mine": str(r.last_m.date()), "last_out": str(r.last_o.date()),
         "D1": bool(r.D1), "D2": bool(r.D2), "D4": bool(r.D4)} for i, r in bad.head(6).iterrows()]
    full["only_in_mine_examples"] = only_m[:5]
    full["only_in_output_examples"] = only_o[:5]
    RES["check1_full_population"] = full
    log("CHECK1 full population", {k: v for k, v in full.items() if "examples" not in k})

    # ------------------------------------------------ CHECK 1b: 200-segment stratified sample
    rng = np.random.default_rng(SEED)
    allseg = j.copy()
    allseg["phase"] = allseg.phase_m.fillna(allseg.phase_o).astype(int)
    allseg["labs"] = allseg.lab_m.fillna(allseg.lab_o).astype(int)
    allseg["flag"] = allseg[["D1", "D2", "D4"]].fillna(False).astype(bool).any(axis=1)
    sample = []
    quota = {(1, 0): 34, (1, 1): 34, (2, 0): 33, (2, 1): 33, (3, 0): 33, (3, 1): 33}
    for (p, lab), q in quota.items():
        pool = allseg[(allseg.phase == p) & (allseg.labs == lab)]
        fl = pool[pool.flag].index.to_numpy()
        nf = min(len(fl), q // 2)                                 # oversample D1/D2/D4 cases
        pick = list(rng.choice(fl, nf, replace=False)) if nf else []
        rest = pool.index.difference(pick).to_numpy()
        pick += list(rng.choice(rest, q - nf, replace=False))
        sample += pick
    assert len(sample) == 200 and len(set(sample)) == 200
    s200 = allseg.loc[sample]
    # re-derive traces event by event for the sample and compare sequences
    Lg = L.set_index("ucid")
    seqmm, rows = [], []
    for sid in sample:
        ucid, p = sid.rsplit("|p", 1)
        p = int(p)
        cr = c.loc[c.ucid == ucid].iloc[0]
        lab = int(cr[f"lab{p}"]); ent = cr[f"e{p}"]; ext = cr[f"x{p}"]
        evs = Lg.loc[[ucid]]
        evs = evs[evs.date < ext]
        keep = bool((evs.date >= ent).any()) and bool(cr.inwin) and not bool(cr.D3) and bool(cr[f"has{p}"])
        o = outs[p]
        oo = o[o.segment_id == sid]
        rec = {"segment_id": sid, "phase": p, "D1": bool(cr.D1), "D2": bool(cr.D2), "D4": bool(cr.D4),
               "kept_mine": keep, "in_output": len(oo) > 0}
        probs = []
        if keep != (len(oo) > 0):
            probs.append("presence")
        if len(oo):
            if {"deviant": 1, "regular": 0}[oo.label.iloc[0]] != lab: probs.append("label")
            if pd.Timestamp(oo.entry_date.iloc[0]) != ent: probs.append("entry_date")
            if pd.Timestamp(oo.exit_date.iloc[0]) != ext: probs.append("exit_date")
            if len(oo) != len(evs): probs.append("n_events")
            if len(evs) and pd.Timestamp(oo.timestamp.max()) != evs.date.max(): probs.append("last_event_date")
            if len(oo) == len(evs) and (list(oo.timestamp) != list(evs.date.dt.strftime("%Y-%m-%d"))
                                        or list(oo.Activity) != list(evs.Activity)):
                probs.append("event_sequence")
            rec.update(exit_out=oo.exit_date.iloc[0], n_out=int(len(oo)))
        rec.update(label_mine=lab, exit_mine=str(ext.date()), n_mine=int(len(evs)), problems=probs)
        rows.append(rec)
        if probs:
            seqmm.append(rec)
    RES["check1_sample"] = {
        "n": 200, "strata": {f"p{p}_lab{l}": q for (p, l), q in quota.items()},
        "n_flagged_D1_D2_D4": int(s200.flag.sum()),
        "flag_breakdown": {k: int(s200[k].fillna(False).astype(bool).sum()) for k in ["D1", "D2", "D4"]},
        "mismatching_segments": len(seqmm),
        "mismatch_fields": pd.Series([p for r in seqmm for p in r["problems"]]).value_counts().to_dict() if seqmm else {},
        "examples": seqmm[:8]}
    log("CHECK1 sample", {k: v for k, v in RES["check1_sample"].items() if k != "examples"})

    # ------------------------------------------------ CHECK 3: data-wide leakage/consistency
    c3 = {}
    for k, o in outs.items():
        ts = pd.to_datetime(o.timestamp); en = pd.to_datetime(o.entry_date); ex = pd.to_datetime(o.exit_date)
        r = {}
        r["rows"] = int(len(o))
        r["ts_ge_exit"] = int((ts >= ex).sum())
        r["remaining_ne_exit_minus_ts"] = int(((ex - ts).dt.days != o.remaining_in_phase).sum())
        r["in_phase_ne_ts_ge_entry"] = int(((ts >= en).astype(int) != o.in_phase).sum())
        nh = o.groupby("segment_id", sort=False).agg(nh=("n_hist", "first"), cnt=("in_phase", lambda x: int((x == 0).sum())))
        r["n_hist_ne_count_out_of_phase"] = int((nh.nh != nh.cnt).sum())
        # out-of-phase events must precede in-phase events
        r["hist_after_in_phase"] = int((o.in_phase.groupby(o.segment_id, sort=False).diff().fillna(0) < 0).sum())
        r["ts_not_sorted"] = int((ts.groupby(o.segment_id, sort=False).diff().dt.days.fillna(0) < 0).sum())
        r["event_nr_ne_pos+1"] = int((o.event_nr != o.pos + 1).sum())
        r["segments_with_no_in_phase"] = int((o.groupby("segment_id").in_phase.sum() == 0).sum())
        r["filing_year_range"] = [int(o.filing_year.min()), int(o.filing_year.max())]
        f = c.set_index("ucid").filing.reindex(o.ucid.unique())
        r["ucids_filed_outside_window"] = int((~f.between(pd.Timestamp(WIN[0]), pd.Timestamp(WIN[1]))).sum())
        fy = o.drop_duplicates("ucid").set_index("ucid").filing_year
        r["filing_year_ne_raw"] = int((fy != f.reindex(fy.index).dt.year).sum())
        r["D3_cases_present"] = int(c.set_index("ucid").D3.reindex(o.ucid.unique()).sum())
        r["segment_id_ne_ucid|pk"] = int((o.segment_id != o.ucid + f"|p{k}").sum())
        c3[f"p{k}"] = r
    # column-level leakage review
    feat_views = {v: cols["views"][v] for v in ["M", "P", "PP", "B0G", "G"]}
    feats = [x for v in feat_views.values() for x in v]
    outcome_pat = re.compile(r"(resolution_type|disposition_group|is_case_|activity_count_|total_|terminated|"
                             r"pending|unique_.*judge_count|phase\d_(exit|los|activities|completed|entered|entry)|"
                             r"case_status|terminating|counsels_count|_count$|pro_hac_vice|label|remaining|exit|entry|"
                             r"n_hist|in_phase|ucid|filing)", re.I)
    c3["feature_cols_matching_outcome_pattern"] = [x for x in feats if outcome_pat.search(x) and x != "M_related_case_count"]
    c3["never_features_in_views"] = [x for x in cols["never_features"] if x in feats]
    c3["S_cols_in_main_views"] = [x for x in cols["views"]["S"] if x in feats]
    d8_susp = ["plaintiffs_count", "plaintiffs_counsels_count", "Defendants_count", "Defendants_counsels_count",
               "plaintiffs_share_pro_hac_vice", "Defendants_share_pro_hac_vice"]
    c3["D8_suspects_outside_S"] = [x for x in feats if any(x.endswith(s) for s in d8_susp)]
    c3["S_view"] = cols["views"]["S"]
    allcols = pq.ParquetFile(os.path.join(PROC, "court_phase1.parquet")).schema_arrow.names
    fixed = ["segment_id", "ucid", "Activity", "judge", "timestamp", "label", "remaining_in_phase", "in_phase", "n_hist",
             "entry_date", "exit_date", "filing_year"]
    c3["file_cols_not_accounted"] = [x for x in allcols if x not in feats + cols["views"]["S"] + fixed + cols["never_features"]]
    c3["typed_lists_vs_views"] = sorted(set(cols["static_cat"] + cols["static_num"] + cols["dynamic_cat"] + cols["dynamic_num"])
                                        ^ set(feats + cols["views"]["S"]))
    RES["check3"] = c3
    log("CHECK3", json.dumps(c3)[:1500])

    # ------------------------------------------------ CHECK 2: 500 feature cells by brute force
    feature_check(c, L, outs, cols, rng)

    # ------------------------------------------------ extra full-population checks
    extra_full(c, outs)
    if not os.environ.get("G2_TARGETED"):
        extra_reaudit(c, L, outs, cols)
    tag = "_targeted" if os.environ.get("G2_TARGETED") else ""
    with open(os.path.join(ROOT, "verification", f"g2_results{tag}.json"), "w") as fh:
        json.dump(RES, fh, indent=1, default=str)
    log("done")


def extra_full(c, outs):
    """Full-population checks of open_cases (B0G) and PP_prev_phase_days_* for every output row."""
    C = c[~c.D3]
    fs = np.sort(C.filing.values); ts_ = np.sort(C.term.values)
    ci = c.set_index("ucid")
    r = {}
    for k in (1, 2, 3):
        o = pq.read_table(os.path.join(PROC, f"court_phase{k}.parquet"),
                          columns=["ucid", "timestamp", "open_cases", "PP_prev_phase_days_1", "PP_prev_phase_days_2",
                                   "remaining_in_phase", "timesincecasestart", "exit_date"]).to_pandas()
        t = pd.to_datetime(o.timestamp).values
        own = ci.reindex(o.ucid)
        selfopen = ((own.filing.values <= t) & (own.term.values > t)).astype(int)
        exp = np.searchsorted(fs, t, "right") - np.searchsorted(ts_, t, "right") - selfopen
        rr = {"rows": int(len(o)), "open_cases_mismatch": int((exp != o.open_cases.values).sum())}
        if k > 1:
            e1 = (own.x1 - own.e1).dt.days.values
            rr["prev_days_1_mismatch"] = int((e1 != o.PP_prev_phase_days_1.values).sum())
            m = own.D2.values
            rr["prev_days_1_mismatch_D2_rows"] = int(((e1 != o.PP_prev_phase_days_1.values) & m).sum())
            rr["prev_days_1_eq_raw_D2"] = int(((own.x1_raw - own.e1).dt.days.values == o.PP_prev_phase_days_1.values)[m].sum())
            # leak test: for D2 rows, prev_days_1 - timesincecasestart - remaining_in_phase == 0 ?
            first_gap = (pd.to_datetime(o.timestamp) - own.filing.values).dt.days.values - o.timesincecasestart.values
            rr["D2_rows"] = int(m.sum())
            rr["D2_rows_remaining_recoverable_from_features"] = int(
                (o.PP_prev_phase_days_1.values - o.timesincecasestart.values - first_gap == o.remaining_in_phase.values)[m].sum()) if k == 3 else None
            if k == 3:
                e2 = np.where(m, 0, (own.x2 - own.e2).dt.days.values)
                rr["prev_days_2_mismatch"] = int((e2 != o.PP_prev_phase_days_2.values).sum())
            else:
                rr["prev_days_2_nonzero"] = int((o.PP_prev_phase_days_2.values != 0).sum())
        else:
            rr["prev_days_nonzero"] = int(((o.PP_prev_phase_days_1 != 0) | (o.PP_prev_phase_days_2 != 0)).sum())
        r[f"p{k}"] = rr
    RES["extra_full_population"] = r
    log("EXTRA", r)


def extra_reaudit(c, L, outs, cols):
    """Re-audit extras: CSV/parquet twins (p1, p2), rare-level 'other' mapping, M values vs case table."""
    import hashlib
    r = {}
    rng = np.random.default_rng(FSEED + 2)
    # --- twins: row count + 1% sample (every 100th row) compared column by column, plus a hash
    tw = {}
    for k in (1, 2):
        pt = pq.read_table(os.path.join(PROC, f"court_phase{k}.parquet"))
        n_pq = pt.num_rows
        keep, parts, off = [], [], 0
        for ch in pd.read_csv(os.path.join(PROC, f"court_phase{k}.csv"), sep=";", chunksize=250_000, low_memory=False):
            idx = np.arange(off, off + len(ch))
            parts.append(ch[idx % 100 == 0])
            off += len(ch)
        a = pd.concat(parts)
        b = pt.take(np.arange(0, n_pq, 100)).to_pandas()
        diff = []
        for col in b.columns:
            x, y = a[col].reset_index(drop=True), b[col].reset_index(drop=True)
            if (x.astype(str).values == y.astype(str).values).all():
                continue
            xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
            if xn.notna().all() and ((xn - yn).abs() <= 1e-9).all():
                continue
            diff.append(col)
        hs = lambda d: hashlib.md5(pd.util.hash_pandas_object(d.astype(str), index=False).values.tobytes()).hexdigest()
        tw[f"p{k}"] = {"rows_csv": off, "rows_parquet": n_pq, "same_columns": list(a.columns) == list(b.columns),
                       "sample_rows": int(len(b)), "differing_columns": diff,
                       "hash_csv_sample": hs(a.reset_index(drop=True)), "hash_parquet_sample": hs(b)}
    r["twins"] = tw
    log("TWINS", tw)
    # --- rare-level mapping + M values, full population per phase file
    ci = c.set_index("ucid")
    mcat = [x for x in cols["views"]["M"] if x in cols["static_cat"]]
    mnum = [x for x in cols["views"]["M"] + cols["views"]["S"] if x in cols["static_num"]]
    rare, mvals, msample = {}, {}, []
    for k, o in outs.items():
        # judge: rebuild every row of the file from the raw log, aligned on (segment, position)
        seg = o.drop_duplicates("segment_id")[["segment_id", "ucid", "exit_date"]]
        ev = L[["ucid", "date", "rowno", "event_judge", "Activity"]].merge(seg, on="ucid")
        ev = ev[ev.date < pd.to_datetime(ev.exit_date)].sort_values(["segment_id", "date", "rowno"])
        ev["pos"] = ev.groupby("segment_id").cumcount()
        oj = pd.read_parquet(os.path.join(PROC, f"court_phase{k}.parquet"), columns=["segment_id", "judge"])
        oj["pos"] = o.pos.values
        m = oj.merge(ev, on=["segment_id", "pos"], how="left")
        raw = m.event_judge.str.extract(r"\('([^']*)'", expand=False).fillna("missing")
        cnt = raw.value_counts()
        exp = raw.where(raw.map(cnt) >= 10, "other")
        rr = {"rows": int(len(m)), "unaligned_rows": int(m.event_judge.isna().sum()),
              "activity_mismatch": int((m.Activity_x != m.Activity_y).sum()) if "Activity_x" in m else None,
              "judge_mismatch": int((exp.values != m.judge.values).sum()),
              "judge_levels_mapped_to_other": int((cnt < 10).sum()), "judge_rows_other": int((exp == "other").sum())}
        # static M categoricals: counted per segment
        st = pd.read_parquet(os.path.join(PROC, f"court_phase{k}.parquet"), columns=["segment_id", "ucid"] + mcat + mnum)
        st = st.drop_duplicates("segment_id")
        nvary = pd.read_parquet(os.path.join(PROC, f"court_phase{k}.parquet"), columns=["segment_id"] + mcat + mnum) \
            .groupby("segment_id").nunique().gt(1).sum().sum()
        rr["M_values_varying_within_segment"] = int(nvary)
        cs = ci.reindex(st.ucid)
        for col in mcat:
            rv = cs[col[2:]].fillna("missing").astype(str).values
            vc = pd.Series(rv).value_counts()
            e = np.where(pd.Series(rv).map(vc).values >= 10, rv, "other")
            rr[f"{col}_mismatch"] = int((e != st[col].astype(str).values).sum())
            rr[f"{col}_levels_other"] = int((vc < 10).sum())
            mvals.setdefault(col, 0)
            mvals[col] += int((e != st[col].astype(str).values).sum())
        for col in mnum:
            src = col[2:]
            rv = cs[src].map({True: 1.0, False: 0.0, "TRUE": 1.0, "FALSE": 0.0}).fillna(cs[src]) if cs[src].dtype == bool or cs[src].dtype == object else cs[src]
            rv = pd.to_numeric(rv, errors="coerce").fillna(0).values
            mm = int((np.abs(rv - st[col].values.astype(float)) > 1e-9).sum())
            mvals.setdefault(col, 0)
            mvals[col] += mm
        # 200 random M cells (about 67 per phase), checked one by one against the raw case row
        for _ in range({1: 67, 2: 67, 3: 66}[k]):
            i = int(rng.integers(len(st))); col = str(rng.choice(mcat + [x for x in mnum if x.startswith("M_")]))
            raw_v = ci.loc[st.ucid.iloc[i], col[2:]]
            got = st[col].iloc[i]
            if col in mcat:
                exp_v = "missing" if pd.isna(raw_v) else str(raw_v)
                ok = got == exp_v or (got == "other" and rr[f"{col}_levels_other"] > 0)
            else:
                exp_v = 0.0 if pd.isna(raw_v) else float(raw_v)
                ok = abs(exp_v - float(got)) <= 1e-9
            msample.append({"phase": k, "segment_id": st.segment_id.iloc[i], "col": col, "raw": str(raw_v),
                            "got": str(got), "ok": bool(ok)})
        rare[f"p{k}"] = rr
        log("RARE/M", k, rr)
    r["rare_level_and_alignment"] = rare
    r["M_S_full_population_mismatches"] = mvals
    ms = pd.DataFrame(msample)
    r["M_sample_200"] = {"n": int(len(ms)), "mismatches": int((~ms.ok).sum()),
                         "other_cells": int((ms.got == "other").sum()), "examples_bad": ms[~ms.ok].head(5).to_dict("records")}
    RES["reaudit_extras"] = r
    log("M sample", r["M_sample_200"])


# ====================================================================== features
def build_trace(c_row, L_by, k):
    ext = c_row[f"x{k}"]
    ev = L_by.loc[[c_row.ucid]]
    return ev[ev.date < ext].reset_index()


def p_value(col, tr, i, c_row, k, all_judges=True):
    """Brute-force P/PP feature at trace position i."""
    row = tr.iloc[i]
    t = row.date
    acts = tr.Activity.tolist()
    if col == "Activity":
        return row.Activity
    if col == "timesincelastevent":
        return 0 if i == 0 else (t - tr.date.iloc[i - 1]).days
    if col == "timesincecasestart":
        return (t - tr.date.iloc[0]).days
    if col == "event_nr":
        return i + 1
    if col == "month":
        return t.month
    if col == "weekday":
        return t.weekday()
    if col.startswith("P_attr_"):
        return int(row["attribute_" + col[len("P_attr_"):]])
    if col == "P_judge_is_magistrate":
        js = parse_judges(row.event_judge)
        return int(bool(js) and js[0][1] == "Magistrate Judge")
    if col == "judge":
        js = parse_judges(row.event_judge)
        return js[0][0] if js else "missing"
    if col == "PP_n_self_loops":
        return sum(acts[j] == acts[j - 1] for j in range(1, i + 1))
    if col == "PP_n_return_loops":
        return sum(acts[j] == acts[j - 2] and acts[j] != acts[j - 1] for j in range(2, i + 1))
    if col == "PP_n_repeats":
        return (i + 1) - len(set(acts[: i + 1]))
    if col == "PP_n_distinct_judges":
        ids = set()
        for s in tr.event_judge.iloc[: i + 1]:
            js = parse_judges(s)
            if js:
                ids.update(x[0] for x in js) if all_judges else ids.add(js[0][0])
        return len(ids)
    if col in ("PP_events_last30", "PP_events_last90"):
        w = 30 if col.endswith("30") else 90
        d = tr.date.iloc[: i + 1]
        return int((d > t - pd.Timedelta(days=w)).sum())
    if col == "PP_prev_phase_days_1":
        return 0 if k == 1 else (c_row.x1 - c_row.e1).days
    if col == "PP_prev_phase_days_2":
        return 0 if k < 3 or c_row.D2 else (c_row.x2 - c_row.e2).days
    raise KeyError(col)


def g_value(col, t, c_row, k, C, x1col="x1", filed_self=False):
    """Brute-force global feature at date t for case c_row (phase k), scanning all cases (after D3)."""
    me = C.ucid.values == c_row.ucid
    open_ = (C.filing.values <= t) & (C.term.values > t) & ~me
    def inph(j):
        e = C[f"e{j}"].values; x = C[x1col].values if j == 1 else C[f"x{j}"].values
        return (e <= t) & (x > t) & ~me
    w30 = lambda s, w: (s > t - np.timedelta64(w, "D")) & (s <= t)
    if col == "open_cases":
        return int(open_.sum())
    if col == "G_court_open_same_type":
        return int((open_ & (C.case_type.values == c_row.case_type)).sum())
    m = re.fullmatch(r"G_court_open_phase(\d)", col)
    if m:
        return int(inph(int(m.group(1))).sum())
    m = re.fullmatch(r"G_court_(filed|closed|net_flow)(?:_last)?(\d+)", col.replace("_last", "_last"))
    m = re.fullmatch(r"G_(court|judge)_(filed|closed|net_flow)(?:_last)?(30|90)", col)
    jd = c_row.District_Judge
    hasj = isinstance(jd, str)
    jm = (C.District_Judge.values == jd) if hasj else np.zeros(len(C), bool)
    if m:
        scope, what, w = m.group(1), m.group(2), int(m.group(3))
        base = np.ones(len(C), bool) if scope == "court" else jm
        if scope == "judge" and not hasj:
            return 0
        f = int((w30(C.filing.values, w) & base & (~me | filed_self)).sum())
        cl = int((w30(C.term.values, w) & base & ~me).sum())
        return {"filed": f, "closed": cl, "net_flow": f - cl}[what]
    if col.startswith("G_judge_") and not hasj:
        return 0
    if col == "G_judge_open":
        return int((open_ & jm).sum())
    if col == "G_judge_open_same_type":
        return int((open_ & jm & (C.case_type.values == c_row.case_type)).sum())
    m = re.fullmatch(r"G_judge_open_phase(\d)", col)
    if m:
        return int((inph(int(m.group(1))) & jm).sum())
    if col == "G_judge_open_same_phase":
        return int((inph(k) & jm).sum())
    if col == "G_judge_share_of_court":
        jo = int((open_ & jm).sum()); co = int(open_.sum())
        return jo / co if co else 0.0
    if col == "G_mag_open":
        mj = c_row.Magistrate_Judge
        if not isinstance(mj, str):
            return 0
        return int((open_ & (C.Magistrate_Judge.values == mj)).sum())
    if col in ("G_judge_hist_median_los", "G_judge_hist_n_closed"):
        cl = jm & (C.term.values < t)
        if col == "G_judge_hist_n_closed":
            return int(cl.sum())
        los = (C.term.values[cl] - C.filing.values[cl]) / np.timedelta64(1, "D")
        return float(np.median(los)) if len(los) else 0.0
    t = pd.Timestamp(t)
    mo = t.month
    if col == "G_month_sin":
        return float(np.sin(2 * np.pi * mo / 12))
    if col == "G_month_cos":
        return float(np.cos(2 * np.pi * mo / 12))
    if col == "G_quarter":
        return (mo - 1) // 3 + 1
    if col == "G_days_to_year_end":
        return (pd.Timestamp(year=t.year, month=12, day=31) - t).days
    if col == "G_is_december":
        return int(mo == 12)
    raise KeyError(col)


def feature_check(c, L, outs, cols, rng):
    # global state from all cases after D3 (RD-07); fixed phase intervals, D1/D2/D4 have no phase-2 interval
    C = c[~c.D3].reset_index(drop=True)
    cidx = c.set_index("ucid")
    L_by = L.set_index("ucid")
    gcols = cols["views"]["G"] + cols["views"]["B0G"]
    pcols = cols["views"]["P"] + cols["views"]["PP"]
    plan = [("G", 180), ("P", 320)]
    if os.environ.get("G2_FSEED"):
        rng = np.random.default_rng(FSEED)
    if os.environ.get("G2_TARGETED"):                            # supplementary targeted sample
        rng = np.random.default_rng(FSEED + 1)
        gcols = [x for x in gcols if re.search(r"filed|net_flow|phase1|same_phase", x)]
        pcols = ["PP_n_distinct_judges", "PP_prev_phase_days_1", "PP_prev_phase_days_2"]
        plan = [("G", 200), ("P", 100)]
    cells = []
    sizes = {k: len(o) for k, o in outs.items()}
    for grp, n in plan:
        pool = gcols if grp == "G" else pcols
        for _ in range(n):
            k = int(rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
            ridx = int(rng.integers(sizes[k]))
            col = str(rng.choice(pool))
            cells.append((k, ridx, col, grp))
    res = []
    tables = {}
    for k in (1, 2, 3):
        idx = sorted({r for kk, r, _, _ in cells if kk == k})
        t = pq.read_table(os.path.join(PROC, f"court_phase{k}.parquet")).take(idx).to_pandas()
        t.index = idx
        tables[k] = t
    judge_counts = {}
    for (k, ridx, col, grp) in cells:
        orow = tables[k].loc[ridx]
        pos = int(outs[k].pos.iloc[ridx])
        cr = cidx.loc[orow.ucid].copy()
        cr["ucid"] = orow.ucid
        tr = build_trace(cr, L_by, k)
        t = tr.date.iloc[pos]
        ts_ok = str(t.date()) == orow.timestamp
        if grp == "G":
            exp = g_value(col, np.datetime64(t), cr, k, C)
        else:
            exp = p_value(col, tr, pos, cr, k)
        got = orow[col]
        if isinstance(exp, str):
            ok = exp == got
        else:
            ok = abs(float(exp) - float(got)) <= 1e-6
        rec = {"phase": k, "row": ridx, "segment_id": orow.segment_id, "pos": pos, "timestamp": orow.timestamp,
               "col": col, "group": grp, "expected": exp, "got": got.item() if hasattr(got, "item") else got,
               "ok": bool(ok and ts_ok), "ts_ok": ts_ok}
        rec["explained_by"] = ""
        if not rec["ok"]:
            alt = []
            if grp == "G":
                alt.append(("D2_raw_phase1_exit", g_value(col, np.datetime64(t), cr, k, C, x1col="x1_raw")))
                alt.append(("filed_count_includes_self", g_value(col, np.datetime64(t), cr, k, C, filed_self=True)))
            if col == "PP_prev_phase_days_1" and cr.D2:
                alt.append(("D2_raw_phase1_exit", (cr.x1_raw - cr.e1).days))
            if col == "PP_n_distinct_judges":
                alt.append(("distinct_judges_counts_first_judge_only", p_value(col, tr, pos, cr, k, all_judges=False)))
            if pd.Timestamp(cr.x1_raw) != pd.Timestamp(cr.x1) and k == 1:
                alt.append(("D2_raw_phase1_exit_trace", "trace differs"))
            for name, v in alt:
                if not isinstance(v, str) and abs(float(v) - float(rec["got"])) <= 1e-6:
                    rec["explained_by"] = name
                    break
        res.append(rec)
    df = pd.DataFrame(res)
    bad = df[~df.ok]
    # rare-level judge: resolve "other" cells by counting occurrences of the raw id among the file's rows
    # (rows of the phase file = re-derived traces, so count the raw first-judge id over those rows)
    other = bad[(bad.col == "judge") & (bad.got == "other")]
    for i, r in other.iterrows():
        o = outs[r.phase][["ucid", "exit_date"]].drop_duplicates()
        ev = L[L.ucid.isin(o.ucid)].merge(o, on="ucid")
        ev = ev[ev.date < pd.to_datetime(ev.exit_date)]
        n = int((ev.event_judge.str.extract(r"\('([^']*)'", expand=False) == r.expected).sum())
        if n < 10:
            df.loc[i, "ok"] = True
            df.loc[i, "note"] = f"rare level ({n} rows) -> other"
    bad = df[~df.ok]
    RES["check2"] = {
        "n_cells": int(len(df)), "n_G_or_open_cases": int((df.group == "G").sum()), "n_P_PP": int((df.group == "P").sum()),
        "distinct_columns_covered": int(df.col.nunique()),
        "by_phase": df.phase.value_counts().sort_index().to_dict(),
        "mismatches": int(len(bad)), "mismatch_by_col": bad.col.value_counts().to_dict(),
        "mismatch_by_explanation": bad.explained_by.replace("", "UNEXPLAINED").value_counts().to_dict(),
        "timestamp_misaligned": int((~df.ts_ok).sum()),
        "judge_other_cells_checked": int(len(other)),
        "examples": bad.head(15).to_dict("records")}
    tag = "_targeted" if os.environ.get("G2_TARGETED") else ""
    if tag:
        RES["check2_targeted"] = RES.pop("check2")
    df.to_csv(os.path.join(ROOT, "verification", f"g2_feature_cells{tag}.csv"), index=False)
    log("CHECK2" + tag, {k: v for k, v in RES["check2" + tag].items() if k != "examples"})


if __name__ == "__main__":
    main()
