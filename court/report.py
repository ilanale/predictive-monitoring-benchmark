"""Build ``results.md``, figures and tables from the raw result files (DoD-7, PLAN.md §7).

    python -m court.report --results-dir results [--out results.md] [--config configs/experiments.yaml]
        [--data-dir data/processed] [--no-bootstrap]

No number in ``results.md`` is typed by hand: every value comes from ``<results_dir>/runs.csv``,
``predictions/*.csv.gz``, ``params/*.json``, ``importance/*.csv``, ``tables/*__by_prefix.csv``,
``queue.log``, ``data/processed/data_report.json`` / ``docs/data_report.md``, ``RUN_DECISIONS.md``
and the experiment config. Missing or failed jobs are reported as "not run / failed: reason".

Derived artefacts (written under ``<results_dir>``):
* ``tables/summary_classification.csv``, ``tables/summary_regression.csv``, ``tables/job_status.csv``
* ``tables/bootstrap_delta_vs_FS-B0.csv`` (C11; cached, keyed by the prediction files' size/mtime)
* ``tables/importance_by_view.csv`` (C12)
* ``tables/baseline_by_prefix__<task>.csv`` (C13 per prefix length; needs ``data/processed``)
* ``tables/segment_filing_year__p<k>.csv.gz`` and ``tables/drift_by_year.csv`` (Tier 3)
* ``tables/runtime.csv``
* ``figures/*.png``
"""
import argparse
import glob
import json
import os
import re
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd

from court import evaluate, runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS_ORDER = ["FS-M", "FS-MP", "FS-B0", "FS-B0+P", "FS-B0+G", "FS-ALL"]
BASE_FS = "FS-B0"
VIEWS = ["M", "P", "P+", "G", "G+", "S"]
TASK_NAMES = {"court_p1_clf": "T1", "court_p2_clf": "T2", "court_p3_clf": "T3",
              "court_p1_reg": "R1", "court_p2_reg": "R2", "court_p3_reg": "R3"}
PHASE2_REASON = "method needs the upstream bucketers (not wired into the court runner)"


# ----------------------------------------------------------------------------- helpers
def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or x == "":
        return "–"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "–"
    if abs(x) >= 1000:
        return "{:,.0f}".format(x)
    return ("{:.%df}" % nd).format(x)


def fint(x):
    try:
        return "{:,}".format(int(float(x)))
    except (TypeError, ValueError):
        return "–"


def md_table(df, cols=None, headers=None):
    cols = cols or list(df.columns)
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def task_type_of(task):
    return "classification" if task.endswith("_clf") else "regression"


def phase_of(task):
    return int(task.split("_")[1][1:])


def rel(path, base):
    return os.path.relpath(path, base)


# ----------------------------------------------------------------------------- job state
def load_runs(results_dir):
    """runs.csv with one row per job_id: the last ``done`` row if any, else the last row."""
    runs = runner.read_runs(results_dir)
    if not len(runs):
        return runs
    runs = runs.reset_index(drop=True)
    runs["_order"] = np.arange(len(runs))
    runs["_done"] = (runs["status"] == "done").astype(int)
    runs = runs.sort_values(["job_id", "_done", "_order"]).groupby("job_id", sort=False).tail(1)
    num = ["runtime_s", "n_train_prefixes", "n_test_prefixes", "n_train_segments", "n_test_segments",
           "max_prefix_len", "auc", "f1", "pr_auc", "mae", "median_ae", "r2_log", "baseline_mae_overall",
           "baseline_mae_per_prefix"]
    for c in num:
        if c in runs:
            runs[c] = pd.to_numeric(runs[c], errors="coerce")
    return runs.drop(columns=["_order", "_done"]).sort_index()


def queue_events(results_dir):
    path = os.path.join(results_dir, "queue.log")
    started, running = set(), None
    if not os.path.exists(path):
        return started, running
    with open(path) as f:
        for line in f:
            m = re.match(r"\S+ \S+ (START|DONE|FAILED|TIMEOUT) (\S+)", line)
            if not m:
                continue
            started.add(m.group(2))
            if m.group(1) == "START":
                running = m.group(2)
            elif running == m.group(2) and m.group(1) in ("DONE", "FAILED"):
                running = None
    return started, running


def job_status_table(cfg, runs, results_dir):
    from court.run_queue import expand_jobs
    jobs = expand_jobs(cfg)
    started, running = queue_events(results_dir)
    rows = []
    byid = runs.set_index("job_id") if len(runs) else pd.DataFrame()
    for j in jobs:
        jid = j["job_id"]
        parts = jid.split("__")
        if j["kind"] == "tune":
            task, method, fs, learner, tier = parts[1], parts[2], cfg.get("tuned_fs", "FS-ALL"), parts[3], "tune"
        else:
            tier, task, method, fs, learner = parts
        r = byid.loc[jid] if jid in byid.index else None
        if r is not None:
            status = r["status"]
            reason = r.get("flags", "") or ""
            if status != "done":
                status = "%s" % status
        elif method in runner.PHASE2_METHODS:
            status, reason = "not run", PHASE2_REASON
        elif jid == running:
            status, reason = "running", "in progress at report time"
        else:
            status, reason = "not run", "not reached by the queue at report time"
        rows.append(dict(job_id=jid, kind=j["kind"], tier=j["tier"] if j["kind"] == "fit" else tier,
                         queue_tier=j["tier"], task=task, method=method, fs=fs, learner=learner,
                         status=status, reason=reason))
    known = {r["job_id"] for r in rows}
    for _, r in runs.iterrows():  # jobs run outside the configured matrix
        if r["job_id"] not in known:
            rows.append(dict(job_id=r["job_id"], kind="tune" if r["job_id"].startswith("tune__") else "fit",
                             tier=r.get("tier", ""), queue_tier=r.get("tier", ""), task=r.get("task", ""),
                             method=r.get("method", ""), fs=r.get("fs", ""), learner=r.get("learner", ""),
                             status=r["status"], reason=r.get("flags", "")))
    return pd.DataFrame(rows)


def tuning_note(runs, task, method, learner="xgboost"):
    tid = runner.tune_id(task, method, learner)
    r = runs[runs["job_id"] == tid]
    if not len(r):
        return "tuning not run"
    r = r.iloc[0]
    if r["status"] != "done":
        return "tuning %s (%s) → default params" % (r["status"], r["flags"] or "no detail")
    return "tuned" + (" (%s)" % r["flags"] if r["flags"] else "")


# ----------------------------------------------------------------------------- predictions
_PRED_CACHE = {}


def pred_path(results_dir, job_id):
    return os.path.join(results_dir, "predictions", job_id + ".csv.gz")


def load_pred(results_dir, job_id):
    p = pred_path(results_dir, job_id)
    if p not in _PRED_CACHE:
        _PRED_CACHE[p] = pd.read_csv(p) if os.path.exists(p) else None
    return _PRED_CACHE[p]


def file_sig(path):
    st = os.stat(path)
    return "%d:%d" % (st.st_size, int(st.st_mtime))


# ----------------------------------------------------------------------------- C11 fast bootstrap
def _weighted_auc_rows(y, p, W):
    """AUC for each row of the weight matrix ``W`` (B × n): Mann-Whitney with ties = 1/2."""
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(ps) != 0) + 1])
    Ws = W[:, order]
    pos = y[order] == 1
    Pg = np.add.reduceat(np.where(pos, Ws, 0.0), starts, axis=1)
    Ng = np.add.reduceat(np.where(pos, 0.0, Ws), starts, axis=1)
    cumN = np.cumsum(Ng, axis=1) - Ng
    P, N = Pg.sum(1), Ng.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (Pg * (cumN + 0.5 * Ng)).sum(1) / (P * N)
    auc[(P == 0) | (N == 0)] = np.nan
    return auc


def fast_paired_bootstrap(pred_x, pred_base, task_type, n_boot=evaluate.N_BOOT, seed=evaluate.BOOT_SEED,
                          max_cells=2e7):
    """Vectorised equivalent of ``evaluate.paired_bootstrap`` (same seed, same resamples).

    A resample of segments with replacement is a vector of multiplicities ``c``; duplicating a
    segment's rows c times is the same as giving them weight c, so AUC / MAE are computed as
    weighted statistics for many resamples at once."""
    key = ["segment_id", "prefix_len"]
    m = pred_x[key + ["y_true", "y_pred"]].merge(pred_base[key + ["y_pred"]], on=key, suffixes=("_x", "_b"))
    if len(m) != len(pred_x) or len(m) != len(pred_base):
        raise ValueError("prediction frames do not cover the same test prefixes")
    metric = evaluate.primary_metric(task_type)
    f = evaluate.clf_metrics if task_type == "classification" else evaluate.reg_metrics
    point = f(m["y_true"], m["y_pred_x"])[metric] - f(m["y_true"], m["y_pred_b"])[metric]
    segs, inv = np.unique(m["segment_id"].to_numpy(), return_inverse=True)
    S = len(segs)
    rng = np.random.default_rng(seed)
    C = np.empty((n_boot, S), dtype=np.float64)
    for b in range(n_boot):  # same draw sequence as evaluate.paired_bootstrap
        C[b] = np.bincount(rng.integers(0, S, S), minlength=S)
    y = m["y_true"].to_numpy()
    px = m["y_pred_x"].to_numpy(float); pb = m["y_pred_b"].to_numpy(float)
    if metric == "auc":
        y = y.astype(int)
        deltas = np.empty(n_boot)
        step = max(1, int(max_cells // max(1, len(m))))
        for lo in range(0, n_boot, step):
            W = C[lo:lo + step][:, inv]
            deltas[lo:lo + step] = _weighted_auc_rows(y, px, W) - _weighted_auc_rows(y, pb, W)
    else:
        n_s = np.bincount(inv, minlength=S).astype(float)
        ex = np.bincount(inv, weights=np.abs(y - px), minlength=S)
        eb = np.bincount(inv, weights=np.abs(y - pb), minlength=S)
        deltas = (C @ ex - C @ eb) / (C @ n_s)
    lo_, hi_ = np.nanpercentile(deltas, [2.5, 97.5])
    return {"metric": metric, "delta": point, "ci_low": lo_, "ci_high": hi_, "n_boot": n_boot, "n_segments": S}


def bootstrap_table(runs, results_dir, n_boot=evaluate.N_BOOT, enabled=True):
    path = os.path.join(results_dir, "tables", "bootstrap_delta_vs_%s.csv" % BASE_FS)
    cache = pd.read_csv(path, dtype={"sig": str}) if os.path.exists(path) else pd.DataFrame()
    done = runs[(runs["status"] == "done") & ~runs["job_id"].str.startswith("tune__")]
    rows = []
    for (tier, task, method, learner), g in done.groupby(["tier", "task", "method", "learner"], sort=False):
        base_id = runner.make_job_id(tier, task, method, BASE_FS, learner)
        if base_id not in set(g["job_id"]):
            continue
        for fs in [f for f in FS_ORDER if f != BASE_FS] + sorted(set(g["fs"]) - set(FS_ORDER)):
            jid = runner.make_job_id(tier, task, method, fs, learner)
            if jid not in set(g["job_id"]):
                continue
            px, pb = pred_path(results_dir, jid), pred_path(results_dir, base_id)
            if not (os.path.exists(px) and os.path.exists(pb)):
                continue
            sig = "%s|%s|%d|%d" % (file_sig(px), file_sig(pb), n_boot, evaluate.BOOT_SEED)
            hit = cache[(cache["job_id"] == jid) & (cache["sig"] == sig)] if len(cache) else cache
            if len(hit):
                rows.append(hit.iloc[0].to_dict()); continue
            if not enabled:
                continue
            t0 = time.time()
            r = fast_paired_bootstrap(load_pred(results_dir, jid), load_pred(results_dir, base_id),
                                      task_type_of(task), n_boot=n_boot)
            r.update(job_id=jid, base_job_id=base_id, tier=tier, task=task, method=method, fs=fs,
                     learner=learner, sig=sig, seconds=round(time.time() - t0, 2))
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out.to_csv(path, index=False)
    return out


# ----------------------------------------------------------------------------- data-dependent parts
def _phase_frame(data_dir, phase, cols):
    pq = os.path.join(data_dir, "court_phase%d.parquet" % phase)
    csv = os.path.join(data_dir, "court_phase%d.csv" % phase)
    if os.path.exists(pq):
        return pd.read_parquet(pq, columns=cols)
    if os.path.exists(csv):
        return pd.read_csv(csv, sep=";", usecols=cols)
    return None


def segment_years(results_dir, data_dir, phase):
    """segment_id → filing_year (cached in tables/ so the drift check also works without data/)."""
    cache = os.path.join(results_dir, "tables", "segment_filing_year__p%d.csv.gz" % phase)
    if os.path.exists(cache):
        s = pd.read_csv(cache)
        return s.set_index("segment_id")["filing_year"]
    df = _phase_frame(data_dir, phase, ["segment_id", "filing_year"]) if data_dir else None
    if df is None:
        return None
    s = df.drop_duplicates("segment_id").set_index("segment_id")["filing_year"]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    s.reset_index().to_csv(cache, index=False)
    return s


def baseline_by_prefix(results_dir, data_dir, task, max_len, gap_train, ref_overall=None):
    """C13 baselines per in-phase prefix length, rebuilt from data/processed with the runner's
    split and prefix rules (``data_io.split_data_strict``, ``fast_pipeline.prefix_ends``).
    Returns (frame, overall check dict) or (None, reason)."""
    cache = os.path.join(results_dir, "tables", "baseline_by_prefix__%s.csv" % task)
    chk_path = cache.replace(".csv", "__check.json")
    if os.path.exists(cache) and os.path.exists(chk_path):
        with open(chk_path) as f:
            chk = json.load(f)
        if chk.get("max_len") == max_len and chk.get("gap_train") == gap_train:
            return pd.read_csv(cache), chk
    if not data_dir:
        return None, "no data dir"
    from court import data_io
    from court.fast_pipeline import prefix_ends, prepare_log
    cols = ["segment_id", "Activity", "timestamp", "in_phase", "n_hist", "remaining_in_phase"]
    df = _phase_frame(data_dir, phase_of(task), cols)
    if df is None:
        return None, "data/processed not available"
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    train, test, _ = data_io.split_data_strict(df)
    metas = []
    for part, gap in ((train, gap_train), (test, 1)):
        p = prepare_log(part)
        rows, j, _ = prefix_ends(p, max_len, gap)
        ff = p.groupby("segment_id", sort=False)["remaining_in_phase"].ffill().to_numpy()
        metas.append(pd.DataFrame({"prefix_nr": j, "y_reg": ff[rows].astype(float)}))
    bl = evaluate.regression_baselines(metas[0], metas[1])
    te = metas[1]
    frames = []
    for name, pred in (("overall_median", bl["pred_overall"]), ("per_length_median", bl["pred_per_prefix"])):
        d = evaluate.metrics_by_prefix(pd.DataFrame({"prefix_len": te["prefix_nr"], "y_true": te["y_reg"],
                                                     "y_pred": pred}), "regression")
        d["baseline"] = name
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    chk = {"max_len": max_len, "gap_train": gap_train,
           "mae_overall": bl["baseline_overall"]["mae"], "mae_per_prefix": bl["baseline_per_prefix"]["mae"],
           "n_test_prefixes": int(len(te)), "ref_overall": ref_overall}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    out.to_csv(cache, index=False)
    with open(chk_path, "w") as f:
        json.dump(chk, f, indent=1)
    return out, chk


# ----------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
    return plt


FS_COLORS = dict(zip(FS_ORDER, ["#8c8c8c", "#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]))


def by_prefix_frame(results_dir, job_id, task_type):
    p = os.path.join(results_dir, "tables", job_id + "__by_prefix.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    pred = load_pred(results_dir, job_id)
    return evaluate.metrics_by_prefix(pred, task_type) if pred is not None else None


def plot_by_prefix(results_dir, runs, task, fig_dir, baselines=None):
    plt = _plt()
    tt = task_type_of(task)
    metric = "auc" if tt == "classification" else "mae"
    done = runs[(runs["task"] == task) & (runs["status"] == "done") & ~runs["job_id"].str.startswith("tune__")]
    methods = [m for m in ["single_agg", "prefix_agg"] if m in set(done["method"])] + \
        sorted(set(done["method"]) - {"single_agg", "prefix_agg"})
    if not methods:
        return None
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 3.6), squeeze=False, sharey=True)
    for ax, method in zip(axes[0], methods):
        g = done[done["method"] == method]
        for fs in [f for f in FS_ORDER if f in set(g["fs"])]:
            jid = g[g["fs"] == fs]["job_id"].iloc[0]
            d = by_prefix_frame(results_dir, jid, tt)
            if d is None or not len(d):
                continue
            ax.plot(d["prefix_len"], d[metric], label=fs, color=FS_COLORS.get(fs), lw=1.6 if fs in
                    (BASE_FS, "FS-ALL") else 1.0)
        if baselines is not None:
            for name, ls in (("overall_median", ":"), ("per_length_median", "--")):
                b = baselines[baselines["baseline"] == name]
                ax.plot(b["prefix_len"], b["mae"], ls=ls, color="black", lw=1.2, label="baseline: " + name)
        ax.set_title("%s (%s) — %s" % (TASK_NAMES.get(task, task), task, method))
        ax.set_xlabel("in-phase prefix length j")
    axes[0][0].set_ylabel("AUC" if metric == "auc" else "MAE (days)")
    axes[0][-1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = os.path.join(fig_dir, "%s_by_prefix_%s.png" % (metric, task))
    fig.savefig(path); plt.close(fig)
    return path


def plot_importance(imp, fig_dir):
    if not len(imp):
        return None
    plt = _plt()
    d = imp[imp["fs"] == "FS-ALL"]
    if not len(d):
        return None
    piv = d.pivot_table(index=["task", "method"], columns="view", values="share", aggfunc="sum").fillna(0)
    piv = piv[[v for v in VIEWS if v in piv.columns]]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(piv) + 1.2))
    left = np.zeros(len(piv))
    colors = dict(zip(VIEWS, ["#8c8c8c", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728"]))
    labels = ["%s %s" % (TASK_NAMES.get(t, t), m) for t, m in piv.index]
    for v in piv.columns:
        ax.barh(labels, piv[v].to_numpy(), left=left, label=v, color=colors.get(v))
        left += piv[v].to_numpy()
    ax.set_xlabel("share of total XGBoost gain (FS-ALL)")
    ax.legend(ncol=len(piv.columns), fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(fig_dir, "importance_by_view_FS-ALL.png")
    fig.savefig(path); plt.close(fig)
    return path


def plot_drift(drift, fig_dir):
    if not len(drift):
        return None
    plt = _plt()
    tasks = list(dict.fromkeys(drift["task"]))
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.4 * len(tasks), 3.0), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        d = drift[drift["task"] == task]
        metric = "auc" if task_type_of(task) == "classification" else "mae"
        for fs in [f for f in FS_ORDER if f in set(d["fs"])]:
            e = d[d["fs"] == fs].sort_values("filing_year")
            ax.plot(e["filing_year"], e[metric], marker="o", ms=3, label=fs, color=FS_COLORS.get(fs))
        ax.set_title("%s — %s" % (TASK_NAMES.get(task, task), metric.upper()))
        ax.set_xlabel("filing year")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(fig_dir, "drift_by_filing_year.png")
    fig.savefig(path); plt.close(fig)
    return path


# ----------------------------------------------------------------------------- docs inputs
def parse_md_table(text, header_startswith):
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.startswith("|") and l.strip("| ").startswith(header_startswith):
            hdr = [c.strip() for c in l.strip().strip("|").split("|")]
            rows = []
            for l2 in lines[i + 2:]:
                if not l2.startswith("|"):
                    break
                cells = [c.strip() for c in l2.strip().strip("|").split("|")]
                rows.append(dict(zip(hdr, cells)))
            return pd.DataFrame(rows)
    return pd.DataFrame()


def md_section(text, title):
    m = re.search(r"^## %s\s*\n(.*?)(?=^## |\Z)" % re.escape(title), text, flags=re.S | re.M)
    return m.group(1).strip() if m else ""


def read_text(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


# ----------------------------------------------------------------------------- main build
def build(results_dir, out_path, cfg, cfg_path, data_dir, root, do_bootstrap=True, n_boot=evaluate.N_BOOT):
    t_start = time.time()
    fig_dir = os.path.join(results_dir, "figures"); tab_dir = os.path.join(results_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True); os.makedirs(tab_dir, exist_ok=True)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    runs = load_runs(results_dir)
    if not len(runs):
        runs = pd.DataFrame(columns=runner.RUNS_COLUMNS)
    status = job_status_table(cfg, runs, results_dir)
    status.to_csv(os.path.join(tab_dir, "job_status.csv"), index=False)
    fits = runs[~runs["job_id"].str.startswith("tune__")]
    done_fits = fits[fits["status"] == "done"]
    budget = cfg.get("budget", {})
    gap_train = int(budget.get("gap_train", 3))
    S = []  # sections
    notes = []  # things the report could not compute

    # ---- data
    data_json = {}
    for cand in [os.path.join(data_dir or "", "data_report.json"), os.path.join(root, "data/processed/data_report.json")]:
        if cand and os.path.exists(cand):
            with open(cand) as f:
                data_json = json.load(f)
            data_json_path = cand
            break
    data_md = read_text(os.path.join(root, "docs", "data_report.md")) or \
        read_text(os.path.join(data_dir or "", "data_report.md"))

    # ---- classification / regression summary tables
    def summary(tt):
        cols = (["auc", "f1", "pr_auc"] if tt == "classification" else
                ["mae", "median_ae", "r2_log", "baseline_mae_overall", "baseline_mae_per_prefix"])
        d = status[(status["kind"] == "fit") & status["task"].map(lambda t: task_type_of(t) == tt)].copy()
        d = d.merge(runs[["job_id", "n_test_prefixes", "n_test_segments", "max_prefix_len", "params_source"] + cols],
                    on="job_id", how="left")
        d["fs_rank"] = d["fs"].map({f: i for i, f in enumerate(FS_ORDER)}).fillna(99)
        d = d.sort_values(["task", "queue_tier", "method", "fs_rank"])
        d.drop(columns="fs_rank").to_csv(os.path.join(tab_dir, "summary_%s.csv" % tt), index=False)
        return d

    clf, reg = summary("classification"), summary("regression")

    # ---- bootstrap
    boot = bootstrap_table(runs, results_dir, n_boot=n_boot, enabled=do_bootstrap)
    if not do_bootstrap and not len(boot):
        notes.append("paired bootstrap skipped (--no-bootstrap) and no cache present")

    # ---- importance
    imp_rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, "importance", "*.csv"))):
        jid = os.path.basename(p)[:-4]
        if jid not in set(done_fits["job_id"]):
            continue
        d = pd.read_csv(p)
        parts = runner.parse_job_id(jid)
        for _, r in d.iterrows():
            imp_rows.append(dict(parts, job_id=jid, view=r["view"], gain=r["gain"], share=r["share"]))
    imp = pd.DataFrame(imp_rows)
    if len(imp):
        imp.to_csv(os.path.join(tab_dir, "importance_by_view.csv"), index=False)

    # ---- baselines per prefix (C13)
    baselines, bl_checks = {}, {}
    for task in [t for t in cfg["tasks"] if task_type_of(t) == "regression"]:
        r = done_fits[done_fits["task"] == task]
        if not len(r):
            continue
        L = int(r["max_prefix_len"].dropna().iloc[0])
        ref = r["baseline_mae_overall"].dropna()
        try:
            b, chk = baseline_by_prefix(results_dir, data_dir, task, L, gap_train,
                                        float(ref.iloc[0]) if len(ref) else None)
        except Exception as e:  # report must not die on a missing/odd data dir
            b, chk = None, "error: %s" % e
        if b is None:
            notes.append("C13 baseline curves for %s: %s" % (task, chk))
        else:
            baselines[task], bl_checks[task] = b, chk

    # ---- figures
    figs = {}
    for task in cfg["tasks"]:
        figs[task] = plot_by_prefix(results_dir, runs, task, fig_dir, baselines.get(task))
    fig_imp = plot_importance(imp, fig_dir)

    # ---- drift (Tier 3)
    drift_rows = []
    drift_fs = [BASE_FS, "FS-B0+G", "FS-ALL"]
    years = {}
    for task in cfg["tasks"]:
        ph = phase_of(task)
        for method in ["single_agg"]:
            for fs in drift_fs:
                jid = runner.make_job_id("tier1a", task, method, fs, "xgboost")
                if jid not in set(done_fits["job_id"]):
                    continue
                if ph not in years:
                    try:
                        years[ph] = segment_years(results_dir, data_dir, ph)
                    except Exception as e:
                        years[ph] = None; notes.append("filing years for phase %d: %s" % (ph, e))
                if years[ph] is None:
                    continue
                pred = load_pred(results_dir, jid)
                if pred is None:
                    continue
                d = evaluate.metrics_by_year(pred, years[ph], task_type_of(task))
                d["n_segments"] = pred.groupby(pred["segment_id"].map(years[ph]))["segment_id"].nunique().reindex(
                    d["filing_year"]).to_numpy()
                for _, r in d.iterrows():
                    drift_rows.append(dict(r, task=task, method=method, fs=fs, job_id=jid))
    drift = pd.DataFrame(drift_rows)
    if len(drift):
        drift.to_csv(os.path.join(tab_dir, "drift_by_year.csv"), index=False)
    elif cfg["tasks"]:
        notes.append("drift check: no filing_year source (data/processed or cached tables) or no tier1a predictions")
    fig_drift = plot_drift(drift, fig_dir)

    # ---- runtime
    rt = runs[["job_id", "tier", "task", "method", "fs", "status", "runtime_s", "flags"]].copy()
    rt.to_csv(os.path.join(tab_dir, "runtime.csv"), index=False)

    # ================================================================ 1. summary
    n_fit = (status["kind"] == "fit").sum()
    n_t1 = status[(status["kind"] == "fit") & status["queue_tier"].isin(["tier1a", "tier1b"])]
    bullets = []
    bullets.append("Jobs: %d of %d Tier-1 fits done; %d tuning jobs done of %d; %d fits/tunings failed or timed out; "
                   "%d Tier-1 fits not run yet." % (
                       (n_t1["status"] == "done").sum(), len(n_t1),
                       ((status["kind"] == "tune") & (status["status"] == "done")).sum(),
                       (status["kind"] == "tune").sum(),
                       status["status"].isin(["failed", "timeout"]).sum(),
                       (n_t1["status"].isin(["not run", "running"])).sum()))
    for task in [t for t in cfg["tasks"] if task_type_of(t) == "classification"][:2]:
        d = clf[(clf["task"] == task) & (clf["status"] == "done")].dropna(subset=["auc"])
        if len(d):
            best = d.loc[d["auc"].idxmax()]
            b0 = d[(d["fs"] == BASE_FS) & (d["method"] == best["method"])]
            bullets.append("RQ1 %s: best AUC %s (%s, %s); paper baseline FS-B0 with the same method: %s." % (
                TASK_NAMES[task], fmt(best["auc"]), best["fs"], best["method"],
                fmt(b0["auc"].iloc[0]) if len(b0) else "–"))
    for task in [t for t in cfg["tasks"] if task_type_of(t) == "regression"][:2]:
        d = reg[(reg["task"] == task) & (reg["status"] == "done")].dropna(subset=["mae"])
        if len(d):
            best = d.loc[d["mae"].idxmin()]
            bullets.append("RQ2 %s: best MAE %s days (%s, %s) vs. %s days for the per-length median baseline "
                           "and %s for the overall median." % (
                               TASK_NAMES[task], fmt(best["mae"], 1), best["fs"], best["method"],
                               fmt(best["baseline_mae_per_prefix"], 1), fmt(best["baseline_mae_overall"], 1)))
    if len(boot):
        sel = boot[(boot["fs"] == "FS-B0+G") & (boot["method"] == "single_agg")]
        parts_ = []
        for _, r in sel.iterrows():
            parts_.append("%s Δ%s %s [%s, %s]" % (TASK_NAMES.get(r["task"], r["task"]), r["metric"].upper(),
                                                   fmt(r["delta"], 3 if r["metric"] == "auc" else 1),
                                                   fmt(r["ci_low"], 3 if r["metric"] == "auc" else 1),
                                                   fmt(r["ci_high"], 3 if r["metric"] == "auc" else 1)))
        if parts_:
            bullets.append("RQ3, global extras (FS-B0+G − FS-B0, single_agg, 95%% CI): " + "; ".join(parts_) +
                           ". For AUC higher is better, for MAE lower is better.")
    flagged = runs[(runs["flags"].fillna("") != "") | (runs["status"] != "done")]
    if len(flagged):
        tune_flags = flagged[flagged["job_id"].str.startswith("tune__")]
        dp = fits[fits["flags"].fillna("").str.contains("default_params")]
        bullets.append("Flags: %d tuning job(s) hit the time cap (%s); %d fit(s) used default XGBoost params "
                       "because tuning did not finish." % (
                           len(tune_flags), ", ".join("%s: %s/%s" % (r["job_id"].replace("tune__", "").replace(
                               "__xgboost", ""), r["status"], r["flags"]) for _, r in tune_flags.iterrows()),
                           len(dp)))
    S.append("## 1. Summary\n\n" + "\n".join("- " + b for b in bullets[:8]))

    # ================================================================ 2. data
    sec = ["## 2. Data and cleaning", ""]
    if data_json:
        sec.append("Source: `%s` (generated %s) and `docs/data_report.md`." % (
            rel(data_json_path, root) if data_json_path.startswith(root) else data_json_path,
            data_json.get("generated", "?")))
        sec.append("")
        inp = pd.DataFrame([dict(file=k, size=fint(v.get("size")), lines=fint(v.get("lines")),
                                 md5="`%s`" % v.get("md5"), ok="%s" % (v.get("md5_ok") and v.get("lines_ok", True)))
                            for k, v in data_json.get("inputs", {}).items()])
        if len(inp):
            sec += ["**Inputs (MD5 checked at ingest)**", "", md_table(inp, headers=["File", "Bytes", "Lines (wc -l)",
                                                                                     "MD5", "OK"]), ""]
        pf = data_json.get("phase_fixes", {})
        steps = [
            ("Event-log rows (full log)", data_json.get("n_events_full_log")),
            ("Event rows of case-table cases", data_json.get("n_events_case_table_cases")),
            ("Cases in case table", data_json.get("n_cases_table")),
            ("Cases whose duplicate sum ≠ total_activity_count", data_json.get("dup_sum_mismatch_cases")),
            ("D1 cases (all / in window)", "%s / %s" % (fint(pf.get("flags_all", {}).get("D1")),
                                                        fint(pf.get("flags_window", {}).get("D1")))),
            ("D2 cases (all / in window)", "%s / %s" % (fint(pf.get("flags_all", {}).get("D2")),
                                                        fint(pf.get("flags_window", {}).get("D2")))),
            ("D3 cases dropped (all / in window)", "%s / %s" % (fint(pf.get("flags_all", {}).get("D3")),
                                                                fint(pf.get("flags_window", {}).get("D3")))),
            ("D4 cases (all / in window)", "%s / %s" % (fint(pf.get("flags_all", {}).get("D4")),
                                                        fint(pf.get("flags_window", {}).get("D4")))),
            ("Cases after D3 (used for global features)", pf.get("after_D3_all_cases")),
            ("D5 window cases, before → after D3", "%s → %s" % (fint(pf.get("window_before_fixes", {}).get("cases")),
                                                                fint(pf.get("window_after_fixes", {}).get("cases")))),
            ("D2 phase-1 exit moved to trial start (RD-18), window", pf.get("D2_phase1_exit_changed_window")),
        ]
        for k in ["1", "2", "3"]:
            steps.append(("Phase-%s segments built / dropped empty (D7)" % k, "%s / %s" % (
                fint(data_json.get("segments", {}).get("built", {}).get(k)),
                fint(data_json.get("traces", {}).get("p%s_empty_dropped" % k)))))
        sec += ["**Cleaning counts**", "", md_table(pd.DataFrame(
            [dict(step=s, n=v if isinstance(v, str) else fint(v)) for s, v in steps]), headers=["Step", "Count"]), ""]
        ph = []
        for k in ["p1", "p2", "p3"]:
            p = data_json.get("phases", {}).get(k)
            if not p:
                continue
            q = p.get("in_phase_len_quantiles", {})
            ph.append(dict(phase=k, segments=fint(p.get("segments")), rows=fint(p.get("rows")),
                           prefixes=fint(p.get("in_phase_rows")), rate=fmt(p.get("label_rate_deviant"), 3),
                           q50=fmt(q.get("0.5"), 0), q90=fmt(q.get("0.9"), 0),
                           q90pos=fmt(p.get("in_phase_len_q90_positive"), 0)))
        if ph:
            sec += ["**Segments per phase**", "", md_table(pd.DataFrame(ph), headers=[
                "Phase", "Segments", "Trace rows", "In-phase rows (prefixes)", "Label rate (moved on)",
                "In-phase length q50", "q90", "q90 positives"]), ""]
    else:
        sec.append("not available: `data/processed/data_report.json` not found.")
        notes.append("data section: data_report.json not found")
    S.append("\n".join(sec))

    # ================================================================ 3. protocol and deviations
    tun = budget.get("tuning", {})
    sec = ["## 3. Protocol and deviations from the paper", ""]
    sec.append("Settings read from `%s`:" % cfg_path)
    sec.append("")
    proto = [
        ("Tasks", ", ".join("%s = `%s`" % (TASK_NAMES.get(t, t), t) for t in cfg["tasks"])),
        ("Feature sets", ", ".join(cfg.get("feature_sets", []))),
        ("Tiers", "; ".join("%s: %s × %s" % (k, ",".join(v.get("methods", [])), ",".join(v.get("learners", [])))
                            for k, v in cfg.get("tiers", {}).items())),
        ("Split", "strict temporal (`split_data_strict`), 80% train / 20% test by first in-phase event"),
        ("Tuning", "TPE, %s iterations, %s-fold CV, ≤ %s training segments, CV gap %s, on %s; reused for the "
                   "other FS" % (tun.get("n_iter"), tun.get("cv_folds"), fint(tun.get("max_train_segments")),
                                 tun.get("gap"), cfg.get("tuned_fs"))),
        ("Prefixes", "in-phase lengths j = 1, 1+g, … ≤ L; L = min(%s, q90 in-phase length); train gap g = %s, "
                     "test gap 1" % (budget.get("max_prefix_cap"), budget.get("gap_train"))),
        ("Time caps", "tuning %s min, final fit %s min; over the cap: rerun on half the training segments, "
                      "then flagged" % (tun.get("cap_minutes"), budget.get("final_fit", {}).get("cap_minutes"))),
        ("XGBoost threads", str(budget.get("xgb_n_jobs"))),
        ("Bootstrap", "%d paired resamples of test segments, seed %d, 95%% percentile CI" % (n_boot, evaluate.BOOT_SEED)),
    ]
    sec += [md_table(pd.DataFrame([dict(k=a, v=b) for a, b in proto]), headers=["Setting", "Value"]), ""]
    sec += ["**Deviations from the paper's protocol (by design, PLAN.md §4)**", "",
            "- Prefixes are cut only inside the current phase (C6, D9); traces still start at filing, so the "
            "aggregation covers the whole case so far.",
            "- Regression tasks R1–R3 replace the outcome label (C1–C5): XGBRegressor on log1p(days), MAE loss.",
            "- Max prefix length L uses the in-phase length; training gap g = %s instead of 1 (C7, paper RQ3.3 "
            "setting)." % budget.get("gap_train"),
            "- Tuning budget reduced to %s TPE iterations on ≤ %s segments, tuned on %s only (8-hour budget)." % (
                tun.get("n_iter"), fint(tun.get("max_train_segments")), cfg.get("tuned_fs")),
            "- Only `single_agg` and `prefix_agg` with XGBoost in Tier 1; KNN, index-based, SVM, LSTM out of "
            "scope (D12).", ""]
    rd = parse_md_table(read_text(os.path.join(root, "RUN_DECISIONS.md")), "Time")
    if len(rd):
        rd = rd.rename(columns=lambda c: c.strip())
        sec += ["**Run decisions** (all entries of `RUN_DECISIONS.md`)", "",
                md_table(rd, cols=[c for c in ["ID", "Decision", "Choice", "Why"] if c in rd.columns]), ""]
    else:
        notes.append("RUN_DECISIONS.md not found or not parseable")
    S.append("\n".join(sec))

    # ================================================================ 4. RQ1
    def perf_table(d, cols, hdrs, nds):
        rows = []
        for _, r in d.iterrows():
            row = dict(task=TASK_NAMES.get(r["task"], r["task"]), method=r["method"], fs=r["fs"])
            if r["status"] == "done":
                for c, nd in zip(cols, nds):
                    row[c] = fmt(r[c], nd)
                row["n"] = fint(r["n_test_prefixes"])
                row["note"] = (r["reason"] or "").replace("|", "/")
            else:
                for c in cols:
                    row[c] = "–"
                row["n"] = "–"
                row["note"] = ("%s: %s" % (r["status"], r["reason"])).replace("|", "/")
            row["tuning"] = tuning_note(runs, r["task"], r["method"])
            rows.append(row)
        return md_table(pd.DataFrame(rows), cols=["task", "method", "fs"] + cols + ["n", "tuning", "note"],
                        headers=["Task", "Method", "FS"] + hdrs + ["Test prefixes", "Params", "Flags / status"])

    sec = ["## 4. RQ1 — phase-transition classification (T1–T3)", "",
           "AUC is pooled over all test prefixes (prefix-weighted, as in the paper); F1 at threshold 0.5; "
           "PR-AUC = average precision. T3 is secondary (D11).", ""]
    clf_t1 = clf[clf["queue_tier"].isin(["tier1a", "tier1b"])]
    if len(clf_t1):
        sec += [perf_table(clf_t1, ["auc", "f1", "pr_auc"], ["AUC", "F1@0.5", "PR-AUC"], [3, 3, 3]), ""]
    for task in [t for t in cfg["tasks"] if task_type_of(t) == "classification"]:
        if figs.get(task):
            sec.append("![AUC by prefix length, %s](%s)" % (task, rel(figs[task], out_dir)))
        else:
            sec.append("*%s: no finished runs, no plot.*" % task)
    S.append("\n".join(sec))

    # ================================================================ 5. RQ2
    sec = ["## 5. RQ2 — time remaining in the phase (R1–R3)", "",
           "MAE and median absolute error in days; R² on the log1p scale. Baselines (C13): training median of "
           "remaining days overall, and per in-phase prefix length.", ""]
    reg_t1 = reg[reg["queue_tier"].isin(["tier1a", "tier1b"])]
    if len(reg_t1):
        sec += [perf_table(reg_t1, ["mae", "median_ae", "r2_log", "baseline_mae_overall", "baseline_mae_per_prefix"],
                           ["MAE", "MedAE", "R² (log)", "Baseline MAE (overall median)",
                            "Baseline MAE (per-length median)"], [1, 1, 3, 1, 1]), ""]
    chk_rows = []
    for task, c in bl_checks.items():
        chk_rows.append(dict(task=TASK_NAMES[task], mine=fmt(c["mae_overall"], 2), ref=fmt(c.get("ref_overall"), 2),
                             per=fmt(c["mae_per_prefix"], 2), n=fint(c["n_test_prefixes"])))
    if chk_rows:
        sec += ["Baseline curves in the plots are rebuilt from `data/processed` with the runner's split and prefix "
                "rules; consistency check of the overall-median baseline against `runs.csv`:", "",
                md_table(pd.DataFrame(chk_rows), headers=["Task", "Rebuilt MAE (overall median)",
                                                          "runs.csv MAE (overall median)",
                                                          "Rebuilt MAE (per-length median)", "Test prefixes"]), ""]
    for task in [t for t in cfg["tasks"] if task_type_of(t) == "regression"]:
        if figs.get(task):
            sec.append("![MAE by prefix length, %s](%s)" % (task, rel(figs[task], out_dir)))
        else:
            sec.append("*%s: no finished runs, no plot.*" % task)
    S.append("\n".join(sec))

    # ================================================================ 6. RQ3
    sec = ["## 6. RQ3 — ablation against the paper baseline FS-B0", "",
           "Δ = metric(FS) − metric(FS-B0) on the same test prefixes; 95%% CI from %d paired bootstrap resamples "
           "of test segments (C11). Classification: Δ AUC (positive = better than FS-B0). Regression: Δ MAE in days "
           "(negative = better than FS-B0). A CI that excludes 0 is marked *." % n_boot, ""]
    if len(boot):
        b = boot.copy()
        b["fs_rank"] = b["fs"].map({f: i for i, f in enumerate(FS_ORDER)}).fillna(99)
        b = b.sort_values(["task", "tier", "method", "fs_rank"])
        rows = []
        for _, r in b.iterrows():
            nd = 3 if r["metric"] == "auc" else 1
            sig = "*" if (r["ci_low"] > 0 or r["ci_high"] < 0) else ""
            rows.append(dict(task=TASK_NAMES.get(r["task"], r["task"]), method=r["method"], fs=r["fs"],
                             metric="Δ" + r["metric"].upper(), d=fmt(r["delta"], nd) + sig,
                             ci="[%s, %s]" % (fmt(r["ci_low"], nd), fmt(r["ci_high"], nd)),
                             n=fint(r["n_segments"])))
        sec += [md_table(pd.DataFrame(rows), headers=["Task", "Method", "FS", "Metric", "Δ vs FS-B0", "95% CI",
                                                      "Test segments"]), ""]
        miss = done_fits[(done_fits["fs"] != BASE_FS) & ~done_fits["job_id"].isin(set(boot["job_id"]))]
        if len(miss):
            sec.append("No Δ for: " + ", ".join("`%s` (FS-B0 run of the same task/method missing)" % j
                                                 for j in miss["job_id"]) + ".")
    else:
        sec.append("not computed: no pair of finished runs with FS-B0, or bootstrap disabled.")
    sec += ["", "**Importance by view** (C12): share of total XGBoost gain per view (M = meta, P = paper pre-phase, "
                "P+ = court pre-phase extras, G = `open_cases`, G+ = global extras).", ""]
    if len(imp):
        piv = imp.pivot_table(index=["task", "method", "fs"], columns="view", values="share", aggfunc="sum")
        piv = piv[[v for v in VIEWS if v in piv.columns]].reset_index()
        piv["fs_rank"] = piv["fs"].map({f: i for i, f in enumerate(FS_ORDER)}).fillna(99)
        piv = piv.sort_values(["task", "method", "fs_rank"])
        views = [v for v in VIEWS if v in piv.columns]
        tb = pd.DataFrame([dict({"task": TASK_NAMES.get(r["task"], r["task"]), "method": r["method"], "fs": r["fs"]},
                                **{v: fmt(r[v], 3) if pd.notna(r[v]) else "–" for v in views})
                           for _, r in piv.iterrows()])
        sec += [md_table(tb, headers=["Task", "Method", "FS"] + views), ""]
        if fig_imp:
            sec.append("![Importance by view, FS-ALL](%s)" % rel(fig_imp, out_dir))
    else:
        sec.append("not available: no importance files for finished runs.")
    S.append("\n".join(sec))

    # ================================================================ 7. Tier 2/3
    sec = ["## 7. Tier 2 and Tier 3", ""]
    t2 = status[status["queue_tier"] == "tier2"]
    if len(t2):
        t2r = t2.merge(runs[["job_id", "auc", "mae"]], on="job_id", how="left")
        rows = []
        for _, r in t2r.iterrows():
            res = ("AUC %s" % fmt(r["auc"]) if task_type_of(r["task"]) == "classification" else
                   "MAE %s" % fmt(r["mae"], 1)) if r["status"] == "done" else "–"
            rows.append(dict(job=r["job_id"], status=r["status"], reason=(r["reason"] or "").replace("|", "/"),
                             result=res))
        sec += ["Tier 2 (queue state at report time):", "", md_table(pd.DataFrame(rows), headers=[
            "Job", "Status", "Reason / flags", "Result"]), ""]
    else:
        sec.append("Tier 2: not configured.")
    sec += ["Tier 3: the drift check is in section 8. Logistic/linear regression learners: not run — not part of "
            "the configured tiers (`%s`)." % cfg_path]
    S.append("\n".join(sec))

    # ================================================================ 8. drift
    sec = ["## 8. Drift check — test performance by filing year", "",
           "Computed from the saved Tier-1a predictions (single_agg) joined to `filing_year`; no retraining. The test "
           "set is the latest 20% of segments by phase entry, so the oldest filing years hold only long-running "
           "cases; read small-n years with care.", ""]
    if len(drift):
        rows = []
        for task in list(dict.fromkeys(drift["task"])):
            d = drift[drift["task"] == task]
            metric = "auc" if task_type_of(task) == "classification" else "mae"
            piv = d.pivot_table(index="filing_year", columns="fs", values=metric)
            nseg = d.groupby("filing_year")["n_segments"].max()
            for y, r in piv.iterrows():
                row = {"task": TASK_NAMES.get(task, task), "metric": metric.upper(), "year": int(y),
                       "n": fint(nseg.get(y))}
                for fs in drift_fs:
                    row[fs] = fmt(r.get(fs), 3 if metric == "auc" else 1)
                rows.append(row)
        sec += [md_table(pd.DataFrame(rows), cols=["task", "metric", "year", "n"] + drift_fs,
                         headers=["Task", "Metric", "Filing year", "Test segments"] + drift_fs), ""]
        if fig_drift:
            sec.append("![Drift by filing year](%s)" % rel(fig_drift, out_dir))
    else:
        sec.append("not computed: %s" % (notes[-1] if notes else "no data"))
    S.append("\n".join(sec))

    # ================================================================ 9. runtime
    sec = ["## 9. Runtime", ""]
    if len(runs):
        tot = runs.groupby("tier")["runtime_s"].agg(["count", "sum"]).reset_index()
        tot_rows = [dict(tier=r["tier"], n=fint(r["count"]), m=fmt(r["sum"] / 60, 1)) for _, r in tot.iterrows()]
        tot_rows.append(dict(tier="**total**", n=fint(len(runs)), m=fmt(runs["runtime_s"].sum() / 60, 1)))
        sec += [md_table(pd.DataFrame(tot_rows), headers=["Tier", "Jobs recorded", "Runtime (min)"]), ""]
        rows = [dict(job=r["job_id"], status=r["status"], s=fmt(r["runtime_s"], 1),
                     flags=(r["flags"] or "").replace("|", "/")) for _, r in runs.iterrows()]
        sec += ["<details><summary>Per job</summary>", "", md_table(pd.DataFrame(rows), headers=[
            "Job", "Status", "Runtime (s)", "Flags"]), "", "</details>"]
    else:
        sec.append("no runs recorded.")
    S.append("\n".join(sec))

    # ================================================================ 10. limitations
    sec = ["## 10. Limitations and open issues", ""]
    lim = md_section(data_md, "Limitations")
    if lim:
        sec += ["From the data report:", "", lim, ""]
    bad = status[status["status"].isin(["failed", "timeout"])]
    flagged_fits = runs[(runs["status"] == "done") & (runs["flags"].fillna("") != "")]
    items = []
    for _, r in bad.iterrows():
        items.append("`%s`: %s (%s)." % (r["job_id"], r["status"], r["reason"] or "no detail"))
    for _, r in flagged_fits.iterrows():
        items.append("`%s`: finished with flags `%s`." % (r["job_id"], r["flags"]))
    nr = status[status["status"].isin(["not run", "running"])]
    for reason, g in nr.groupby("reason"):
        items.append("%d job(s) not run (%s), e.g. `%s`." % (len(g), reason, g["job_id"].iloc[0]))
    items.append("The upstream port was validated on a synthetic log only (P4 skipped: the paper's datasets were "
                 "not provided), so there is no published-AUC reproduction check.")
    items.append("Hyperparameters were tuned on %s and reused for the other feature sets of the same task and method, "
                 "which can favour %s." % (cfg.get("tuned_fs"), cfg.get("tuned_fs")))
    sec += ["Run issues:", ""] + ["- " + i for i in items]
    if notes:
        sec += ["", "Not computed by this report:", ""] + ["- " + n for n in notes]
    S.append("\n".join(sec))

    # ================================================================ 11. reproduce
    S.append("\n".join([
        "## 11. Commands to reproduce", "", "```",
        "export PYTHONPATH=$PWD:$PWD/experiments",
        "python -m court.run --config %s            # data → tuning → runs → report (resumable)" % cfg_path,
        "# or stage by stage:",
        "python -m court.data_prep.build",
        "python -m court.run_queue --config %s --tier tier1a --tier tier1b" % cfg_path,
        "python -m court.report --results-dir %s --out %s" % (rel(results_dir, root) if os.path.abspath(
            results_dir).startswith(root) else results_dir, rel(out_path, root) if os.path.abspath(out_path).startswith(
            root) else out_path),
        "python -m unittest discover -s tests", "```"]))

    head = ["# Results — court-case predictive process monitoring", "",
            "*Generated by `python -m court.report` on %s from `%s` (%d runs.csv rows). Do not edit by hand.*" % (
                time.strftime("%Y-%m-%d %H:%M"), results_dir, len(runner.read_runs(results_dir))), ""]
    text = "\n".join(head) + "\n\n".join(S) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return {"out": out_path, "seconds": time.time() - t_start, "notes": notes, "n_boot_pairs": len(boot)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out", default=None, help="default: results.md in the repo root")
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--root", default=ROOT, help="repo root holding RUN_DECISIONS.md and docs/")
    ap.add_argument("--no-bootstrap", action="store_true", help="use cached bootstrap results only")
    ap.add_argument("--n-boot", type=int, default=evaluate.N_BOOT)
    a = ap.parse_args(argv)
    cfg_path = a.config
    cfg = runner.load_config(cfg_path if os.path.isabs(cfg_path) else os.path.join(a.root, cfg_path))
    results_dir = a.results_dir or os.path.join(a.root, cfg.get("results_dir", "results"))
    data_dir = a.data_dir or os.path.join(a.root, cfg.get("data_dir", "data/processed"))
    out = a.out or os.path.join(a.root, "results.md")
    info = build(results_dir, out, cfg, cfg_path, data_dir, a.root, do_bootstrap=not a.no_bootstrap, n_boot=a.n_boot)
    print("wrote %s in %.1fs (%d bootstrap pairs)" % (info["out"], info["seconds"], info["n_boot_pairs"]))
    for n in info["notes"]:
        print("note:", n)


if __name__ == "__main__":
    main()
