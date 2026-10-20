"""Resumable experiment queue with hard time caps (PLAN.md §5, DoD-2/5).

    python -m court.run_queue --config configs/experiments.yaml [--tier tier1a] [--only court_p1]
        [--dry-run] [--sample-frac 0.02] [--results-dir results]

* expands jobs: per tier (tier_order), per task (T1, R1, T2, R2, T3, R3), per method/learner:
  one tuning job (on ``tuned_fs``; shared across tiers) then one fit job per feature set;
* skips jobs whose job_id has status ``done`` in ``results/runs.csv``;
* each job runs in its own subprocess (``python -m court.runner --job-id ...``) with a hard cap
  (tuning ``budget.tuning.cap_minutes``, fit ``budget.final_fit.cap_minutes``); on timeout the
  process is killed and rerun once on half the training segments (flag ``timeout_halved``);
  a second timeout / a crash is recorded with status ``timeout`` / ``failed`` and the queue goes on;
* logs to ``<results_dir>/queue.log``.
"""
import argparse
import os
import subprocess
import sys
import time

from court import runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def expand_jobs(cfg, tiers=None, only=None):
    jobs, seen = [], set()
    tasks = cfg["tasks"]
    for tier in cfg.get("tier_order", list(cfg["tiers"])):
        if tiers and tier not in tiers:
            continue
        tc = cfg["tiers"][tier]
        fss = tc.get("feature_sets", cfg["feature_sets"])
        learners = tc.get("learners", [tc["learner"]] if "learner" in tc else ["xgboost"])
        for task in tasks:
            for method in tc.get("methods", []):
                for learner in learners:
                    tid = runner.tune_id(task, method, learner)
                    if tid not in seen:
                        seen.add(tid); jobs.append({"job_id": tid, "kind": "tune", "tier": tier})
                    for fs in fss:
                        jid = runner.make_job_id(tier, task, method, fs, learner)
                        if jid not in seen:
                            seen.add(jid); jobs.append({"job_id": jid, "kind": "fit", "tier": tier})
        for ej in tc.get("extra_jobs", []) or []:
            jid = runner.make_job_id(tier, ej["task"], ej["method"], ej["fs"], ej["learner"])
            if jid not in seen:
                seen.add(jid); jobs.append({"job_id": jid, "kind": "fit", "tier": tier})
    if only:
        pats = [p for p in only.split(",") if p]
        jobs = [j for j in jobs if any(p in j["job_id"] for p in pats)]
    return jobs


def job_method(job_id):
    parts = job_id.split("__")
    return parts[2] if parts[0] == "tune" else parts[2]


def done_ids(results_dir):
    runs = runner.read_runs(results_dir)
    return set(runs.loc[runs["status"] == "done", "job_id"]) if len(runs) else set()


def log(results_dir, msg):
    os.makedirs(results_dir, exist_ok=True)
    line = "%s %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(results_dir, "queue.log"), "a") as f:
        f.write(line + "\n")


def run_one(job, args, cap_s, train_frac=None, flags=""):
    cmd = [sys.executable, "-m", "court.runner", "--job-id", job["job_id"], "--config", args.config,
           "--data-dir", args.data_dir, "--results-dir", args.results_dir]
    if args.sample_frac:
        cmd += ["--sample-frac", str(args.sample_frac)]
    if train_frac:
        cmd += ["--train-frac", str(train_frac)]
    if flags:
        cmd += ["--flags", flags]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=cap_s)
        return ("ok" if p.returncode == 0 else "failed"), time.time() - t0, (p.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - t0, ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--tier", action="append", help="tier name(s); default: all in tier_order")
    ap.add_argument("--only", default=None, help="comma-separated substrings of job ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    cfg = runner.load_config(args.config)
    args.data_dir = args.data_dir or cfg.get("data_dir", "data/processed")
    args.results_dir = args.results_dir or cfg.get("results_dir", "results")
    jobs = expand_jobs(cfg, args.tier, args.only)
    done = done_ids(args.results_dir)
    b = cfg.get("budget", {})
    caps = {"tune": 60 * float(b.get("tuning", {}).get("cap_minutes", 20)),
            "fit": 60 * float(b.get("final_fit", {}).get("cap_minutes", 10))}
    todo = [j for j in jobs if j["job_id"] not in done]
    log(args.results_dir, "queue: %d jobs, %d done, %d to run%s" % (
        len(jobs), len(jobs) - len(todo), len(todo), " (dry run)" if args.dry_run else ""))
    for job in todo:
        if job_method(job["job_id"]) in runner.PHASE2_METHODS:
            log(args.results_dir, "SKIP %s (method needs upstream bucketers: phase 2)" % job["job_id"])
            continue
        if args.dry_run:
            log(args.results_dir, "DRY %s cap=%ds" % (job["job_id"], caps[job["kind"]]))
            continue
        log(args.results_dir, "START %s" % job["job_id"])
        status, dt, err = run_one(job, args, caps[job["kind"]])
        if status == "timeout":
            log(args.results_dir, "TIMEOUT %s after %.0fs; rerun on half the training segments" % (job["job_id"], dt))
            status, dt, err = run_one(job, args, caps[job["kind"]], train_frac=0.5, flags="timeout_halved")
        if status == "ok":
            log(args.results_dir, "DONE %s %.0fs" % (job["job_id"], dt))
            continue
        parts = runner.parse_job_id(job["job_id"]) if job["kind"] == "fit" else {}
        runner.append_run(args.results_dir, dict(parts, job_id=job["job_id"], status=status, runtime_s=round(dt, 1),
                                                 flags=("timeout_halved" if status == "timeout" else
                                                        err.strip().splitlines()[-1][:200] if err.strip() else ""),
                                                 finished_at=time.strftime("%Y-%m-%d %H:%M:%S")))
        log(args.results_dir, "%s %s %.0fs %s" % (status.upper(), job["job_id"], dt, err.strip()[-500:]))
    log(args.results_dir, "queue finished")


if __name__ == "__main__":
    main()
