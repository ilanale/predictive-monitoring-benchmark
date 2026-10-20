"""Single entry point (DoD-2): data build -> experiment queue -> report.

    python -m court.run --config configs/experiments.yaml [--skip-data] [--tier tier1a] [--sample-frac f]
"""
import argparse
import importlib.util
import os
import subprocess
import sys

from court import runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def step(cmd):
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--skip-data", action="store_true", help="reuse data/processed if it exists")
    ap.add_argument("--tier", action="append")
    ap.add_argument("--sample-frac", type=float, default=None)
    a = ap.parse_args(argv)
    cfg = runner.load_config(os.path.join(ROOT, a.config) if not os.path.isabs(a.config) else a.config)
    data_dir = cfg.get("data_dir", "data/processed")
    have_data = os.path.exists(os.path.join(ROOT, data_dir, "court_columns.json"))
    if not (a.skip_data and have_data):
        step([sys.executable, "-m", "court.data_prep.build"])
    q = [sys.executable, "-m", "court.run_queue", "--config", a.config]
    for t in a.tier or []:
        q += ["--tier", t]
    if a.sample_frac:
        q += ["--sample-frac", str(a.sample_frac)]
    step(q)
    if importlib.util.find_spec("court.report") is not None:
        step([sys.executable, "-m", "court.report", "--config", a.config])
    else:
        print("court.report not present yet; skipping report", flush=True)


if __name__ == "__main__":
    main()
