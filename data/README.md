# data/ (git-ignored except this file)

- `raw/` — symlinks to the read-only inputs in PLAN.md §0.3. Never copy, edit or move the originals.
- `interim/` — compact parquet of the full event log (case-table cases only), built by `court/data_prep/ingest.py`.
- `processed/` — `court_phase{1,2,3}.csv` (`;`-separated, benchmark format) and `data_report.md`.
