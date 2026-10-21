# HANDOFF — 8-hour autonomous run, 2026-10-21

**Run:** T0 02:04 IDT → handoff 08:15 IDT (deadline 09:49). The run was planned in PLAN.md (approved, all D1–D12 defaults, cut to 8h).
**Repo:** `github.com/ilanale/predictive-monitoring-benchmark`, branch **`court-adaptation`**; local clone at `claude-run/predictive-monitoring-benchmark/`.
**Deliverables (this folder):**
- `court-ppm_2026-10-21.zip`: code + results via `git archive`, no data
- `court-ppm_2026-10-21.bundle`: full git history
- this file

All numbers below are copied from the generated `results.md`, and auditor A4 checked every number in it against the raw result files (VERIFICATION.md, G6).

## What's done
- **WS1 data:** `data/processed/court_phase{1,2,3}`.
  - 127,610 / 60,412 / 2,484 segments.
  - Built from the full event log (MD5 checked) in about 30 s.
  - D1–D12 applied. An independent audit re-derived all 190,506 segments and 800 feature cells from the raw files with 0 mismatches, after one fix: the D2 phase-1 exit date, which leaked trial events (RD-18).
- **WS2 port:** a single `compat:` commit runs the 2019 code on Python 3.15 / pandas 3 / sklearn 1.9 / xgboost 3.4. Its behaviour is pinned by a synthetic-log baseline test (DoD-4).
- **WS3 pipeline:**
  - C1–C13 regression/classification switches.
  - A fast path identical to the upstream path on real data (max diff 3e-17, identical predictions).
  - Paired bootstrap, importance by view, median baselines, a resumable queue and a one-command rebuild (`python -m court.run`).
  - 52 tests pass.
- **Experiments:** Tier 1 = 62 of 72 fits + 10 of 12 tunings (XGBoost; single_agg and prefix_agg × 6 feature sets × 6 tasks).
- **Docs:**
  - `results.md`: generated, 11 sections.
  - `README.md` (DoD-8) and `CHANGES_FROM_UPSTREAM.md`.
  - `DECISIONS.md` (D1–D12 only) and `RUN_DECISIONS.md` (RD-01..RD-30).
  - `VERIFICATION.md`: G2, G4 and G6 all PASS, 0 open blockers.

## Headline results (single_agg, test = last 20% of segments by phase entry)
- **RQ1, classification:**
  - Best T1 (will the case get an answer/plea?) is FS-MP with AUC 0.825; the paper baseline FS-B0 gets 0.813 (Δ +0.012, 95% CI [0.011, 0.014]).
  - T2 (trial or summary judgment): FS-MP 0.797 vs FS-B0 0.788.
  - T3 (verdict): about 0.93 for every feature set that uses the trace.
  - Meta alone loses 0.07–0.21 AUC everywhere, so the case's own event history is what carries the signal.
- **RQ2, days left in the phase:** the models beat the training-median baselines.

  | Task | Best MAE (days) | Median baseline (days) |
  |---|---|---|
  | R1 | 144 | 156 |
  | R2 | 220 | 257 |
  | R3 | 256 (prefix_agg) | 294 |

  R3 R² on the log scale is about 0, so R3 is weak.
- **RQ3, the global (inter-case) view:**
  - Your G+ extras **lower** classification AUC: T1 Δ −0.011 [−0.014, −0.008], T2 Δ −0.031 [−0.051, −0.010].
  - Even the paper's single feature, `open_cases`, costs T1 about 0.013 (FS-MP vs FS-B0+P).
  - G+ **helps** R2 (Δ MAE −4.5 days [−6.7, −2.6]; FS-ALL −6.0) and is neutral or noisy for phase 3.
  - The audit traced the classification loss to strong temporal drift in the G features.
    - Test values sit 1.5–3 SD above train.
    - Court-wide `open_cases` peaks at about 14,800 in 2016 and falls to about 120 by 2021, because the case table holds **only closed cases**.
    - Restricting the test set to prefixes ending ≤ 2016 shrinks the loss (T1 −0.008, T2 −0.020 with a CI touching 0).
  - **Read the G results as "global features under drift and a censoring artifact"**, not as a clean negative result.

## What isn't done, and why
- **Tier 1 gaps:**
  - 8 phase-2 prefix_agg fits were cut (RD-28) to meet the deadline.
  - 2 R1 prefix_agg fits (FS-B0+G, FS-ALL) timed out twice (RD-29).
  - R2 tuning timed out, so all R2 fits use XGBoost default params (RD-24).
  - R1 was tuned on half the sample.
- **Tier 2 not run:** FS-B0 re-tune, D1/D8 sensitivity, state/cluster/laststate bucketing and RF. Reasons:
  - Claude hit a usage limit from 02:51 to 05:40; the queue kept running, but the orchestration and agents were paused (RD-25).
  - The extra bucketing methods are not wired into the court runner.
- **Tier 3:** only the drift check ran. Logistic/linear regression was not run.
- **P4 skipped:** the port was checked on a synthetic log only, with no reproduction of the paper's AUCs.
- **⚠ Needs your decision (RD-30):** the independent auditor committed 800 row-level audit cells (`verification/g2_feature_cells*.csv`: segment id, date, feature value) at 02:2x. They were pushed with `court-adaptation`. They are removed at HEAD and are not in the zip, but they are still in the branch history. Rewriting published history needs your OK. To purge them:
  ```bash
  cd ~/Documents/Thesis/claude-run/predictive-monitoring-benchmark
  pip install git-filter-repo
  git filter-repo --path-glob 'verification/g2_feature_cells*' --invert-paths --refs court-adaptation
  git push --force origin court-adaptation
  ```
  `git filter-repo` removes the `origin` remote, so re-add it first with `git remote add origin https://github.com/ilanale/predictive-monitoring-benchmark.git`.

## Next three steps
1. **Fix the G view before drawing RQ3 conclusions.**
   - Recompute the global features from the **full event log** (175,268 cases, including the 7,161 still open), not only the closed-case table.
   - Add detrended or relative variants (judge load / court load, z-scored by year).
   - Switch tuning CV to time-ordered folds (RD-27).
   - Then rerun Tier 1a; it takes about 1.5 h on this Mac.
2. **Close the Tier-1 gaps and run Tier 2:**
   - R2 tuning with a 45-min cap.
   - The 2 R1 prefix_agg timeouts and the 8 cut phase-2 fits.
   - The FS-B0 re-tune, and the D1 and D8 sensitivity runs.

   Use `python -m court.run_queue --tier tier1b --tier tier2` after raising the caps in `configs/experiments.yaml`.
3. **Check two data assumptions against the source:**
   - Whether `District_Judge` in the case table is the judge at the end of the case.
   - The judge-level jump in 2016 (judge counts roughly double).

   Then start the thesis chapter from `results.md`, sections 4–8.
