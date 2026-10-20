# STATUS

**T0 = 2026-10-21 02:04 IDT.** 8-hour autonomous run (PLAN.md, approved 2026-10-21). Handoff by **09:49 IDT** (T0+7:45).

Machine: Apple M5, 10 cores (4P+6E), 24 GB RAM (70% cap ≈ 16.8 GB), 267 GB free. Python 3.15.0 venv at `../.venv` (xgboost 3.4.1, hyperopt 0.3.0, pandas 3.0.6, scikit-learn 1.9.1).

| Gate | Due (IDT) | State |
|---|---|---|
| Bootstrap | 02:19 | |
| G1 port | 03:19 | |
| G2 data | 04:04 | |
| G3 pipeline | 05:19 | |
| G4 review → Tier 1a | 05:34 | |
| Tier-1b projection / cut check | 07:34 | |
| G5 Tier 1 + Tier-2 cut-off | 08:04 | |
| Writing starts (latest) | 08:19 | |
| G6 results | 09:34 | |
| Handoff | 09:49 | |

## Log (one line per milestone: time · who · what)
- 02:04 · orchestrator · T0. Fork cloned, branch `court-adaptation`, upstream push disabled, `data/raw` symlinks.
