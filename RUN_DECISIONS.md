# RUN_DECISIONS — decisions made during the autonomous run

D1–D12 are fixed (see `DECISIONS.md`). Every other decision taken during the run is logged here. Rule: the conservative option (less leakage, fewer assumptions) wins.

| Time (IDT) | ID | Who | Decision | Options | Choice | Why |
|---|---|---|---|---|---|---|
| 02:15 | RD-01 | orchestrator | Interface between WS1 and WS3 | ad hoc vs. written contract | `court/SCHEMA.md`, fixed at bootstrap | lets A2 and A3 work in parallel without guessing column names |
| 02:15 | RD-02 | orchestrator | FS-MP contents ("M + P, no global") | M+P paper part only / M+P+P+ | M + P + P+ | PLAN.md §1.1 defines the Pre-phase view as including P+; FS-MP is the "intra-case only" set |
| 02:15 | RD-03 | orchestrator | Who registers court tasks in `experiments/dataset_confs.py` (WS1 step 8) | A2 edits the upstream file / A2 writes `court/court_confs.py`, A3 wires it in | the second | A2 must not touch upstream code (PLAN.md §6); avoids conflicts with A1's compat commit |
| 02:15 | RD-04 | orchestrator | Agent isolation | branches only / git worktrees | worktrees `../worktrees/wt-{port,data,pipeline}` on branches `ws-*`; all share the main repo's `data/` via symlink | parallel work without checkout clashes; one copy of the large data |
| 02:20 | RD-05 | orchestrator | When A3 starts | after G1 (plan) / now, on `court/`-only modules, upstream edits after G1 | start now, phased | saves ~1h of the 8h budget; no merge conflicts because upstream files are edited only after A1's compat commit is merged |
