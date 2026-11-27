# Phase I freeze (2026-11-27)

Phase I is closed: tuning, training (Tier A), the frozen held-out evaluation, the research analysis and its audit.
From this point the files listed below are a fixed record. Phase II must not edit them, re-run them in place, or re-analyse the held-out results with Phase II knowledge.

## Status at freeze

| Item | State |
|---|---|
| Final evaluation | 11 checkpoints, `final/freeze_v1.yaml` (ff0f0ce5…a453), lock written 12 s before the first evaluation |
| Audit | 217 checks, 0 failed; report reproduced independently, 0 mismatches |
| Post-evaluation integrity | **Closed.** `TRAINED_RUNS_UNCHANGED`; 43 checks, 0 failed (`AUDIT.md` section 4) |
| Open audit items | None material. Attested, not re-runnable: the server-side `sha256sum -c` run, and `final_eval.log` (not recovered) |
| GRPO-A_base_s2026 | FAILED at step 98. C4 unavailable (counted p = 1). A1/A2 post-hoc only |

## Manifests

| File | Covers |
|---|---|
| `PHASE1_FREEZE.sha256` | Every file under `results/research_analysis/` and `results/final_eval/`, the analysis and audit scripts, and `final/freeze_v1.yaml`. Paths are relative to the project root |
| `PHASE1_INPUTS.sha256` | The recovered server evidence the analysis read: `~/a5_final` (FINAL_CHECKPOINTS.json, FINAL_EVAL.sha256, BEFORE_FINAL.sha256, post_final_integrity.log) and the run-tree manifests in `~/a5_sessions` and `~/a5_recovery`. Paths are relative to the WSL home directory |

The sha256 of `PHASE1_FREEZE.sha256` itself is recorded in the Phase II plan (`PHASE2_PLAN.md`), which is outside the frozen set.

## Verify

```bash
# project root (WSL or Git Bash)
sha256sum -c --quiet results/research_analysis/PHASE1_FREEZE.sha256 && echo PHASE1_OUTPUTS_UNCHANGED
# WSL home
cd ~ && sha256sum -c --quiet <repo>/results/research_analysis/PHASE1_INPUTS.sha256 && echo PHASE1_INPUTS_UNCHANGED
```

## Rules for Phase II

1. Phase I numbers are quoted from these files, never recomputed into them.
2. Corrections go into a new, dated erratum file next to this one. The frozen file stays as it is.
3. Phase II outputs go under `results/phase2/`.
4. The original run trees on the server (`runs/sft`, `runs/grpo`, `runs/recovery`, `runs/final_eval`) stay read-only. Phase II runs use new folders.
