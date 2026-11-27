# Phase 1: final artifact audit

Date: 2026-11-26. Read-only. Generated evidence: `AUDIT_GENERATED.md` and `audit.json` (217 checks), plus `training_audit/` (the preparation-phase audit re-run on the same trees).
Reproduce with `scripts/research_analysis.py` (command at the end of this file).

## Verdict

| | Count |
|---|---|
| Checks run | 217 |
| Failed | **0** |
| Not verifiable from the laptop | **2**, both low-stakes: local re-hash of weight files (covered by matching records instead); the launcher log. The third, post-evaluation integrity, was **closed on 2026-11-27** (section 4) |
| Post-evaluation integrity (`post_final_integrity/`) | 43 checks, 0 failed: `TRAINED_RUNS_UNCHANGED` |
| Training-artifact audit (`final_audit.py`, re-run) | 0 unresolved; 2 explained (unchanged from the preparation audit) |
| Statistics in `results/final_eval/report.json` reproduced independently | 238 metric and 382 comparison fields; 0 mismatches; largest absolute deviation 4.9e-7 |

The held-out results are internally consistent. They are traceable to the locked checkpoints and were produced under the frozen protocol.
The post-evaluation integrity evidence was supplied on 2026-11-27 and verified (section 4). Phase I is frozen (`PHASE1_FREEZE.md`).

## 1. What was verified

**Freeze and lock**
- `final/freeze_v1.yaml` still hashes to ff0f0ce5…a453. The lock (`FINAL_CHECKPOINTS.json`) and `report.json` both record this hash.
- The lock lists exactly the 11 frozen checkpoints.
- The lock was written at Unix time 1795743992, 12 s before the first evaluation started.
- The failed `GRPO-A_base_s2026` is absent from the lock and from `~/a5_final`.
  - Its incident record says FAILED, attempted step 98, gate 0.0794 > 0.05.
  - Its run folder has no DONE.json.

**Checkpoint identity**
- For each of the 10 trained checkpoints, the locked weights equal the `weights_sha256` in the training run's DONE.json lineage, and its `checkpoint/lineage.json` agrees.
- BASE equals the registered Qwen2.5-Math-1.5B weights (f7aa8d66…).
- Every evaluation's `summary.json` records evaluating the locked hash, at the frozen lineage stage.
- GRPO-B's parent is SFT-reverified_n512_s0 (chain base > sft > grpo).

**Evaluation protocol**, identical for all 11 checkpoints:
- the same code hash (0e0b0014…, equal to the lock);
- verifier 2.0.0 with the locked `verify.py` and `grader.py` hashes;
- vLLM 0.7.2 on an RTX A6000, max_model_len 2048, prefix caching off, gpu_mem 0.8, greedy decoding, 1,024 new tokens, seed 2026;
- the frozen prompts.

The configurations differ only in run folder, weights and prompt list.

**Dataset identity**
- Both held-out files' sha256 match `data/stage0/MANIFEST.json` (GSM8K test 9b99a963…, MATH val 526d0477…) in all 24 evaluation files.
- n = 1,319 and 5,000, with unique ids. Every evaluation used the same item ids.

**Completion and counts**
- Every checkpoint has DONE.json.
- Item-level correct counts equal summary.json and DONE.json.
- There were 0 verifier errors.

**Isolation**
- The held-out item ids are disjoint from the development ids (GSM8K dev, MATH dev) and from the 4,423 SFT training ids (the union over all SFT arms).
- Text-level decontamination was not repeated; the Stage 0 result stands.

**Manifests**
- `FINAL_EVAL.sha256` lists exactly the 58 recovered files, and all 58 match. The lock is among them.
- `SESSION_A.sha256`, `SESSION_B.sha256` and `RECOVERY.sha256` re-verify.
- The session manifests were written retrospectively, as recorded in the preparation audit.

**Training artifacts**
- `scripts/final_audit.py` re-run: 0 unresolved.
- Still explained, not problems:
  - 25 duplicated metric rows in SFT-reverified_full_s0 (a recorded resume);
  - 15 weight/state files that were not downloaded.

**Report reproduction**
- Every accuracy, Wilson interval, second-verifier and relaxed rate, format rate, length, truncation count, paired difference, CI, McNemar p and Holm p in `report.json` was recomputed from item files.
- The recomputation uses separate code: scipy `binomtest` for the exact McNemar test and Wilson interval, and a numpy Holm implementation.
- The only differences are Wilson bounds at about 5e-7, because scipy uses z = 1.959964 and the report uses 1.96.

## 2. Corrections to how `REPORT.md` should be read (the generated report is unchanged)

1. **"supported" in the generated report means "a difference was detected" (two-sided).** It does not mean a hypothesis direction was confirmed. Three rows need care:
   - **C1 and C2:** re-verified data scored lower than exact-match data. The motivating expectation, that recovered traces would help, is therefore contradicted, not supported.
   - **"SFT reverified_n128 vs base question-only"** (GSM8K −4.2 pp, MATH −8.8 pp): this SFT arm is worse than the base model's question-only prompt.
   - `statistical_comparisons.csv` adds a plain-language `reading` column that states the direction.
2. **C4 is unavailable** because the pre-registered GRPO-A failed. It is counted with p = 1, so the confirmatory family keeps m = 4. A1 and A2 are not used for C4 anywhere.
3. **GPU-hours** in the report are 2 × the training run's wall-clock and include in-training dev evaluations.
   - For A1 and A2 the figure covers only steps 81–100: 2.28 and 2.51 GPU-h.
   - Their cumulative cost including GRPO-A steps 1–80 is 7.36 and 7.59 GPU-h.

## 3. Not verifiable from the laptop

| Item | Why | What would close it |
|---|---|---|
| ~~Trained runs unchanged by the evaluation~~ | **Closed 2026-11-27**, see section 4 | – |
| Weight files re-hashed locally | Weights stay on the server | Already covered by the pre-run `check` (11 hashes matched) and by each summary.json recording the locked hash; listed for completeness |
| `final_eval.log` | Not recovered | Optional; the launch order is reconstructed from run.json and DONE.json timestamps (span 1.49 h) |

The evaluation is read-only by construction: it loads weights and writes only under `runs/final_eval/`. Even so, the post-evaluation check is the evidence the plan promised, so it was reported as not verified until the evidence arrived. It is now closed (section 4).

## 4. Post-evaluation integrity: closed (2026-11-27)

**Evidence supplied by the user**, recovered from the server to `~/a5_final/`:

| File | sha256 | Bytes |
|---|---|---|
| `BEFORE_FINAL.sha256` | f56eeb52…d382a | 39,229 (349 entries) |
| `post_final_integrity.log` | edc8bb46…9c60e | 23 |

**Verified by `scripts/verify_post_final_integrity.py`: 43 checks, 0 failed.** Output is in `post_final_integrity/post_final_integrity.json`.

1. **The log is the complete expected output.**
   - It contains exactly one line, `TRAINED_RUNS_UNCHANGED`.
   - The plan's step-4 command is `sha256sum -c --quiet BEFORE_FINAL.sha256 && echo TRAINED_RUNS_UNCHANGED`. With `--quiet`, matching files print nothing, and the echo runs only when every file matched. A single sentinel line is therefore the full output of a pass.
2. **The manifest is well-formed and in scope.**
   - All 349 lines have the form `<sha256>  <path>`, with unique, relative paths.
   - Every path is under `sft/`, `grpo/` or `recovery/` (17 runs, including the failed GRPO-A and the tuning runs).
   - It lists no `final_eval` output.
3. **It covers the locked checkpoints.** For all 10 trained checkpoints in the lock, the `checkpoint/model.safetensors` hash in BEFORE_FINAL equals the lock hash. Each run's DONE.json, run.json and checkpoint files are listed.
4. **Every entry matches a record made at training time, so the manifest describes the trained state.**
   - 332 entries have a local copy in `~/a5_sessions` or `~/a5_recovery`. All 332 re-hash identically on the laptop.
   - The 17 entries without a local copy are the 16 weight files and GRPO-A's `state.pt`. Each equals its training-time record:
     - for weights, the run's DONE.json lineage `weights_sha256`;
     - for `state.pt`, the `source_state_sha256` that both recovery branches recorded when they started from step 80.
   - Every entry in SESSION_A, SESSION_B, RECOVERY and BEFORE_RECOVERY (273 to 151 entries each) is present with an identical hash. None is missing.
5. **The failed run is preserved.** `grpo/A_base_s2026` is listed (21 files) and still has no DONE.json.

**What this establishes.** The server's run files matched BEFORE_FINAL after the evaluation. BEFORE_FINAL matches hashes recorded during training. So the trained runs, including every evaluated checkpoint and the failed GRPO-A, were in their training-time state after the final evaluation. This conclusion does not depend on exactly when BEFORE_FINAL was written; its file time on the laptop is the download time.

**Residual (stated, not a gap in the evidence chain).**
- The sha256sum check itself ran on the server and cannot be re-run from the laptop. Its pass is attested by the recovered log, as the plan specified.
- The BASE weights sit outside the runs tree and are not in BEFORE_FINAL, by design. They are covered by the pre-run `check` and by every summary.json recording the locked hash f7aa8d66….

`AUDIT_GENERATED.md` and `audit.json` were produced before this evidence arrived. They still list three unverifiable items and are kept unchanged as the record of that run; this section supersedes their first item.

## Reproduce

```bash
# WSL, from the project root
V=<repo>/.venv/bin/python
$V scripts/research_analysis.py --final ~/a5_final --runs ~/a5_sessions --runs ~/a5_recovery --out results/research_analysis_rerun
PYTHONPATH=~/a5_plotlib $V scripts/research_figures.py --data results/research_analysis_rerun
$V scripts/verify_post_final_integrity.py --final ~/a5_final --runs ~/a5_sessions --runs ~/a5_recovery --out results/research_analysis_rerun/post_final_integrity
```
