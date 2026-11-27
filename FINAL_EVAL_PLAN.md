# Final evaluation preparation (2026-11-26)

Development sets only. No held-out (final) file has been opened, and no checkpoint is chosen by any score. The held-out checkpoint list is frozen in `final/freeze_v1.yaml` (section 5). It follows the registry alone: the final step of each pre-registered run, plus both post-hoc branches.

## 0. Readiness: GO for deployment and server checks; final evaluation conditional on them

**Unresolved audit issues: 0.** The 16 missing-artifact issues from the first audit are resolved by the recovered Session A/B tree. 2 items are explained, not problems (section 1).

| Item | State |
|---|---|
| 14 Session A/B runs, the failed original, A1 and A2 | All 17 present with the expected outcome. Configuration hashes, verifier 2.0.0, dataset identity, checkpoint lineage and dev item files are consistent |
| Manifests | All local files match their manifests. `BEFORE_RECOVERY.sha256`: 258 of 273 verified locally; the other 15 are weights and optimizer state, never downloaded. `SESSION_A.sha256`: 151 of 151. `SESSION_B.sha256`: 107 of 107. `RECOVERY.sha256`: 74 of 74 |
| GRPO-B's parent | Equals the registered `SFT-reverified_n512_s0` (full sha256 f57b1af6…; lineage base > sft > grpo) |
| Learning rates | Both main-run rates re-derived from the tuning runs with the pre-registered rule: SFT 1e-5 and GRPO 1e-5, as recorded |
| Freeze file | Unchanged: sha256 ff0f0ce5… in the working tree, `a5_server_v6.tar.gz` and `a5_update_v6.tar.gz`. Last written 15:49 PST, before the Session A/B manifests were created (16:29 PST) |
| Held-out isolation | No recovered artifact names a held-out set. Every run evaluated only `gsm8k_dev` and `math_dev`. No `final_eval` folder or lock exists |
| V6 | Built against the V5 server package. The update's base hash equals `a5_server_v5.tar.gz`. Rehearsed V2 → V3 → V5 → V6 (section 6) |

**GO/NO-GO:**
- **GO** for deploying V6 and running the server checks (section 6, steps 2-3).
- The final GPU evaluation (step 4) is GO only if `server_final_eval.sh check` prints one `ok` line per checkpoint, which confirms the 11 weight files against the hashes audited here.
- It also needs `A5_AUTHORIZED=yes` and `A5_FINAL_EVAL_APPROVED=yes`.
- If any check fails: NO-GO. Do not evaluate; send the output.

**Remaining server-side requirements** (cannot be done on this machine):
1. sha256 of the 11 frozen weight files: `check` hashes each and compares it with DONE.json. Here they are verified only as metadata (DONE.json = checkpoint/lineage.json = `BEFORE_RECOVERY.sha256`).
2. The base model's weights against `PT_BASE_SHA256` (f7aa8d66…), also done by `check`.
3. `verify_upload.py` after applying V6 (196 files).
4. `BEFORE_FINAL.sha256` of the trained runs before evaluating, and the unchanged check after.

## 1. Artifact audit (`scripts/final_audit.py`, `results/final_prep/audit_full/AUDIT.md`)

**Every run is checked for:**
- outcome against expectation (DONE.json, FAILED.json or the incident record);
- run.json config hash;
- recorded configuration against the registered config file;
- verifier version and hashes;
- dataset identity: the data file's sha256 against `data/stage0/MANIFEST.json` and the local file; SFT example IDs against the subset file and their recorded hash;
- checkpoint metadata: `checkpoint/lineage.json` against DONE.json;
- parent weights;
- metric rows: duplicates, gaps, last step against DONE.json;
- dev item files against DONE.json;
- every manifest, file by file;
- agreement between manifests.

| Group | Runs | Result |
|---|---|---|
| SFT tuning (seed 100, reverified n512) | lr 1e-5, 2e-5, 5e-5 | done; 32 steps; all checks pass |
| SFT main (seed 0, lr 1e-5) | exact_full, reverified_full, sizecontrol, exact_n512, reverified_n512, n128, n2048 | done; all checks pass. reverified_full: 25 duplicated rows, explained below |
| GRPO tuning (from base, 30 steps) | lr 2e-6, 5e-6, 1e-5 | done; all checks pass |
| GRPO-B_sft512_s2026 | parent = reverified_n512 | done; 100 steps; 7.35 GPU-h; all checks pass |
| GRPO-A_base_s2026 | pre-registered | **failed** at step 98 (incident record); 97 rows; both copies (sessions and recovery trees) identical; its step-80 `state.pt` in `BEFORE_RECOVERY.sha256` equals the incident backup (73114307…) |
| A1, A2 | post-hoc | done; 100 steps; all checks pass |

**Explained items (not problems):**
1. **SFT-reverified_full_s0, 25 duplicated metric rows.**
   - The run was interrupted, then resumed from its step-150 save. Steps 151-175 were logged twice; this predates the V5 rewind fix.
   - The first replayed step is identical (loss 0.258388 both times). Later losses differ by at most 7×10⁻⁴, from GPU nondeterminism.
   - Reports keep the last row per step, which is the trajectory that produced the checkpoint.
   - The resume is recorded in `resumes.jsonl` with the same code hash.
2. **15 manifest entries not downloaded.** These are the 14 `model.safetensors` files and the original's `state.pt`, excluded by design. Each recorded weight hash equals the run's DONE.json and `checkpoint/lineage.json`.

**Manifest provenance.**
- `SESSION_A.sha256` and `SESSION_B.sha256` were generated **retrospectively** for transfer verification ("Historical session manifests were missing", created 2026-11-26 16:29 PST). They prove the transfer, not the state during the sessions.
- That state is evidenced by `BEFORE_RECOVERY.sha256`, written on the server before the recovery branches ran. All 258 non-weight files of both session manifests appear in it with identical hashes, and it lists no non-weight file that the session manifests miss.
- The recovery tree is verified by `RECOVERY.sha256`, written by the V5 runbook.

## 2. Development results (`results/final_prep/dev_summary/`)

Strict accuracy (frozen verifier 2.0.0) with Wilson 95% CI. GSM8K dev n = 1,000 (from GSM8K train); MATH dev n = 500 (never in a training pool). **One training seed per arm.** Labels: *registered (dev)* means a pre-registered final comparison computed on dev data, which is not the confirmatory test. *Exploratory* and *post-hoc* are as in section 5.

| Run | GSM8K strict (95% CI) | relaxed | MATH strict (95% CI) | relaxed | training GPU-h |
|---|---|---|---|---|---|
| Base model, r1_zero (GRPO-A step 0) | 21.8 (19.4-24.5) | 52.3 | 7.2 (5.2-9.8) | 27.4 | 0 |
| SFT tune lr 1e-5 / 2e-5 / 5e-5 (seed 100) | 86.1 / 85.0 / 81.8 | 87.4 / 85.6 / 82.4 | 33.4 / 27.8 / 30.2 | | 0.37 / 0.44 / 0.62 |
| SFT reverified n128 / n512 / n2048 / full 4,423 | 68.0 / 82.0 / 84.2 / 81.9 | 80.5 / 83.9 / 84.7 / 82.9 | 23.6 / 30.2 / 32.6 / 29.6 | | 0.12 / 0.44 / 1.03 / 2.69 |
| SFT exact n512 / full 3,420 | 88.9 / 88.8 | 89.1 / 88.8 | 32.4 / 31.6 | | 0.44 / 1.83 |
| SFT reverified size control 3,420 | 83.6 | 84.0 | 30.2 | | 1.57 |
| GRPO tune lr 2e-6 / 5e-6 / 1e-5 (30 steps) | 67.6 / 84.6 / 87.5 | 75.2 / 87.5 / 87.8 | 17.4 / 29.6 / 27.8 | | 1.95 / 1.83 / 1.93 |
| GRPO-B (sequential), step 100 | **92.8** (91.0-94.2) | 92.8 | **47.8** (43.5-52.2) | 50.2 | 7.35 (+0.44 parent) |
| GRPO-A (direct), step 75; failed run, diagnostic | 92.9 (91.1-94.3) | 92.9 | 39.8 (35.6-44.2) | 41.0 | 4.76 to step 75 |
| A1 / A2, step 100 (post-hoc) | 92.7 / 91.9 | 92.7 / 91.9 | 41.6 / 41.2 | | 2.28 / 2.51 (+5.09 to step 80) |

The base model's question-only dev score was never measured (`EVAL-base-dev` not run). The final evaluation runs both prompts.

### 2a. Registered comparisons on dev data (exact McNemar, Holm m = 4 per set; C4 unavailable, counted as p = 1)

| Comparison (b minus a) | GSM8K diff pp (95% CI), Holm p | MATH diff pp (95% CI), Holm p |
|---|---|---|
| C1 data quality, full: exact → re-verified | **−6.9** [−9.3, −4.5], 3e-8 | −2.0 [−6.0, +2.0], 0.87 |
| C2 data quality, n512: exact → re-verified | **−6.9** [−9.2, −4.6], 3e-8 | −2.2 [−5.9, +1.5], 0.87 |
| C3 pathway B vs its SFT parent | **+10.8** [+8.3, +13.3], 2e-16 (relaxed +8.9) | **+17.6** [+13.1, +22.1], 9e-13 (relaxed +18.2) |
| C4 pathway A vs base | unavailable: run failed | unavailable |

### 2b. What the dev data say (exploratory unless stated; Holm within the 13 exploratory contrasts per set)

- **Training-seed variance is large, and it qualifies every single-seed contrast.**
  - `SFT-tune_lr1e-5` and `SFT-reverified_n512_s0` have identical data, learning rate and epochs; only the training seed differs (100 vs 0).
  - They differ by **+4.1 pp on GSM8K** [+2.1, +6.1] (Holm p 0.0009) and +3.2 pp on MATH (n.s.).
  - McNemar tests cover item sampling only. A 4 pp seed effect is more than half of the C1/C2 gap, so the dev data do not establish that re-verified data are worse than exact-match data. They establish that this seed-0 re-verified run is worse.
  - The held-out test inherits the same limitation; the frozen plan and EXPERIMENT_PLAN section 3 already state it.
- **Data source.**
  - Exact-match data beat re-verified data on GSM8K in both sizes and in the size control: exact_full vs sizecontrol −5.2 pp (Holm 4e-5).
  - Volume made no difference: sizecontrol vs reverified_full −1.7 (n.s.).
  - On MATH, all source contrasts are n.s. (−0.6 to −2.2).
  - The n512 subsets share 387 of 512 examples; `exact_full` is a subset of `reverified_full`.
- **Data size.** 128 → 512 gives +14.0 pp GSM8K (Holm 8e-15), mostly format: relaxed gives only +3.4. 512 → 2,048 gives +2.2 (n.s.), and 2,048 → 4,423 gives −2.3 (n.s.). Beyond 512 examples, more SFT data bought nothing detectable for 2.3 extra GPU-hours.
- **SFT learning rate.** 1e-5 ≥ 2e-5 (−1.1, n.s.) > 5e-5 (−4.3, Holm 0.002). Training loss falls further at higher rates (0.345 / 0.238 / 0.144) while dev accuracy falls, consistent with overfitting 512 traces.
- **GRPO learning rate.** 1e-5 > 5e-6 (−2.9, n.s. after Holm) >> 2e-6 (−19.9, Holm 8e-34). At 2e-6, 30 steps leave 83% → 40% zero-advantage groups and reward 0.25.
- **Direct vs sequential GRPO.**
  - On GSM8K they are equal at matched GRPO steps: A step 75 vs B step 75, +0.2 pp. In the post-hoc comparison of A1 and B at step 100: +0.1.
  - On MATH, B is ahead: +4.6 at step 75 (p 0.04, Holm 0.29), and +6.2 / +6.6 against A1 / A2 at step 100 (post-hoc, Holm 0.006).
  - The SFT parent adds 0.44 GPU-h. Exploratory and post-hoc; one seed.
- **Evaluation is deterministic.** The SFT parent's final evaluation and GRPO-B's step-0 evaluation of the same weights agree on every item (0 of 1,500 discordant), as do A1/A2 at step 80.

### 2c. Diminishing returns to GRPO compute, and training dynamics (`fig2_grpo.png`, `fig3_grpo_dynamics.png`)

| Segment | GPU-h | GSM8K pp (Holm p, m = 9) | MATH pp (Holm p) |
|---|---|---|---|
| B 0 → 25 | 1.65 | **+10.7** (7e-17) | **+15.6** (6e-12) |
| B 25 → 100 | 5.14 | +0.1 (1) | +2.0 (1) |
| A 0 → 25 | 1.45 | **+67.7** | **+23.6** |
| A 25 → 50 | 1.65 | **+2.9** (0.02) | **+8.4** (0.0003) |
| A 50 → 75 | 1.66 | +0.5 (1) | +0.6 (1) |

- **Most of the gain comes in the first 25 steps.**
  - **B:** 25 of 100 steps, 22% of its GRPO compute, give all the detectable GSM8K gain and most of the MATH gain. The remaining 5.1 GPU-h are not detectable on GSM8K. MATH 50 → 100 shows +4.8 (unadjusted p 0.02, Holm 0.12).
  - **A:** gains stop after step 50.
- **Zero-advantage groups** (all 8 rollouts equally rewarded) per 25-step window:
  - B: 0.54 → 0.70 → 0.72 → 0.65. A: 0.48 → 0.62 → 0.66 → 0.69.
  - From step 26 on, about two thirds of every batch carries no gradient. The registered trigger for zero-advantage filtering (informative rate < 0.6, Stage 6) is met in both pathways. That intervention is Tier B/C and is not part of this evaluation.
- **Entropy.**
  - B: 0.29 → 0.06 → 0.056, then rising to 0.11 over steps 76-100.
  - A: 0.29 → 0.046 → 0.044. The collapse at step 97 is separate.
- **Response length.**
  - B grows steadily: 373 → 413 → 432 → 472 tokens.
  - In B's last 10 steps, incorrect answers average 710 tokens against 384 for correct ones, and truncation is 6.2%.
  - A is flat at about 400 tokens. The GRPO tuning runs shorten responses to 200-260 tokens in 30 steps.
- **Numerical consistency.**
  - 0 mismatched parameters at every exact weight check in every run.
  - Mean rollout log-prob gap: every run starts near 0.010 and settles at 0.003-0.005 once entropy falls. The step maximum is 0.0127 for B and at most 0.0125 for the tuning runs. For A it is 0.0388, at step 97 only; the branches stay at most 0.0042.
  - p99 gap: B at most 0.138. Only the failed original approached the 0.05 gate.

## 3. Recovery trajectories, A1 and A2 (`results/final_prep/recovery_trajectories/`)

**Confirmed (direct measurement):**
- **The restore was exact.** At step 81, A1 and the original have identical reward (0.6445), mean length (465.25 tokens), rollout gap (0.003309) and entropy. Gradient norms differ by about 1 in 10⁵, from non-deterministic GPU reductions. Trajectories diverge from step 82.
- **Both branches evaluated the restored state with identical answers** (0 discordant items on both sets), so evaluation on fixed weights is deterministic.
- **The failure did not recur in the replay.** A1 used the same state, learning rate and data order and ran to step 100. Its step-97 reward was 0.703 (original 0.055), with no early warning, no step below 0.30, mean gap at most 0.0042, and p99 at most 0.123. A2 is the same: gap at most 0.0040, p99 at most 0.117.
- **Update size follows the learning rate.** A2/A1 means over steps 81-100: update RMS 0.477, update L2 0.477, parameter change 0.479, gradient norm 0.90. The Adam update scales with the learning rate.
- **Training metrics are indistinguishable**, steps 81-100, A1 vs A2:
  - reward 0.712 vs 0.724 (sd 0.06);
  - entropy 0.030 vs 0.032;
  - informative groups 0.30 vs 0.30;
  - format 0.972 vs 0.965;
  - truncation 2.7% vs 3.4%;
  - length 403 vs 417 tokens.
- **Dev accuracy did not change from step 80 to 100 in either branch**, and the branches do not differ. All exact McNemar tests, Holm within set:

| Comparison (b minus a) | Set | Diff pp (95% CI) | p | Holm p |
|---|---|---|---|---|
| A1 100 vs step 80 | GSM8K | +0.9 [-0.5, +2.3] | 0.272 | 0.815 |
| A2 100 vs step 80 | GSM8K | +0.1 [-1.2, +1.4] | 1 | 1 |
| A2 vs A1, step 100 | GSM8K | -0.8 [-2.2, +0.6] | 0.332 | 0.815 |
| A1 100 vs step 80 | MATH | -0.4 [-4.7, +3.9] | 0.927 | 1 |
| A2 100 vs step 80 | MATH | -0.8 [-4.7, +3.1] | 0.762 | 1 |
| A2 vs A1, step 100 | MATH | -0.4 [-4.4, +3.6] | 0.923 | 1 |

- **A2 drifted to a doubled `</think>` tag.** Saved rollouts with the tag rose from 2 of 256 at step 81 to 220 of 256 at step 100. At step 100 it appears in 944 of 1,000 GSM8K and 404 of 500 MATH dev answers. A1 rose to 69 of 256 at step 89, then fell to about 0 (0 at step 100). The original showed it at step 75 (132 of 1,000) and not at step 80 (2).
  - The verifier's format check accepts the tag, so the reward does not see it. Accuracy among tagged answers (91.6%) matches the rest.
  - A2's MATH dev answers are longer (607 vs 536 tokens), with more truncation (64 vs 35) and a lower format rate (87.2 vs 93.0), but strict MATH accuracy is unchanged.

**Hypotheses (not tested):**
- The original's collapse at steps 96-97 was trajectory-specific, not a deterministic consequence of the step-80 state with lr 1e-5. One replay that did not fail supports this but cannot estimate how often failure happens.
- The doubled tag drifts freely because nothing rewards or penalizes it. Its direction in A2 vs A1 is not attributable to the learning rate: one run each, and A1 drifted too before reverting.

## 4. The plateau after step 50 (`fig4_dev_vs_compute.png`)

| Segment | Training GPU-h | GSM8K diff pp (Holm p) | MATH diff pp (Holm p) |
|---|---|---|---|
| step 0 → 25 | 1.45 | +67.7 (<1e-100) | +23.6 (<1e-20) |
| 25 → 50 | 1.65 | +2.9 (0.020) | +8.4 (0.0003) |
| 50 → 75 | 1.66 | +0.5 (1) | +0.6 (1) |
| restored 80 → A1 100 | 1.70 | +0.9 (1) | -0.4 (1) |
| restored 80 → A2 100 | 1.92 | +0.1 (1) | -0.8 (1) |
| 50 → A1 100, over the whole tail | 3.69 | +0.3 (1) | +2.4 (1) |

Holm is applied within each set over the nine exploratory steps in `REPORT_DATA.md`.

**Confirmed:**
- After step 50 (3.1 training GPU-hours), no dev change is detectable on either set over a further 3.7 GPU-hours, which is more than half of the total training compute.
- GSM8K gains per GPU-hour: 47 pp, then 1.8 pp, then 0.3 pp or less.

**Measured correlates of the plateau (associations, not causes):**
- **Less learning signal.** The share of groups where all 8 answers were correct rose from 15% (steps 1-25) to 47% (26-50) and 54% (51-95; 55-56% in both branches). Informative groups fell from 0.52 to 0.38, then 0.33.
- **Narrower sampling.** Mean entropy fell from 0.29 to 0.046, then 0.041 nats. Training reward was flat from step 51 to 95 (-0.004 per 10 steps, naive SE 0.007).
- **GSM8K near its ceiling for this model.** Across the six checkpoints from step 25 on, 815 of 1,000 items are always right and 29 always wrong. Relaxed accuracy equals strict from step 50 on, so the remaining errors are wrong answers, not formatting.
- **Checkpoint-to-checkpoint noise is as large as the tail's gains.** Between step 75 and the restored step 80 (5 updates), 43 GSM8K and 77 MATH items change answer. Differences of about 1 pp between nearby checkpoints are within that churn.

**Hypothesis:** with one update per batch and a group-mean baseline, the shrinking share of informative groups limits further gains. Testing it would need the registered difficulty-aware or filtering interventions (Tier C), not more steps.

## 5. Frozen held-out plan (`final/freeze_v1.yaml`)

**Rule:**
- Evaluate the final-step `checkpoint/model.safetensors` of each listed run, whose sha256 must equal its DONE.json lineage.
- Each checkpoint is evaluated once, with greedy decoding, 1,024 tokens, vLLM, prefix caching off, context 2,048, seed 2026 and one GPU type.
- Sets: GSM8K test (1,319, primary) and MATH val final (5,000, secondary).

| Checkpoint | Class | Prompt |
|---|---|---|
| Base model | pre-registered | r1_zero and question_only |
| SFT-exact_full_s0, SFT-reverified_full_s0, SFT-reverified_sizecontrol_s0, SFT-exact_n512_s0, SFT-reverified_n512_s0, SFT-reverified_n128_s0, SFT-reverified_n2048_s0 | pre-registered | r1_zero |
| GRPO-B_sft512_s2026 | pre-registered | r1_zero |
| A1 (step 100), A2 (step 100) | **post-hoc**, both evaluated, neither selected | r1_zero |
| GRPO-A_base_s2026 | **not evaluated**: failed, no final step, nothing substituted | - |
| Six tuning runs | not evaluated: tuning only | - |

**Comparison families** (paired exact McNemar, paired difference with 95% CI, Holm within each family and set):

| Family | Label | Comparisons |
|---|---|---|
| confirmatory, m = 4 | pre-registered confirmatory | C1 full re-verified vs exact; C2 n512 re-verified vs exact; C3 GRPO-B vs its SFT parent; **C4 pathway A vs base: unavailable (failed), counted with p = 1**, so the family is not shrunk after the failure |
| sft_vs_base, m = 7 | pre-registered secondary | each SFT arm vs the base model's question-only prompt |
| descriptive | descriptive (no claim) | composition, volume, the sample-efficiency steps, GRPO-B vs base (both prompts), base prompts |
| posthoc, m = 4 | post-hoc exploratory | A1 and A2 vs base r1_zero, A1 vs base question-only, A2 vs A1 |

**Evidence categories used in every report:**
- **Pre-registered result:** a checkpoint and comparison fixed before training (registry; plan sections 3 and 10).
- **Statistically supported:** Holm p < 0.05 within a pre-registered family. On GSM8K the difference must also be at least 1.4 pp, the frozen detectable size; anything smaller is reported as "not detected" whatever its p.
- **Post-hoc diagnostic:** A1/A2 and everything in sections 3 and 4. Descriptive, never pooled with or substituted for a pre-registered result.
- **Exploratory hypothesis:** an explanation not tested by a designed contrast. The "Hypotheses" paragraphs above are examples.

**Reported per checkpoint and set** (`scripts/final_report.py`):
- strict accuracy with Wilson CI, second-verifier and relaxed accuracy;
- format and clean-format rates, doubled `</think>`;
- tokens (mean, median, p90; correct vs incorrect), truncation, failure categories, verifier errors;
- training GPU-hours and evaluation throughput.

The report refuses a lock written for a different freeze file, and flags any evaluation whose weights differ from the lock.

**Cost:** 12 evaluation passes (the base model twice). From the dev timings (MATH 500 in 75-86 s, GSM8K 1,000 in 63-73 s), each pass takes about 12-17 minutes. With two GPUs in parallel that is about 1.5-2 hours, about 3.5 GPU-hours.

## 6. Commands

Replace `0,1` with your allocated GPUs.

**1. Recovery and audit: done.** Sessions A/B are in `~/a5_sessions` (RECOVERED_OK), and the audit (`results/final_prep/audit_full/AUDIT.md`) reports 0 unresolved problems. To reproduce it (WSL; a new `--out` folder each time):
```bash
cd <repo>
<repo>/.venv/bin/python scripts/final_audit.py --runs ~/a5_sessions --runs ~/a5_recovery --out results/final_prep/audit_recheck   # "Unresolved problems: 0"
```

**2. WSL: send the update**
```bash
cd <repo>/dist && sha256sum -c a5_update_v6.tar.gz.sha256
scp a5_update_v6.tar.gz a5_update_v6.tar.gz.sha256 <host>:/srv/research/posttrain-v2/
```

**3. SERVER (`ssh <host>`): apply it, record checksums, run checks (start nothing)**
```bash
tmux new -s a5-final
export A5_AUTHORIZED=yes A5_GPUS=0,1 A5_ROOT=/srv/research/posttrain-v2
cd $A5_ROOT && sha256sum -c a5_update_v6.tar.gz.sha256 && mkdir a5_before_v6 && tar tzf a5_update_v6.tar.gz | while read -r f; do [ ! -f "$f" ] || cp --parents -p "$f" a5_before_v6/; done && tar xzf a5_update_v6.tar.gz
cd $A5_ROOT/a5 && source scripts/server_env.sh
cd $PT_RUNS && find sft grpo recovery -type f -exec sha256sum {} + > $A5_ROOT/BEFORE_FINAL.sha256; cd $A5_ROOT/a5
python3 scripts/verify_upload.py                 # 0 missing/changed
bash scripts/server_preflight.sh                 # PREFLIGHT_OK
bash scripts/server_final_eval.sh plan           # 12 checkpoints; GRPO-A_base_s2026 and tuning runs "not evaluated"
bash scripts/server_final_eval.sh ready          # READY (final)
bash scripts/server_final_eval.sh check          # one "ok" line per checkpoint; CHECK_OK (final)
```
The 16-character hashes printed by `check` must equal the hashes audited here. Any difference is NO-GO.

| Checkpoint | Expected | Checkpoint | Expected |
|---|---|---|---|
| BASE | f7aa8d665ab44d3b | SFT-reverified_n128_s0 | 622f1201d8ae8f7b |
| SFT-exact_full_s0 | 02c6ba5396a4d113 | SFT-reverified_n2048_s0 | 65f61e4c0f5e830b |
| SFT-reverified_full_s0 | 02ecf90239277d41 | GRPO-B_sft512_s2026 | 83dd404abf349f3a |
| SFT-reverified_sizecontrol_s0 | 765bcbcbc2daa37d | GRPO-A-replay80-lr1e-5-s2026 | bf3453c6c71da3a3 |
| SFT-exact_n512_s0 | c4f4b4e94efa551a | GRPO-A-recovery80-lr5e-6-s2026 | a4055bd25156cf73 |
| SFT-reverified_n512_s0 | f57b1af68b655397 | | |

**4. SERVER: final evaluation (only if every step-3 check passed, and you approve)**
```bash
export A5_FINAL_EVAL_APPROVED=yes
set -o pipefail
bash scripts/server_final_eval.sh run 2>&1 | tee -a $A5_ROOT/final_eval.log      # lock first, then two GPU workers
bash scripts/server_final_eval.sh status
cd $PT_RUNS && sha256sum -c --quiet $A5_ROOT/BEFORE_FINAL.sha256 && echo TRAINED_RUNS_UNCHANGED
find final_eval -type f -exec sha256sum {} + > FINAL_EVAL.sha256; cd $A5_ROOT/a5; unset A5_FINAL_EVAL_APPROVED
```

**5. WSL: recover and report**
```bash
mkdir -p ~/a5_final
rsync -a <host>:/srv/research/posttrain-v2/runs/final_eval/ ~/a5_final/
rsync -a <host>:/srv/research/posttrain-v2/runs/FINAL_EVAL.sha256 <host>:/srv/research/posttrain-v2/final_eval.log ~/a5_final/
(cd ~/a5_final && sed 's#  final_eval/#  #' FINAL_EVAL.sha256 | sha256sum -c --quiet && echo RECOVERED_OK)
cd <repo>
<repo>/.venv/bin/python scripts/final_report.py --final ~/a5_final --runs ~/a5_sessions --runs ~/a5_recovery --out results/final_eval
```

| Message | Action |
|---|---|
| `REFUSED: ...` or `no DONE.json` / `FAILED.json present` / `weights differ` in `check` | Do not evaluate. Send the output |
| `exists and differs ... refusing to replace a lock` | Never delete the lock. Send it |
| `STOPPED (final evaluation)` after an interruption | Run the same `run` command; finished checkpoints are skipped, and an interrupted one restarts from scratch on the same weights |
| Out of memory | Do not retry. Send the log |
