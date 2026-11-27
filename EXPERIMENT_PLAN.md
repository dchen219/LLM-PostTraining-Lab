# Experiment Plan

Status: Stage 0 complete locally. **No model has been trained.** Every training result below is planned, not achieved.
Single source for runs: `EXPERIMENT_REGISTRY.csv` (62 entries), generated with the configs by `scripts/make_configs.py`.

## 0. Authorship, stated plainly

- `cs336_alignment/*.py` (except small edits) is the public `datawhalechina/diy-llm` tutorial; the grader is sail-sg's (Apache-2.0).
- `posttrain/`, `configs/`, `scripts/`, `tests/` are new and were written by Claude at your request. They are not the tutorial's code, but you should be able to explain, modify and defend every part before claiming the work, and follow any disclosure rules that apply to you.
- The data is `garg-aayush/sft-cs336-assign5-datasets` (gpt-oss-120b traces on math12k).

## 1. What Stage 0 established (evidence in `results/stage0/`, `data/stage0/`)

| Finding | Evidence | Consequence |
|---|---|---|
| Exact-string filter rejected 1,340 traces; 1,049 (78.3%) pass the strict verifier, giving 4,545 (+30.0%) | `verifier_audit.json`, `recovery_audit.jsonl` | New data arm |
| The 1,049 were confirmed: 781 by an independent canonicalizer, 148 as exact rationals, 80 by math-verify, 40 read by Claude (equivalent, some with units or a degree sign dropped); 0 numeric conflicts; identical ID hash under two library stacks | same | Recovery is sound; human spot-check still advised |
| Verifier false accepts: 3 of 13,017 mutations flagged, of which 2 are real (`\sqrt[47]{8}` vs `\sqrt[4]{8}`, `21x\sqrt{10x}` vs `10x\sqrt{21x}`); 1 of 19,857 random pairs, and that one is a legitimate match (`2` vs `\$2`) | same | Low false-positive risk on this answer distribution; conflicting units are ignored (known gap) |
| Verifier false rejects: 118 of the 291 still-rejected traces are equal by the independent check | same | RL reward has false negatives; kept strict, not loosened |
| Non-strict ("full") mode accepts more wrong answers (missing base, 0.5 vs 50%, duplicate list) | same | Do not use it as a reward |
| Base accuracy, full GSM8K test, same weights: R1-Zero 17.89% (236/1,319), question-only 65.28% (861/1,319) | `base_eval/` | Prompt elicitation must be a baseline; see section 2 |
| R1-Zero failures: 433 formatted but wrong, 163 right by relaxed extraction but misformatted, 487 misformatted and wrong | `failure_decomposition.json` | Relaxed accuracy 30.25%; formatting alone does not explain the gap to 65.28% |
| In pure bf16, an Adam-sized step leaves 91.3% of weights unchanged at lr 1e-5 (82.8% at 2e-5) | `bf16_update_check.json` | fp32 master weights + bf16 autocast. One-step bound; real Adam steps vary in size |
| GSM8K train/eval overlap with SFT and RL problems: 0 exact, 0 shared 13-grams, 0 fuzzy | `contamination.json` | GSM8K is a clean final set with respect to our data |
| MATH val overlaps training: 6 exact and 119 fuzzy SFT problems | same | 120 SFT and 162 RL problems excluded from training |
| 7 normalized-duplicate SFT problems (2 true duplicates, 5 differ only by sign or brackets) | `MANIFEST.json` | All 7 conservatively excluded from subsets |
| At n = 512, 479-486 raw samples are already verified-correct | `MANIFEST.json` | Raw vs re-verified at 512 has little contrast; see section 4 |

The base model may have seen GSM8K or MATH during pretraining. That cannot be ruled out here.

## 2. Baselines and what counts as improvement

Report both base prompts. A gain over 17.89% mostly measures learning the output protocol. **Claims about better reasoning must beat the question-only baseline (65.28% on the full test) or be made against an SFT parent under the same prompt.**
The local numbers use Hugging Face greedy decoding. Cloud runs use vLLM, so the base model is re-evaluated there and only same-engine results are compared.

## 3. Evaluation protocol

| Set | n | Use |
|---|---|---|
| `eval_gsm8k_dev_1000` (from GSM8K *train*) | 1,000 | All tuning and model selection |
| `eval_math_dev_500` (untraced math12k problems) | 500 | Secondary dev |
| `eval_gsm8k_test_final` | 1,319 | Locked; once per pre-registered checkpoint |
| `eval_math_val_final` | 5,000 | Locked; secondary |

- GSM8K test has already been used: for the base evaluation here, and for hyperparameter selection by the tutorial and any earlier runs of yours. No hyperparameter is inherited from those results.
- Report the **final step**, never the best checkpoint.
- Greedy decoding, 1,024 tokens, strict verifier. Primary metric: GSM8K accuracy. Secondary: MATH, format, length, truncation.
- Pre-registered final comparisons: (1) reverified_full vs exact_full, (2) reverified_n512 vs exact_n512, (3) pathway B vs its SFT parent, (4) pathway A vs base. Paired exact McNemar with Holm correction, plus seed mean and SD. McNemar covers item sampling only, not training-seed variance.
- Write the chosen hyperparameters and checkpoint list into this file before setting `ALLOW_FINAL_EVAL=1`.

## 4. Experiments by tier

**Mandatory (24):** base evals (local done, dev, final); sync, SFT and GRPO pilots; SFT lr tuning (3); matched 512 raw/exact/reverified x 3 seeds (9); full exact/reverified/size-control x 1 seed (3); one EI run; one GRPO run per pathway.
**Recommended (25):** full-data seeds 2 and 3; sample-efficiency curve 128/256/1,024/2,048 x 3 seeds; GRPO lr tuning; GRPO seeds 2 and 3.
**Optional (13):** exact-match curve ends; gated interventions (section 6).

Design notes
- Subsets are nested and share one ordering across arms, so arms differ only where eligibility differs. Data seeds and training seeds move together.
- Full-data arms use 3,420 exact and 4,423 re-verified (the 3,496 and 4,545 after removing near-duplicates of MATH val and duplicates). The size-control arm draws 3,420 from the re-verified pool, separating volume from composition.
- Fixed epochs means more data also gets more optimizer steps. `train.max_steps` supports a step-matched follow-up.
- **Expect small effects** for raw vs re-verified at 512 (about 5-6% of samples differ). A null result there is likely and should be reported. The better-powered contrast is full exact vs full re-verified vs size control, because the exact filter also skews toward simply formatted answers.
- Learning rates and epochs are UNTUNED starting points.

## 5. Stage 5 diagnostics (implemented and logged every GRPO step)

All-correct, all-incorrect and zero-advantage groups; informative rate; reward and within-group variance; gradient norm; importance ratio mean/min/max; approximate KL; clip fraction (outside range, and active); entropy; response length overall/correct/incorrect; truncation; GPU peak memory; rollout throughput; phase timings (sync, rollout, reward, old log-probs, update); wall-clock and cost. PyTorch peak memory per device is reported; on the vLLM GPU it may miss memory outside the torch allocator.

With one update per on-policy batch the ratio is exactly 1, so clipping cannot act (unit-tested). Clip metrics matter only for `epochs > 1` or `update_batch` smaller than the rollout batch.

## 6. Stage 6: decision rules fixed in advance

Choose at most two, only if the trigger is observed in pathway B seed 2026:

| Intervention | Trigger | Control | Success criterion |
|---|---|---|---|
| Zero-advantage filtering, `compute_only` | informative rate < 0.6 and update_s > 25% of step time | same run unfiltered | Lower update time; the gradient is identical by construction (unit-tested) |
| Difficulty-aware sampling | informative rate < 0.6 and prompts revisited | uniform sampling | Higher dev accuracy per GPU-hour, 3 seeds |
| Off-policy reuse (e2) | rollout_s > 50% of step time | `X_onpolicy_u4`, same updates/step | Equal accuracy at lower rollout cost |
| Length/std normalization | length drift or a grad-norm problem | lr-matched control | Pre-stated per run |

The tutorial's off-policy sweep changed updates per step (1/4/8/16) along with reuse; the control above avoids that. `renormalize` filtering changes the effective learning rate and needs its own lr control. Failure-driven data generation is design-only: it needs an error taxonomy from a finished run and is not implemented.

## 7. Limitations

- Single model size, single teacher, one verifier for both reward and evaluation (reward hacking of the verifier would not be visible in the metric; read samples).
- vLLM sampling is not bitwise reproducible across hardware. Resume is bit-exact on CPU for SFT; GPU and vLLM are not guaranteed.
- Resume restores the vLLM engine from the base weights and then syncs.

## 8. Amendments after the pilot (2026-11-24)

Evidence: `CLOUD_PILOT_REPORT.md`. Tiers, costs and the decision: `FULL_TRAINING_DECISION.md`. No hyperparameter was chosen from pilot accuracy.

- GRPO, SFT and EI configs now use `micro_batch: 4` (8 ran out of memory for GRPO and is predicted to exceed the 70 GiB gate for full-data SFT). `train.token_budget` exists and is off.
- GSM8K dev is about 17 points easier than GSM8K test for the base model (question-only). Use dev for relative choices only.
- Trained models are compared with the question-only base prompt, not only with R1-Zero.
- Weight sync has an exact parameter check in addition to the unchanged 0.05 log-prob tolerance. (Superseded by section 11.)
- Section 6 triggers must still be evaluated on pathway B; the pilot started from the base model and does not satisfy them.

## 9. Amendments after the server recovery (2026-11-25)

Evidence: `RESEARCH_SERVER_AUDIT.md`, `SERVER_MIGRATION.md`.

- Registry: 67 rows. Tier A (17) PLANNED, Tier B HOLD_B, Tier C DEFERRED, three HISTORICAL rows and two gated dev evaluations of the old checkpoints. No planned run is replaced.
- No hyperparameter, checkpoint or stopping decision may use the historical evaluation scores: they are GSM8K test scores.
- The historical dev evaluations are valid only if those checkpoints were not trained on GSM8K train, which is where the dev set comes from.
- On the server, `PT_HW=a6000` sets micro-batch 2 (plus a 3,300-token cap for GRPO and EI). Batch sizes, loss divisors, learning rates and step counts are unchanged.
- `configs/eval/final.yaml` must also run the question-only prompt before the final evaluation is used.

## 10. Final Tier A plan (2026-11-25)

Fifteen runs, no new experiments. Order and commands: `SERVER_MIGRATION.md` section 4.

| Session | Runs | Answers |
|---|---|---|
| 1. sft-tune | 3 | SFT learning rate |
| 2. sft-main | 7 | Data quality (exact vs re-verified, with size control), sample efficiency (128, 512, 2,048, full), RL parent |
| 3. grpo-tune | 3 | GRPO learning rate |
| 4. grpo-main | 2 | SFT to GRPO, and base to GRPO |

Moved to HOLD_B: `SFT-reverified_n256_s0`, `SFT-reverified_n1024_s0` (no pre-registered comparison uses them).

**Learning-rate rule, fixed before any run.** Take the final-step GSM8K dev accuracy (n = 1,000). Choose the highest; if another rate is within 1.0 pp of it, choose the smallest such rate. The same rule applies to SFT and GRPO. GRPO rates are tuned from the base model and used for both pathways.

**Measuring improvement.** Main contrasts: SFT arms against the base model's question-only prompt on the same engine; pathway B against its own SFT parent; pathway A against base. Paired exact McNemar with Holm correction, final step only.
On historical data, re-evaluating identical weights changed 39 of 1,319 answers, and the smallest difference detectable with 80% power was about 1.4 pp. A GRPO gain below that must be reported as not detected. All models in one comparison use the same engine settings (caching off, same context length, same GPU type).

Planned GRPO differs from the historical run in what mattered: fp32 master weights, 25,600 rollouts instead of 512 at the saved step, prefix caching off, exact sync checks, final step instead of best, dev instead of test.

## 11. Synchronization gates (approved by the user 2026-11-25)

| Check | Role | When | Failure |
|---|---|---|---|
| Every engine parameter equals the policy parameter converted to the engine dtype: same names, dtype, shape and bits; no unverified engine parameter; nothing compared counts as failure | **Primary gate** | Before every rollout batch and every evaluation (SFT, GRPO, EI) | Run stops |
| Mean absolute gap between the engine's sampling log-probs and the policy's, over every token of the actual rollouts, at most 0.05 (the pilot's gate, unchanged); every token must have a value | **Separate correctness gate** | Every GRPO step, before the update; cannot be disabled | Run stops before training on the batch |
| Original prompt log-prob probe, 0.05 | Diagnostic | Every 10 steps (every step in validation) | Logged as `sync_probe_over_tol`; never stops a run |

Replaces the 0.05 probe stop in sections 8 and 10. Reason: identical weights read 0.025 to 0.044 on the probe, and embeddings ten steps stale read 0.049.

## 12. Reward protocol and readiness (2026-11-25, before any Tier A run)

**Verifier 2.0.0, frozen.** The strict grader is unchanged. Two additive rules apply only to answers it rejects: a whole-string lead-in ("Answer is 2.") whose remainder the strict grader must accept, and an answer that is exactly n! (n at most 20) against an integer reference. Every run records the verifier's hashes and refuses to start if they differ from `posttrain/verifier_lock.py`.

- Evidence (`results/verifier_v2/crosscheck.json`): 8,832 recorded outputs, 6 newly accepted, all confirmed by exact rational arithmetic and math-verify; nothing previously accepted is rejected; no recorded dev score changes; no SFT label would change, and labels and subsets are not rebuilt.
- The reviewed response "Answer: $8!$ by dividing by 9" stays rejected. 8! = 40,320 (product, SymPy and log-gamma agree), so "$8!$" alone is accepted, but the response as written reads as 8!/9 = 4,480.
- Other unevaluated arithmetic and prose around an answer stay rejected, so that a policy is not paid for not computing or for hedging.

**Human annotation is no longer required** (user decision). The AI-assisted review (18 of 20, original gate of 19 not met) is kept in `results/review_ai_assisted/`.

**Readiness is automated and separate.** SFT: package unchanged, frozen verifier and golden cases (dev scores choose the learning rate), validation gates for sync, gradients and memory. GRPO adds: resume, exact check every step, no mismatch, rollout log-probs, the reward and sync regression tests, and all ten SFT runs finished.

**Known limit of strict scoring, reported with every result.** math-verify accepts 146 of the base model's 1,000 GSM8K dev answers (`r1_zero`) that strict scoring rejects, against 1 for the SFT pilot. Gains over base in strict accuracy are therefore partly answer formatting. Reports show strict (primary), second-verifier and relaxed accuracy; a gain absent from relaxed accuracy is not a mathematics gain.

**Clipping.** The registered GRPO runs make one update per rollout batch, so the ratio is exactly 1 and the clip fraction is 0 (section 5; observed at all 11 validation steps). The update then equals REINFORCE with a group-mean baseline up to scale. No result may be attributed to clipping.

**Code changes after the GPU validation** (code 4a28e40b): `verify.py`, new `verifier_lock.py`, `common.py` (records the verifier), `evaluate.py` and `train_grpo.py` (log which rule accepted). Scoring and logging only; losses, engines, synchronization, batching and optimizers are unchanged, so the GPU validation is not repeated.

**Parent for pathway B** stays the registered `SFT-reverified_n512_s0`, whatever its dev score. Final step only; the step 25, 50 and 75 evaluations are diagnostics.

## 13. GRPO-A_base_s2026 failed (2026-11-26)

97 updates completed; step 98 was rejected by the rollout log-probability gate before any update. It is reported as a failed pre-registered run and is never resumed or replaced (`recovery/incident_GRPO-A_base_s2026.json`). Tier A: 14 completed, 1 failed. Two post-hoc diagnostic branches start from its verified step-80 state in new folders: A1 replays it with lr 1e-5, and A2 uses lr 5e-6 from the first update after step 80. Every other setting is identical and every gate is unchanged. They are not registered experiments and are not used to select anything for the pre-registered comparisons. Diagnosis, mechanism and commands: `GRPO_A_INCIDENT.md`.

**Code changes** (training math unchanged): a rejected batch is saved before exiting and `FAILED.json` is written; a resume moves later log rows aside instead of duplicating them; the failed run's folder is refused; per-step update size from the AdamW state; more rollout-gap detail (tokens over 1 and 5 nats, worst token); `posttrain/fork_grpo.py`, `scripts/server_recovery.sh`, `scripts/recovery_report.py`.

**V5 (2026-11-26, approved).** Both branches save every step's rollouts (`grpo.save_rollouts_every: 1`, an allowlisted instrumentation change with no effect on updates or random numbers). Diagnostic criteria are registered before training in `recovery/diagnostic_criteria.yaml`: early warning, reward below 0.30, two consecutive steps below 0.30, the 0.05 gate, and completion at step 100. They are diagnostic only. The early-warning rule was derived retrospectively from the failed run and has not been validated.

## 14. Final evaluation frozen (2026-11-26, before any held-out result)

Both post-hoc branches finished at step 100 with every gate passed; their dev accuracy did not change from step 80 (`FINAL_EVAL_PLAN.md` sections 2-4). The held-out checkpoint list is frozen in `final/freeze_v1.yaml` by the registry alone: the final step of the base model (both prompts), the seven main SFT arms and GRPO-B, plus A1 and A2 labelled post-hoc. GRPO-A_base_s2026 failed and is not evaluated; its pre-registered comparison (pathway A vs base) is reported as unavailable and counted with p = 1 in the confirmatory family (m = 4). Tuning runs are not evaluated. Families, statistics and evidence labels: `FINAL_EVAL_PLAN.md` section 5. Evaluation runs only through `scripts/server_final_eval.sh`, which hashes every checkpoint into `FINAL_CHECKPOINTS.json` before the first evaluation and needs `A5_FINAL_EVAL_APPROVED=yes`.
