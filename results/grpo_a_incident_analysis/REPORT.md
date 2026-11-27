# GRPO-A_base_s2026: pre-recovery incident analysis

Source: `~/a5_recovery/grpo/A_base_s2026` (metrics.jsonl, run.json, data.json, eval/, rollouts/), read only. Produced by `scripts/incident_analysis.py`; all numbers are in `analysis.json`. Development sets only. One run: everything below describes this run and is not a rate or a general law.

## Conclusions

1. **Verification is not the cause.** The evidence rules this out as far as it reaches: no verifier errors in 97 steps, every saved reward reproduces, and format stayed at 87.5% at the collapse. The step-97 rollouts themselves were not saved.
2. **Optimization instability is supported.** A healthy run on a plateau, with entropy already collapsed to about 0.03, broke in a single update (update 96). That update had the largest gradient of the run.
3. **Numerical inconsistency between vLLM and PyTorch is the earliest warning, and a large consequence.**
   - **Early warning:** the rollout log-prob gap rose above its recent baseline at steps 94-95, at a level ten times below the gate.
   - **Consequence:** at step 97 it was 3.5-5 times what the same entropy produced elsewhere in the run.
   - **Open:** whether it is cause, early symptom, or both. The logs cannot separate these.

## 1. Integrity (confirmed)

| Check | Result |
|---|---|
| Rows | 101 = 97 training steps (1..97, contiguous) + 4 evaluations (steps 0, 25, 50, 75) |
| Duplicate step rows | **0**; no `resumes.jsonl`. The run never resumed, and the step-80 state was never loaded |
| `DONE.json` | absent |
| Exact weight check | ran at all 97 steps, 0 mismatches |
| Verifier errors (training) | 0 in 24,832 rollouts |
| Clipping | ratio exactly 1.0 at every step, clip fraction 0, as expected with one update per batch |
| Learning rate | 1e-5 after warm-up (step 5 on), constant |

## 2. Trajectory (fig1_trajectory.png)

| Phase | Steps | What happened |
|---|---|---|
| Format learning | 1-20 | Reward 0.02 → 0.62, format rate 0.10 → 1.00, entropy 0.54 → 0.11 |
| Plateau | 20-95 | Reward 0.55-0.86, format 0.88-1.00. Entropy fell to 0.025-0.07 and stayed there. Gradient norm 0.017-0.051, log-prob gap 0.003-0.005 |
| Collapse | 96-97 | Described below |

**Collapse, step by step:**
- **Update 96:** gradient norm 0.081, 1.6 times the largest earlier value. Its batch had 16 informative groups (typical 8-12), but even per informative group the norm was high (z = 4.2).
- **Step 97** (rollouts from the weights that update produced):
  - Reward 0.055, with 24 of 32 groups all wrong.
  - Format rate still 0.875.
  - Entropy 0.415, ten times higher.
  - Truncation 0.109.
  - Gradient norm 0.208.
  - Mean log-prob gap 0.0388; largest single-token gap 19.8.
- **Next batch:** gap 0.0794, rejected before any update.

Because reward collapsed while format held, the model still wrote the required tags but gave wrong answers. Under a frozen, deterministic verifier with zero errors, that points to the policy, not the reward.

## 3. Earliest warning signs (fig2_warning_scores.png)

Each value is scored against the median and MAD of the previous 20 steps. A signal is only useful if it is quiet before the incident, so each row counts its false alarms over steps 30-93.

| Signal | Alarms (z ≥ 3.5) in steps 30-93 | z at 94 / 95 / 96 / 97 |
|---|---|---|
| Mean log-prob gap | **0 of 64** | 3.3 / 3.1 / **6.4** / 83 |
| p99 log-prob gap | **0 of 64** | **4.6 / 4.0 / 6.8** / 53 |
| Gradient norm | 2 of 64 (84, 87) | **4.0 / 3.6 / 12.9** / 36 |
| Gradient norm per informative group | 7 of 64 | 0.5 / 0.0 / 4.2 / 47 |
| Parameter change per update (sampled) | 5 of 64 (cluster 84-88) | 1.2 / 1.5 / 3.1 / 6.7 |
| Entropy | 3 of 64 | 0.5 / 0.8 / 1.3 / **52** |
| Reward, format, truncation, length | 0-1 of 64 | inside normal range until 97 |
| Largest single-token gap | 9 of 64 | not elevated before 97 |

**Confirmed:** the joint rule "mean and p99 gap both at z ≥ 3" fires at steps 94, 95, 96 and 97 and nowhere in 30-93. It is the earliest signal with no false alarms in this run: three steps before the damaging update was applied, and four before reward fell. The gradient norm rose at the same time, with two false alarms earlier.

**The gate could not have caught it.** The gap at 94-96 (0.0045-0.0057) was half the level of steps 1-19 (about 0.011) and ten times below the 0.05 gate. It stands out only against its own recent baseline.

**Observation, not shown to be linked:** sampled parameter change per update rose about 15% at step 84 and stayed higher. This is a drift, not a spike.

**No warning:** reward, format, truncation, length and entropy were all normal at step 96.

## 4. Possible explanations

| Explanation | Status | Evidence |
|---|---|---|
| Reward or verification problem | **Not supported** | The frozen verifier re-scores all 2,304 saved rollouts (steps 10-90) identically. math-verify disputes 12 accepted rewards, and all 12 are genuinely correct (`50` vs `\$50`, `3 gallons` vs `3`, `20\%` vs `20`, `A,B,C,D` vs `\text{A,B,C,D}`). It accepts 14 rejected ones (false negatives, none after step 50). Verifier errors: 0 in training and evaluation. **Limit:** the step-97 rollouts were not saved, so the collapsed outputs were not inspected directly |
| Optimization instability | **Supported** | Abrupt one-update collapse, largest gradient of the run, entropy ×10, during a plateau (dev 50 → 75 unchanged). Entropy had been about 0.03 for 60 steps; constant lr 1e-5, no KL term, and clipping cannot act |
| Numerical inconsistency (vLLM bf16 vs PyTorch fp32/autocast) | **Early symptom: confirmed. Cause: unresolved** | The gap rose at 94-96, before the collapse, slightly beyond what entropy explains (step 96 is 2.2 SD above the fit in fig3). At step 97 the gap is 5 times the fit and 3.5 times what the base model showed at the same entropy (about 0.011 at entropy 0.4-0.6). So the collapsed model is numerically unusual, not merely more random. Its weights matched exactly at every step, so the difference is in how the same weights compute, consistent with growing activations or logits |
| Stale weights, misalignment, verifier timeouts, memory | **Not supported** | 0 weight mismatches; per-token alignment enforced every step; 0 timeouts; peak 42.4 GiB with no out-of-memory error |

## 5. Intermediate evaluations (fig4_dev_evals.png)

| Step | GSM8K dev strict (n 1,000) | relaxed | clean format | MATH dev strict (n 500) | relaxed | MATH truncated |
|---|---|---|---|---|---|---|
| 0 | 21.8% | 52.3% | 77.7% | 7.2% | 27.4% | 91 |
| 25 | 89.5% | 89.9% | 99.7% | 30.8% | 33.2% | 64 |
| 50 | 92.4% | 92.5% | 99.7% | 39.2% | 40.6% | 45 |
| 75 | 92.9% | 92.9% | **86.8%** | 39.8% | 41.0% | 47 |

Paired McNemar tests on the same items:

| Interval | GSM8K strict | MATH strict |
|---|---|---|
| 0 → 25 | +67.7 pp, of which +37.6 pp is a real answer gain (relaxed) and the rest is format | +23.6 pp (relaxed +5.8 pp) |
| 25 → 50 | +2.9 pp [+1.1, +4.7], p = 0.003 | +8.4 pp [+4.5, +12.3], p < 0.001 |
| 50 → 75 | +0.5 pp [−1.0, +2.0], p = 0.61 | +0.6 pp [−3.5, +4.7], p = 0.85 |

- **The model had plateaued by step 50.** Nothing measurable was gained between 50 and 75, and there is no evaluation between 75 and the collapse.
- **Anomaly:** at step 75, 132 of 1,000 greedy GSM8K answers (59 of 500 on MATH) contain a doubled `</think> </think>`, against 3 at step 50. Accuracy is unaffected and the reward ignores it. In sampled training rollouts it is rare (0-10 of 256) and absent at step 90. It indicates the greedy argmax shifting on one structural token. Its timing (21 steps before the collapse) does not link it to the failure.
- **Saved training rollouts (steps 10-90) look healthy:** 7.3-8.0 distinct responses per group of 8, 0-28 truncations per 256, repeated 4-grams 2-8%, almost no non-ASCII text. Nothing in the step-90 batch foreshadows the collapse.

## 6. Reassessment of A1 and A2 (frozen configurations unchanged)

**Still appropriate:**
- **Step 80 is a clean fork point.** Every warning signal in section 3 comes after it (earliest: step 84 drift, joint gap signal at 94). Both branches start from a state with no anomaly, about 16 updates before the original's damaging update.
- **A1 is a meaningful replay.** Same weights, optimizer state, sampler RNG and generation seeds. Its step 81 should closely match the original step 81 (reward 0.645, length 465, gap 0.0033); later steps may diverge because GPU kernels are not bit-deterministic.
- **A2 isolates the learning rate.** It halves the Adam step size, which bounds the per-update change, and changes nothing else.

**Limits to state before running:**
1. **One run per branch.** If A1 does not collapse, the original collapse may still be a low-probability event, and A2 surviving would then say nothing about lr. A2 is only informative relative to A1 when A1 reproduces the failure.
2. **The step that matters is not saved by default.** Rollouts are saved every 10 steps (90, 100), and the failed-batch dump only captures the rejected batch after a collapse. If a collapse recurs, the batch that produced the damaging update would most likely be missing. Saving rollouts every step would close this gap without changing any update. It is not in the allowed changes and the configurations are frozen, so I have not changed it. **This is a decision for you.**
3. **The gate is a backstop, not an early warning.** The warning appeared at gaps of about 0.005. The new per-step logs (gap p99, tokens above 1 and 5 nats, update size over all parameters) let the same z-score analysis run on the branches afterwards. No threshold changes.
4. **Interpretation rule, fixed before running:**
   - Collapse = any step with training reward < 0.3, or a rollout-gate stop.
   - Early warning = mean and p99 gap both at z ≥ 3 against the previous 20 steps.
   - Report per branch: collapse yes/no and its step, warning step, the step-80 and final dev scores, and A1's step-81 fidelity.

**Cost from the measured timings:**
- 124 s per training step and 80-150 s per evaluation.
- About 55-60 minutes per branch, including loading and hashing the 18.5 GB state and three saves.
- About 2 hours and 4 GPU-hours for both, below the earlier 2-3 hour estimate.
