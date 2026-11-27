# LLM Reasoning Post-Training with SFT and GRPO

**Technical report: LLM engineering and empirical research.** An empirical study of reasoning data engineering, LLM supervised fine-tuning, online reinforcement learning, multi-GPU training and inference infrastructure, numerical stability and checkpoint recovery, and statistical benchmark evaluation on Qwen2.5-Math-1.5B, using 2× NVIDIA RTX A6000. Report date: 2026-11-27. Results frozen (`results/research_analysis/PHASE1_FREEZE.md`).

**Project and provenance.**
- **Project:** The experimental framework investigates mathematical reasoning through supervised fine-tuning and reinforcement learning with verifiable rewards. Existing mathematical verification components and prompt templates are incorporated with their original attribution and licensing, as documented in THIRD_PARTY_NOTICES.md. The post-training system (`posttrain/`), configurations, scripts, tests, experiments and analysis were developed with AI coding assistance (Claude Code) under the author's direction.
- **Third-party inputs:**
  - The SFT reasoning traces are a public dataset (`garg-aayush/sft-cs336-assign5-datasets`, written by gpt-oss-120b on math12k problems). They were **not generated in this project**.
  - The strict grader is the Dr. GRPO grader (Apache-2.0).
  - The R1-Zero prompt is the DeepSeek-R1-Zero format, with the text used in the MIT-licensed Dr. GRPO code.

Section 11 states exactly what came from where.

---

## Contents

1. Abstract · 2. Introduction and contributions · 3. Background: objectives · 4. Contribution 1: reasoning data curation and re-verification · 5. Contribution 2: supervised fine-tuning · 6. Contribution 3: online RL with verifiable rewards · 7. Contribution 4: dual-GPU training and inference architecture · 8. Contribution 5: numerical stability and checkpoint recovery · 9. Contribution 6: statistical benchmark evaluation · 10. Contribution 7: reproducible experimentation · 11. Provenance and engineering summary · 12. Discussion and limitations · 13. Conclusion and future work · 14. Reproducibility appendix

---

## 1. Abstract

**Headline.** Sequential GRPO (SFT, then 100 steps of GRPO) improved held-out accuracy over its own SFT parent by **+13.4 pp on GSM8K and +11.7 pp on MATH**, reaching **86.1% / 72.5%**.
- This is a pre-registered confirmatory comparison (C3): Holm p = 1.9e-25 and 5.7e-73.
- The gain is not a formatting artefact. 234 of the 237 GSM8K items GRPO newly solved were already well-formatted but wrong in the parent. The gain survives relaxed answer extraction and a second, independent verifier (+11.2 pp GSM8K, +10.5 pp MATH).

**Setup.** We post-trained Qwen2.5-Math-1.5B with supervised fine-tuning on teacher reasoning traces and with GRPO, a reinforcement-learning method with verifiable rewards.
- **Scale:** 17 documented training runs.
- **Evaluation:** 11 model groups, each evaluated once on held-out GSM8K test (n = 1,319) and MATH val (n = 5,000).
- **Protocol:** the plan was frozen and hashed before any held-out result existed, and checkpoint identities were locked before evaluation.

**Other findings.**
- **Data quality.** SFT data "recovered" by a strict verifier scored **lower** on GSM8K than exact-match-filtered data: −9.3 pp with the full sets, −8.6 pp with 512 matched examples. This is the opposite of the motivating hypothesis. The result rests on one training seed per arm.
- **Baselines matter.** The base model gains +47.9 pp on GSM8K from the prompt alone. Against its stronger prompt, SFT with 512 or more examples adds 6–16 pp (GSM8K) and 2–4 pp (MATH). SFT on 128 examples is worse than that prompt.
- **Stability.** The pre-registered direct-from-base GRPO run collapsed at update 96. A rollout log-probability gate stopped it before any corrupted update. It is reported as a failed experiment; post-hoc branches from its step-80 state are diagnostics, not a completion of it.
- **Diminishing returns.** On development data, GRPO's measurable gain arrived in the first 25 of 100 steps. After that, about two-thirds of rollout groups carried no learning signal.

All intervals in this report quantify sampling of evaluation questions only. Each arm was trained once.

---

## 2. Introduction and contributions

Reasoning post-training typically combines SFT on teacher-written solutions with reinforcement learning from verifiable rewards (RLVR). Comparisons between such recipes are easily distorted by weak baseline prompts, by checkpoint or hyperparameter selection on the test set, and by unreported unstable runs. This project builds a complete post-training system for a 1.5B math model and uses it in a pre-registered study designed to control those three effects.

The work is organised around seven engineering contributions.

| # | Contribution | Section |
|---|---|---|
| 1 | **Reasoning data curation and re-verification:** teacher traces, exact-match vs strict re-verification, decontamination, controlled subsets | §4 |
| 2 | **Supervised fine-tuning:** response-masked SFT with exact gradient accumulation; learning rate chosen by a rule fixed in advance | §5 |
| 3 | **Online RL with verifiable rewards:** GRPO with online vLLM rollouts, binary verifier rewards and group-relative advantages; direct vs SFT-initialised RL | §6 |
| 4 | **Dual-GPU training and inference architecture:** PyTorch policy + vLLM engine, mixed precision, token-budget micro-batching, weight synchronisation, checkpointing | §7 |
| 5 | **Numerical stability and checkpoint recovery:** exact sync and log-prob gates, collapse diagnosis, controlled checkpoint branching | §8 |
| 6 | **Statistical benchmark evaluation:** frozen held-out protocol, paired statistics with multiplicity control, formatting-vs-reasoning decomposition, dynamics and efficiency | §9 |
| 7 | **Reproducible experimentation:** run registry, pre-registration, checkpoint lock, hash-locked verifier, run records, audits and CPU tests | §10 |

![System architecture](figures/fig00_architecture.png)

---

## 3. Background: objectives

### 3.1 Response-masked supervised fine-tuning

Given a prompt *x* and a teacher response *y* (a reasoning trace ending in `</think> <answer>…</answer>`), SFT minimises the negative log-likelihood of the response tokens only; prompt tokens are masked. The implementation divides by the number of response tokens in the **whole optimizer batch** *B* (32 sequences). Gradient accumulation over micro-batches therefore gives the same gradient as one large batch (unit-tested):

$$\mathcal{L}_{\mathrm{SFT}}(\theta) = -\frac{1}{\sum_{i \in B} |y_i|} \sum_{i \in B} \; \sum_{t=1}^{|y_i|} \log \pi_\theta(y_{i,t} \mid x_i, y_{i,<t})$$

### 3.2 GRPO with verifiable rewards

For each training prompt *q*, GRPO samples *G* = 8 responses online from the current policy and scores each with a binary verifier reward *r* ∈ {0, 1}. The advantage is the reward minus the group mean (no standard-deviation normalisation in this study):

$$A_i = r_i - \frac{1}{G}\sum_{j=1}^{G} r_j$$

Equivalently it is a scaled leave-one-out baseline:

$$A_i = \frac{G-1}{G}\left(r_i - \frac{1}{G-1}\sum_{j \neq i} r_j\right)$$

With the per-token importance ratio

$$\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t} \mid q, o_{i,<t})}$$

the implemented clipped surrogate loss over the *N* = 256 rollouts of a step (32 prompts × 8), with constant length normaliser *L* = 1,024 and ε = 0.2, is

$$\mathcal{L}_{\mathrm{GRPO}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} \frac{1}{L}\sum_{t=1}^{|o_i|} \min\left(\rho_{i,t} A_i,\ \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\, A_i\right)$$

**What actually acts in this configuration.** Each rollout batch is used for exactly one optimizer step.
- **Clipping is inactive.** The ratio is exactly 1 when the update is computed, so clipping never binds; the clip fraction was 0 at every logged step.
- **The update is REINFORCE.** The gradient reduces to REINFORCE with a group-mean baseline:

$$\nabla_\theta \mathcal{L} = -\frac{1}{N L}\sum_{i=1}^{N} A_i \sum_{t} \nabla_\theta \log \pi_\theta(o_{i,t} \mid q, o_{i,<t})$$

- **Nothing else is added.** There is no KL penalty and no reference model. No result in this report can be attributed to clipping.
- **Not a new algorithm.** The project implements a standard GRPO-style objective; it does not propose a new one.

**Zero-advantage groups.** If all 8 responses to a prompt receive the same reward, every advantage in the group is 0. The group then contributes nothing to the gradient but still costs generation and forward/backward compute.

**Constant length normalisation.** Dividing by a constant rather than by each response's length avoids the per-sequence length bias of length-averaged losses.

---

## 4. Contribution 1: Reasoning data curation and re-verification

![Data pipeline](figures/fig14_data_pipeline.png)

**Teacher and student**

| Role | Model | Evidence |
|---|---|---|
| Teacher | gpt-oss-120b. Its 4,836 traces were taken from a public dataset, not generated here. | Source files `sft_gpt-oss-120b.jsonl` and `sft_gpt-oss-120b_filtered.jsonl`, hashed in `data/stage0/MANIFEST.json`; `EXPERIMENT_PLAN.md` §0 |
| Student | Qwen2.5-Math-1.5B, pinned revision `4a83ca6e…` | `configs/`, `PT_BASE_SHA256` in `scripts/server_env.sh` |

{{table:t14_data}}

**Two label filters define the data-quality contrast**
- **Exact-match (3,496 traces).** This is the dataset's own published filtered file: traces whose extracted answer string-matches the reference.
- **Re-verified (4,545 traces), built in this project.** Every trace was re-scored with the strict math verifier, which normalises LaTeX and checks numeric and symbolic equality. This keeps all exact matches and recovers 1,049 of the 1,340 traces the string filter rejected (78.3%).
- **Independent confirmation of the recovered labels.** 781 by a canonicaliser, 148 as exact rationals, 80 by math-verify, 40 by reading. There were 0 numeric conflicts (`EXPERIMENT_PLAN.md` §1).

**Answer verification (reward and metric)**
- **The verifier.** Verifier 2.0.0 wraps the Dr. GRPO strict grader (format check on `<think>`/`<answer>`, answer normalisation, numeric and symbolic equivalence via sympy and math-verify parsing). It adds two narrow acceptance rules: a whole-string "Answer is …" lead-in, and *n*! against an integer reference. Nothing the strict grader accepts is ever rejected.
- **Effect of the two rules.** They changed 6 of 8,832 recorded outputs and no SFT label (`results/verifier_v2/crosscheck.json`).
- **Audit.** On 13,017 answer mutations the verifier produced 3 flags, 2 of them genuine false accepts. On 19,857 random answer pairs it accepted 1, which was a legitimate match. It also has false rejects: 118 of the 291 still-rejected traces are equivalent by an independent check. It was kept strict rather than loosened.
- **Integrity.** The verifier's file hashes are locked, and every run refuses to start if they differ.

**Cleaning and decontamination**
- **GSM8K:** test had 0 exact, 0 shared 13-gram and 0 fuzzy overlaps with the SFT and RL data.
- **MATH:** val overlapped 6 exact and 119 fuzzy SFT problems, so 120 SFT and 162 RL problems were excluded from training.
- **Duplicates:** seven normalised-duplicate SFT problems were excluded.
- **Final pools:** 3,420 exact-match and 4,423 re-verified examples. The re-verified pool contains every exact-match example plus 1,003 recovered ones.

**Controlled sample sizes**
- **Nested subsets.** Subsets share one random ordering across arms, so arms differ only where eligibility differs.
- **Overlap at 512.** The exact-match and re-verified subsets share 387 of 512 examples.
- **Size control.** This arm (3,420 examples drawn from the re-verified pool) shares 2,601 with the exact-match set, which separates composition from volume.
- **Scaling grid.** 128, 512, 2,048 and the full set.

**Evaluation splits**
- **Dev sets, used for all tuning and monitoring:** GSM8K dev (1,000, drawn from GSM8K *train*) and MATH dev (500 untraced math12k problems, a harder subset).
- **Held-out sets:** used once per frozen checkpoint. Their item ids are disjoint from all dev and SFT ids.

---

## 5. Contribution 2: Supervised fine-tuning

### 5.1 Training workflow

```text
Algorithm 1 — SFT step (posttrain/train_sft.py)
  batch B of 32 (prompt, teacher response) pairs, sorted by length
  T ← number of response tokens in B                         # whole-batch denominator
  for each micro-batch m ⊂ B (fixed size, or token-budget packed):
      loss_m ← −Σ_{response tokens in m} log π_θ(token) / T   (bf16 autocast, fp32 weights)
      loss_m.backward()                                       # gradients accumulate
  clip ‖g‖ to 1.0; AdamW step; log loss, entropy, update size
```

The objective is the response-masked negative log-likelihood of Section 3.1. Every micro-batch divides by the token count of the whole batch, so micro-batching and token-budget packing leave the gradient unchanged (unit-tested).

### 5.2 Hyperparameters and learning-rate selection (SFT and GRPO)

{{table:t13_hyperparameters}}

**The learning-rate rule was fixed before tuning:** take the highest final-step GSM8K dev accuracy; if another rate is within 1.0 pp of it, take the smallest such rate. Both SFT and GRPO selected 1e-5. GRPO rates were tuned from the base model over 30 steps and applied to both pathways. The configuration files' defaults are not what ran; the rates actually used are recorded in each run's `run.json`.

{{table:t12_lr_tuning}}

---

## 6. Contribution 3: Online reinforcement learning with verifiable rewards

### 6.1 Training workflow

```text
Algorithm 2 — GRPO step (posttrain/train_grpo.py)
  1  load weights into vLLM in place; check engine tensor == bf16(policy) for all  [gate 1]
  2  sample 32 prompts; vLLM generates 8 responses each (T = 1.0, ≤ 1,024 tokens) with log-probs
  3  reward r ∈ {0,1} from verifier 2.0.0; advantages A = r − group mean
  4  policy recomputes log-probs of the sampled tokens (no grad)
  5  require mean |log p_vLLM − log p_policy| ≤ 0.05; else save batch, FAILED  [gate 2]
  6  loss over 256 rollouts (÷256, ÷1,024), token-budget micro-batches; clip ‖g‖ to 1.0
  7  one AdamW step; log 40+ diagnostics (zero-adv share, entropy, lengths, gaps, timings)
```

**Rollouts are online and on-policy up to engine numerics.** Every batch is generated by the current weights immediately after a verified synchronisation. The residual off-policyness, the gap between vLLM's and PyTorch's log-probabilities on identical weights, is measured on every token (Section 8).

The hyperparameters and the learning-rate rule are shared with SFT (Section 5.2).

### 6.2 Pathways: sequential and direct GRPO

| Pathway | Definition | Status |
|---|---|---|
| **Sequential (GRPO-B)** | base → SFT on 512 re-verified examples → 100 GRPO steps | completed; pre-registered |
| **Direct (GRPO-A)** | base → 100 GRPO steps | **FAILED at step 98**; pre-registered; never evaluated |
| A1, A2 | GRPO-A's step-80 state + 20 steps (lr 1e-5 / 5e-6) | post-hoc diagnostics; evaluated but never substituted for GRPO-A |

---

## 7. Contribution 4: Dual-GPU training and inference architecture

| Component | Implementation | Where |
|---|---|---|
| **Dual-GPU split** | vLLM 0.7.2 rollout engine on GPU 0 (bf16, 80% memory, prefix caching off); fp32 PyTorch policy + AdamW on GPU 1 | `engines.py`, `train_grpo.py` |
| **Weight synchronisation** | Each step the policy state dict is loaded into the live vLLM model in place (`load_weights`), then every engine tensor is compared bit-for-bit with the policy tensor converted to bf16 (names, dtypes, shapes, values) | `engines.verified_sync` |
| **Mixed precision** | fp32 master weights with bf16 autocast. A Stage 0 check showed that pure-bf16 Adam steps at lr 1e-5 leave 91.3% of weights unchanged | `model_io.load_policy`; `results/stage0/bf16_update_check.json` |
| **Token-budget micro-batching** | Micro-batches packed under a padded-token budget (3,300 on A6000) as well as a sequence cap, because peak memory tracked the largest padded micro-batch almost linearly in the pilot. The loss keeps the whole-batch denominator, so the gradient is unchanged (unit-tested) | `batching.py` |
| **Gradient accumulation** | Whole-batch denominators (tokens for SFT, rollouts for GRPO); one optimizer step per batch | `train_sft.py`, `train_grpo.py` |
| **Memory management** | Separate devices for engine and policy; `expandable_segments` allocator; micro-batch 2 on 48 GB; entropy computed one sequence at a time; per-step peak-memory logging | `scripts/server_env.sh`, `losses.py`, `diagnostics.py` |
| **Configuration** | YAML configs generated with the experiment registry from one table, with `${VAR}` expansion; every run records config, code and verifier hashes | `scripts/make_configs.py`, `common.py` |
| **Logging** | Append-only `metrics.jsonl` per run (40+ fields per GRPO step); rollouts saved with token-level log-probs | `common.Run`, `diagnostics.py` |
| **Checkpointing** | Atomic save (temp dir → hash → rename) with `lineage.json`: parent weights hash, config hash, code hash. Training refuses a parent that does not match its lineage | `model_io.py` |
| **Resume** | `state.pt` with model, optimizer and all RNG states. On resume, later log rows and files are moved aside (never deleted) so no step is logged twice. Bit-exact on CPU (tested) | `model_io.py`, `common.Run.rewind` |
| **Launch safety** | Gated launchers refuse to run without explicit authorisation variables, check that the GPUs are idle, and never resume a failed run | `scripts/server_*.sh` |
| **Tests** | 174 CPU tests: gradient equivalence across micro-batching and token budgets, resume bit-exactness, gate behaviour, lineage tamper detection, launcher refusals | `tests/` |

---

## 8. Contribution 5: Numerical stability and checkpoint recovery

### 8.1 Automated correctness gates

1. **Exact parameter synchronisation**, before every rollout batch and every evaluation. Any mismatched tensor stops the run. It passed at every step of every run.
2. **Token-level log-probability consistency**, every GRPO step before the update. The mean |sampling log-prob − policy log-prob| over all rollout tokens must be at most 0.05, and every token must have a value. On failure the batch is saved with token-level detail and `FAILED.json` is written; the model is not updated.
3. **Non-finite loss or gradient** stops the run before `optimizer.step()`.
4. **Incident guard.** A run with an incident record is never resumed, replaced or completed.

### 8.2 The GRPO-A optimization failure

![GRPO-A incident trajectory](figures/fig11_grpo_a_incident_trajectory.png)

GRPO-A (direct GRPO from the base model, lr 1e-5 by the pre-registered rule) went through four phases (`results/grpo_a_incident_analysis/REPORT.md`):
- **Format learning (steps 1–20):** reward 0.02 → 0.62, format rate 0.10 → 1.00, entropy 0.54 → 0.11.
- **Plateau (steps 20–95):** reward 0.55–0.86, entropy 0.025–0.07, gradient norm 0.017–0.051, rollout log-prob gap 0.003–0.005. Dev accuracy at step 75 was 92.9% (GSM8K) and 39.8% (MATH).
- **Collapse (update 96).** Update 96 had the largest gradient of the run (0.081, 1.6× the previous maximum). The rollouts generated from its weights (step 97) showed:
  - reward 0.055, with 24 of 32 groups all wrong;
  - entropy 0.415, about ten times higher;
  - truncation 0.109;
  - a mean log-prob gap of 0.0388, with a largest single-token gap of 19.8 nats;
  - a format rate that held at 0.875.
- **Gate.** The next batch's mean gap was 0.0794, above the pre-registered 0.05. The batch was rejected **before any update**, and the run was marked FAILED after 97 completed updates.

**Diagnosis**

| Hypothesis | Evidence | Verdict |
|---|---|---|
| Verifier fault | 0 verifier errors in 24,832 training rollouts; all saved rewards reproduce; format held at 87.5% at the collapse | Ruled out as far as the evidence reaches |
| Stale or partial weights in vLLM | Exact sync check passed at step 97 and before the rejected step 98 | Not supported |
| Token misalignment, padding, micro-batching | Per-rollout length checks; padding-invariance and micro-batch-equivalence tests | Not supported |
| Clipping failure | Clipping cannot act at ratio 1 | Not applicable |
| **Optimization instability from a low-entropy plateau** | Single-update break after a long plateau at entropy ≈ 0.03; largest gradient of the run | **Supported** |
| **Train–inference numerical drift** | The gap rose above its own baseline at steps 94–95 (about 0.005, ten times below the gate), and at step 97 was 3.5–5× what the same entropy produced elsewhere | Early warning; cause vs consequence **unresolved** |

Reward collapsed while format held: the model still produced the required tags but gave wrong answers. Under a frozen, deterministic verifier this points to the policy rather than the reward. One failure in one seed gives no estimate of how often such collapses occur.

### 8.3 Controlled checkpoint branching

![Collapse and recovery](figures/fig05_collapse_recovery.png)

**The failed run was preserved**
- Its folder is refused by the code, and the launcher skips it.
- Its last saved state (step 80, sha256 `73114307…e1bb`) was used read-only as the source of two branches in new folders.
- The branch criteria were registered before the branches trained (`recovery/diagnostic_criteria.yaml`).

| Branch | Change | Outcome (step 100) | Held-out (GSM8K / MATH) |
|---|---|---|---|
| A1, same-lr replay | none (lr 1e-5) | completed, every gate passed; reproduced the original step 81 exactly, then diverged | 85.5% / 70.5% |
| A2, reduced lr | lr halved to 5e-6 | completed, every gate passed; learned a doubled `</think>` tag | 84.5% / 70.6% |

**Interpretation**
- **Not a completion of GRPO-A.** The branches are post-hoc diagnostics: they describe "GRPO-A's step-80 state plus 20 steps". They do not complete or replace the registered run, and the registered direct-vs-sequential comparison (C4) remains **unavailable**.
- **Likely cause.** Neither branch collapsed. The likeliest reading is a rare, trajectory-dependent instability rather than a deterministic consequence of the step-80 state, but two branches cannot establish that.
- **Preserved state confirmed.** The post-evaluation integrity check confirmed that the failed run's files, including its step-80 state, were unchanged after all evaluation (`results/research_analysis/AUDIT.md` §4).

---

## 9. Contribution 6: Statistical benchmark evaluation

### 9.1 Evaluation protocol

- **Held-out sets.** GSM8K test (1,319, primary) and MATH val (5,000, secondary), each evaluated once per checkpoint.
- **Frozen plan.** `final/freeze_v1.yaml`, sha256 `ff0f0ce5…a453`, lists the 11 model groups by registry alone. A lock with every checkpoint's weights hash was written 12 s before the first evaluation, and each evaluation re-hashed its weights against the lock.
- **Decoding.** Greedy, 1,024 new tokens, vLLM 0.7.2 with prefix caching off, RTX A6000, seed 2026. Final step only.
- **Prompts.** Trained models use r1_zero, the format they were trained on. The base model is evaluated with r1_zero and with a bare question-only prompt; the question-only prompt is its stronger baseline.
- **Scoring.** Strict accuracy (primary), second-verifier accuracy (math-verify) and relaxed accuracy (answer extracted from the `<answer>` tag, the last `\boxed{}` or the last number, ignoring format).

**Evidence labels**

| Label | Meaning |
|---|---|
| Confirmatory | Pre-registered C1–C4, Holm-adjusted (m = 4 per set; C4 counted p = 1) |
| Confirmatory-secondary | Pre-registered SFT vs base question-only (m = 7 per set) |
| Descriptive | Pre-registered; CI and unadjusted p; no claim rests on them |
| Post-hoc | A1/A2 family (frozen, m = 4) and comparisons chosen after the held-out results (m = 5); exploratory |
| Exploratory (dev) | Development-set curves and diagnostics |

### 9.2 Main results

{{table:t01_main_results}}

![Overall performance](figures/fig01_overall_performance.png)

### 9.3 Strongest finding: GRPO over its SFT parent (confirmatory C3)

{{table:t03_confirmatory}}

**C3 is supported on both benchmarks:** +13.4 pp [+11.0, +15.9] on GSM8K and +11.7 pp [+10.5, +13.0] on MATH.

**Why the gain cannot be attributed solely to formatting**
1. **The parent was already well-formatted.** SFT re-verified 512 produces correctly formatted output on 99.6% of GSM8K items, which leaves almost no formatting headroom.
2. **Item-level transitions.** On GSM8K, 234 of the 237 newly correct items were well-formatted but wrong in the parent (the other 3 were truncated). On MATH, 772 of 843 were well-formatted but wrong.
3. **Format-insensitive scoring.** Under relaxed extraction, which ignores the output format entirely, the gain is +11.2 pp (GSM8K) and +10.5 pp (MATH). The second verifier (math-verify), which changes only how a formatted answer is compared, gives the same values.
4. **Beyond any SFT data source.** GRPO-B also beats the best SFT arm (exact 512, a different data source) by +4.9 pp on GSM8K and +10.6 pp on MATH (post-hoc, Holm 2e-5 and 4e-65).

![Gain decomposition](figures/fig10_gain_decomposition.png)

**Cost of the gain: longer outputs**
- Mean response length rose from 138 to 253 tokens on GSM8K and from 295 to 460 on MATH.
- MATH truncation rose from 4.5% to 5.7%; truncated responses account for 20.6% of GRPO-B's MATH errors.
- **Seed caveat.** GRPO-seed variance is unknown (no replicate). The effect is about ten question-sampling standard errors, but its seed robustness is not demonstrated.

### 9.4 SFT against the base model (confirmatory-secondary)

{{table:t04_sft_vs_base}}

- **Six of seven SFT arms beat the base model's best prompt on both benchmarks** (Holm p < 0.01 in every case): +6.4 to +15.7 pp on GSM8K and +2.0 to +4.1 pp on MATH.
- **SFT on 128 examples is worse** than base question-only: −4.2 pp on GSM8K and −8.8 pp on MATH. At *n* = 128 the GSM8K format rate is 82.5%, and relaxed accuracy is 72.5% against 61.5% strict.
- **Much of the margin is answer extraction.** Under relaxed scoring the GSM8K advantage over question-only shrinks to +8.1/+8.3 pp for the exact-match arms, and to between −0.7 and +3.0 pp for the re-verified arms with 512 or more examples (column "relaxed b − a").

### 9.5 Data quality: exact-match vs re-verified (confirmatory C1, C2)

![Data quality](figures/fig02_data_quality.png)

| Contrast | GSM8K | MATH | Label |
|---|---|---|---|
| C1 full: exact 3,420 → re-verified 4,423 | **−9.3** [−11.6, −7.1], Holm 2.7e-15 | **−2.0** [−3.2, −0.8], Holm 0.003 | confirmatory: re-verified worse |
| C2 matched 512 | **−8.6** [−10.8, −6.4], Holm 8.0e-14 | −1.1 [−2.3, +0.0], Holm 0.13 | GSM8K: re-verified worse; MATH: not detected |
| Composition: exact 3,420 → re-verified 3,420 | −6.4 [−8.5, −4.4], p 2.2e-9 | −0.4 [−1.6, +0.8], p 0.55 | descriptive |
| Volume: re-verified 3,420 → 4,423 | −2.9 [−4.9, −0.9], p 0.006 | −1.6 [−2.8, −0.4], p 0.007 | descriptive |

**Direction**
- On GSM8K every arm trained with recovered traces scored 6–9 pp below its exact-match counterpart, and composition carries most of the gap.
- The confirmatory finding is "a difference exists, favouring exact-match data". That contradicts the motivation for recovering traces.

**Mechanism (descriptive, not established)**
- **Real errors.** Of the 171 GSM8K items solved by exact 512 but not by re-verified 512, 169 were well-formatted but wrong.
- **Verifier style.** About a quarter of the C2 gap is verifier style: under the second verifier the gap is −6.4 pp.
- **Untested explanations.** Recovered traces may teach answer styles the strict verifier rejects, and the exact-match filter may select simpler problems. These are not separated here.

**Robustness**
- **To question sampling:** very high (p ≤ 1e-13 on GSM8K).
- **To training seed:** not established (§9.10).

### 9.6 Data scaling (descriptive)

![SFT scaling](figures/fig03_sft_scaling.png)

| Step (re-verified data) | GSM8K | MATH |
|---|---|---|
| 128 → 512 | +11.2 [+8.2, +14.2] | +11.1 [+9.6, +12.5] |
| 512 → 2,048 | +3.3 [+1.0, +5.5] | +1.8 [+0.6, +3.1] |
| 2,048 → 4,423 | −3.9 [−6.0, −1.9] | −2.1 [−3.3, −0.9] |
| exact 512 → exact 3,420 (post-hoc) | +0.1 [−1.9, +2.1] | +0.6 [−0.7, +1.9] |

- **Diminishing returns.** Returns fall sharply after 512 examples. Beyond 512, single-seed differences of 2–4 pp go in both directions and lie within the estimated seed band.
- **Not equivalence.** No difference was detected between exact 512 and exact 3,420; that is not evidence that they are equivalent.
- **Confound.** Fixed epochs mean larger datasets also received more optimizer steps.

### 9.7 Prompt sensitivity and strict vs relaxed accuracy (descriptive)

![Strict, relaxed and format accuracy](figures/fig07_strict_relaxed_format.png)

- **The prompt alone is worth +47.9 pp on GSM8K and +41.2 pp on MATH** for the base model (17.7% → 65.7%; 17.4% → 58.6%).
- **It is not only formatting.** Under r1_zero only 51.1% of the base model's GSM8K outputs are well-formatted, and even among those it is right 34.7% of the time. Of the 689 GSM8K items gained by switching prompt, 286 were well-formatted but wrong under r1_zero.
- **Verifier dependence.** The second verifier adds 14.4 pp (GSM8K) and 5.9 pp (MATH) to the base model under r1_zero. It adds at most 2.2 pp to any SFT arm of 512 examples or more, and 0.0 / 0.6 pp to GRPO-B. Conclusions about trained models are therefore stable to the choice of verifier.
- **Formatting quirk.** The post-hoc A2 branch emits a doubled `</think>` tag in 1,233 of 1,319 GSM8K and 4,406 of 5,000 MATH responses. The format check accepts it (a verifier leniency).

{{table:t02_lengths_truncation}}

### 9.8 Sequential vs direct GRPO

- **Registered answer: unavailable** (GRPO-A failed).
- **Post-hoc only:**
  - **GSM8K:** no difference was detected between either branch and GRPO-B (A1 → GRPO-B +0.6 pp, Holm 1.0). That is not equivalence.
  - **MATH:** GRPO-B is about 2 pp ahead (+2.1 pp vs A1, Holm 6e-4; +1.9 pp vs A2, Holm 0.002). This is within the estimated SFT seed band and comes from branches of a run that later collapsed.
- **Compute was similar:** 7.79 cumulative GPU-h for GRPO-B (including its 0.44 GPU-h SFT parent) against 7.37 and 7.59 for A1 and A2 (including GRPO-A steps 1–80).

{{table:t06_posthoc}}

![GRPO learning curves](figures/fig04_grpo_learning_curves.png)

**Dev curves**
- **GRPO-B:** GSM8K dev moved from 82.0% (its SFT parent) to 92.7% at step 25 and ended at 92.8%. MATH dev moved from 30.2% to 45.8% at step 25 and ended at 47.8%.
- **GRPO-A:** starting from the base model at 21.8% GSM8K dev under r1_zero, it reached 89.5% by step 25 and 92.9% at step 75.

### 9.9 Reward saturation, zero-advantage groups and efficiency (exploratory, dev)

![GRPO dynamics](figures/fig06_grpo_dynamics.png)

{{table:t09_grpo_windows}}

- **Reward saturation.** In GRPO-B, 54% of groups had identical rewards across all 8 samples in steps 0–25, rising to 70% and 72% in the next two windows. Most are all-correct: the policy saturates the training pool's difficulty.
- **Gains stop with the signal.** Only the 0–25 window shows a significant dev gain (+10.7 pp GSM8K, p 9e-18; +15.6 pp MATH, p 7e-13). From step 25 to 100 the net change is +0.1 pp on GSM8K and +2.0 pp on MATH dev, neither significant.
- **Filtering trigger met, not tested.** The pre-registered trigger for zero-advantage filtering was met: the informative rate was below 0.6, and the update took 54% of step time. The intervention was **not run**, so no speedup is claimed.

![Efficiency](figures/fig08_efficiency.png)

| Investment | GPU-h | GSM8K gain | pp per GPU-h |
|---|---|---|---|
| SFT on 512 examples over base question-only (re-verified / exact) | 0.44 | +7.1 / +15.6 | 16 / 35 |
| GRPO-B over its parent (C3) | 7.35 | +13.4 | 1.8 |
| GRPO-B steps 0–25 (dev) | 1.65 | +10.7 (dev) | 6.5 |
| GRPO-B steps 25–100 (dev) | 5.14 | +0.1 (dev, n.s.) | ≈ 0 |
| Failed GRPO-A | 6.61 | — | sunk |

**Diminishing returns.**
- Early stopping near step 25 would probably have kept most of the dev gain for about a quarter of the cost. This is a dev-based inference, **not verified on held-out data**, because only step 100 was evaluated, by design.
- Held-out evaluation of all 11 model groups (6,319 items each) took about 1.5 h on the two A6000s.

{{table:t11_efficiency}}

### 9.10 Statistical analysis

**Methods.** For checkpoints *a* and *b* on the same *n* items, with discordant counts *b* (only *a* correct) and *c* (only *b* correct), the exact McNemar test and the paired difference are

$$p = \min\left(1,\ 2\sum_{k=0}^{\min(b,c)} \binom{b+c}{k}\, 2^{-(b+c)}\right)$$

$$\hat{d} = \frac{c-b}{n}, \qquad \widehat{\mathrm{SE}} = \frac{1}{n}\sqrt{(b+c) - \frac{(c-b)^2}{n}}$$

with 95% CI d ± 1.96 SE.
- **Corrections and thresholds.** Accuracies use Wilson intervals; Holm correction is applied within each family and benchmark. On GSM8K, differences below 1.4 pp are reported as "not detected".
- **Independent reproduction.** Every statistic in the generated report was recomputed by independent code: 238 metric fields and 382 comparison fields, 0 mismatches.

**Relaxed-accuracy sensitivity of the confirmatory family**

{{table:t07_relaxed_sensitivity}}

**Two kinds of uncertainty.**
- **What the CIs cover.** Every CI and p-value treats the trained weights as fixed and covers only **which evaluation questions were sampled**.
- **What they do not cover.** Each arm was trained **once**, so training-seed variance is not quantified.
- **The one seed contrast.** The only same-data, same-learning-rate contrast is on dev: SFT re-verified 512 with seed 0 (82.0% GSM8K dev) against seed 100 (86.1%), a 4.1 pp difference. Treated as a one-degree-of-freedom estimate, this gives a heuristic seed SD for a two-arm contrast:

$$\hat{\sigma}_{\mathrm{seed}} = \sqrt{\Delta^2 - \mathrm{SE}^2_{\mathrm{item}}}, \qquad \mathrm{SE}_{\mathrm{comb}} = \sqrt{\mathrm{SE}^2_{\mathrm{item}} + \hat{\sigma}^2_{\mathrm{seed}}}$$

{{table:t08_seed_sensitivity}}

- **C1 and C2 (GSM8K).** They sit about 2.1–2.3 combined SEs from zero: suggestive, not robust. On dev the C2 gap shrank from 6.9 pp to 2.8 pp under a seed change.
- **MATH effects.** All MATH data-quality and scaling effects lie within the seed band.
- **Large effects.** C3, the prompt effect and the 128 → 512 step are large relative to any plausible seed SD.
- **Fragility of the estimate.** It rests on one pair. The true seed SD could be several times smaller or larger.

**Cross-benchmark agreement (exploratory)**
- **Rankings agree.** Across the 10 trained checkpoints, GSM8K and MATH accuracies are rank-correlated (Spearman ρ = 0.93, p = 1e-4), and 25 of 27 paired comparisons have the same sign on both sets.
- **Dev predicts held-out ranks.** Rank correlations are ρ = 0.99 on GSM8K, 0.94 on MATH, and 31 of 34 comparisons have the same sign.
- **But not levels.** GSM8K dev runs 8 pp above test; MATH dev runs 29 pp below MATH val.
- **Hard items.** 74 GSM8K and 737 MATH items were solved by none of the eight pre-registered trained checkpoints. The base model's question-only prompt solved 3 and 46 of them.

![Cross-benchmark](figures/fig09_cross_benchmark.png)

### 9.11 Hypothesis scorecard

| Hypothesis | Status |
|---|---|
| GRPO improves on its SFT parent (C3) | **Supported** on both benchmarks (confirmatory) |
| GRPO gains are reasoning, not format | **Supported (descriptive):** 234 of 237 GSM8K gains were formatted-wrong before; gain persists under relaxed and second-verifier scoring |
| SFT (n ≥ 512) improves on base question-only | **Supported** (confirmatory-secondary); margin shrinks under relaxed scoring |
| SFT with 128 examples improves on question-only | **Not supported:** worse (difference detected) |
| Re-verified data differs from exact-match data (C1, C2) | **Difference detected on GSM8K, favouring exact-match**; MATH: C1 −2.0 pp, C2 not detected; seed robustness uncertain |
| Recovering rejected traces improves SFT | **Not supported**; contradicted on GSM8K in this single-seed design |
| More SFT data keeps helping | **Inconclusive beyond 512** |
| Direct GRPO improves on base (C4) | **Unavailable:** the pre-registered run failed |
| Sequential beats direct GRPO | **Not tested as registered**; post-hoc: no GSM8K difference detected, MATH +2 pp for sequential |
| GRPO returns diminish with compute | **Supported on dev (exploratory)**; not testable on held-out data |

---

## 10. Contribution 7: Reproducible experimentation

### 10.1 Experimental design: 17 runs

![Experimental design](figures/fig13_experimental_design.png)

{{table:t15_run_inventory}}

The pathway definitions are in Section 6.2.

**Controls.**
- **Shared settings:** every SFT arm uses the same recipe, learning rate, 2 epochs and seed. Arms differ only in data source and size.
- **Pathway difference:** the two GRPO pathways differ only in initialisation.
- **Model selection:** only final steps are evaluated, never best-of.

### 10.2 Protocol and integrity controls

| Control | Implementation | Where |
|---|---|---|
| **Run registry** | Every run is a row of one registry; configs are generated from it, and the gated launchers resolve a run only through the registry | `EXPERIMENT_REGISTRY.csv`, `scripts/make_configs.py`, `scripts/run_experiment.py` |
| **Pre-registration** | The learning-rate rule and the confirmatory comparisons were fixed before the runs they govern. The held-out plan was frozen and hashed before any held-out result existed | `EXPERIMENT_PLAN.md`, `final/freeze_v1.yaml` |
| **Checkpoint lock** | Weight hashes of all 11 evaluated model groups, the freeze hash and the code hash, written before the first evaluation | `results/run_records/a5_final/FINAL_CHECKPOINTS.json` |
| **Hash-locked verifier** | Verifier 2.0.0: every run records the hashes of the reward code, and golden cases are checked before use | `posttrain/verifier_lock.py`, `posttrain/verify.py` |
| **Run records** | Config, code and verifier hashes, checkpoint lineage and append-only metrics for every run | `results/run_records/` |
| **Audits** | A 217-check artifact audit, independent recomputation of every statistic, and a post-evaluation integrity check | `results/research_analysis/AUDIT.md` |
| **Tests** | 174 CPU tests | `tests/` |
| **Code identity** | The public release reorganises the package. Reversing its documented edits reconstructs, file for file, the code version recorded in the checkpoint lock, which ran the held-out evaluation. The training runs recorded earlier versions of that code | `docs/PACKAGE_MIGRATION.md`, `scripts/verify_code_lineage.py` |

---

## 11. Provenance and engineering summary

**Third-party components and inputs (not this project's work)**
- **Experimental setting:** the model, benchmarks, R1-Zero prompt and verifier-based SFT/GRPO recipe follow public materials, including the Stanford CS336 alignment materials (`THIRD_PARTY_NOTICES.md`).
- **Strict grader:** the Dr. GRPO grader (Apache-2.0), unchanged, at `posttrain/verifiers/drgrpo_grader.py`.
- **R1-Zero prompt:** the DeepSeek-R1-Zero format, with the text used in the MIT-licensed Dr. GRPO code.
- **SFT traces:** the public `garg-aayush/sft-cs336-assign5-datasets` (gpt-oss-120b on math12k), including its exact-match filtered subset.
- **Base model:** Qwen2.5-Math-1.5B (Apache-2.0).

**Built in this project (AI-assisted development, author-directed)**

| Area | Contribution |
|---|---|
| Data | Strict re-verification and independent confirmation of recovered labels; decontamination; nested matched subsets with a size control |
| Training | Response-masked SFT with exact accumulation; GRPO with online vLLM rollouts, group-relative advantages and constant length normalisation |
| Infrastructure | Dual-GPU engine/policy split; exact weight sync; fp32-master mixed precision; token-budget micro-batching; atomic lineage checkpoints; bit-exact resume |
| Stability | Rollout log-prob gate with failed-batch capture; incident guard; fork protocol from a verified saved state; per-step update-size diagnostics |
| Evaluation | Hash-locked verifier; frozen plan and checkpoint lock; strict/second/relaxed scoring; paired statistics with Holm families; 217-check audit; independent reproduction; 174 tests |

---

## 12. Discussion and limitations

**What the evidence supports**
- **SFT followed by GRPO is an effective recipe at this scale.** GRPO's gain is a gain in solving problems. This is the strongest result: pre-registered, large relative to question-sampling noise, and stable across three scoring rules.
- **Verifier-based recovery of teacher traces did not improve SFT here.** On GSM8K it was associated with lower accuracy. This is a statement about this teacher, filter and verifier, not about data cleaning in general.
- **Evaluation design changes conclusions:**
  - a 47.9 pp prompt effect;
  - relaxed scoring that adds 27.1 pp to the base model under r1_zero and 11.0 pp to the 128-example SFT arm;
  - a second verifier that adds 14.4 pp to one baseline but at most 0.6 pp to the best model.

**Limitations**
1. **One training seed per arm.** CIs cover question sampling only; GRPO-seed variance is unknown. The design called for three seeds of the matched-512 arms; one was run.
2. **The registered direct-GRPO run failed.** C4 is unanswered; A1/A2 are not a clean direct run.
3. **Prompt sensitivity.** Trained models are evaluated only with r1_zero, and few-shot or chat-template prompts were not tried.
4. **One verifier for reward and metric.** Reward hacking would raise both together. The second verifier moves GRPO-B by at most 0.6 pp and relaxed scoring by at most 0.9 pp, which is reassuring but partial. The format check is lenient, and truncation at 1,024 tokens is scored as wrong.
5. **Possible pretraining contamination.** The pipeline is clean with respect to GSM8K test, but Qwen2.5-Math was pretrained on a large mathematical corpus. GSM8K test also had prior use in the project's history (no hyperparameter was inherited from it).
6. **Numerical effects.** The engine and policy log-probabilities differ slightly (mean gap about 0.004), so samples are slightly off-policy. Evaluation is reproducible on the same engine, GPU type and settings (0 of 1,500 answers changed on a dev re-evaluation), not across setups.
7. **Design confounds.** Fixed epochs confound size with steps; recovered traces differ in answer style and problem mix; one learning rate was used for all arms; no KL penalty; one update per batch.
8. **External validity.** One model family and size, one teacher, one RL pool (6,832 problems), two English short-answer benchmarks, greedy decoding with a 1,024-token budget.

**What is not claimed**
- No novel GRPO algorithm.
- No measured compute acceleration: the filtering intervention was not run.
- No multi-seed robustness.
- No production deployment.

Non-significant differences are reported as "not detected", never as equivalence.

---

## 13. Conclusion and future work

On Qwen2.5-Math-1.5B, SFT followed by 100 GRPO steps raised held-out accuracy to 86.1% on GSM8K and 72.5% on MATH. GRPO contributed a pre-registered, reasoning-driven gain of about 12–13 pp over its SFT parent. Against the motivating hypothesis, verifier-recovered SFT data performed worse than exact-match data on GSM8K in this single-seed design. The pre-registered direct-GRPO run collapsed from a low-entropy plateau; the correctness gates stopped it before a corrupted update, and it is reported as a failed experiment.

The project is complete as scoped. Optional follow-ups, in priority order:
1. **Seed replication** of the 512-example SFT arms and of GRPO-B (sketched, not executed, in `PHASE2_PLAN.md`).
2. **Clean direct-GRPO runs**, with failures counted.
3. **Compute reduction:** measured zero-advantage filtering, and a pre-registered non-inferiority test of early stopping near step 25.
4. **Verifier robustness:** stricter format checks and longer generation budgets.

---

## 14. Reproducibility appendix

Full instructions: `report/REPRODUCIBILITY.md`.

| Artifact | Location | Integrity |
|---|---|---|
| Frozen evaluation plan | `final/freeze_v1.yaml` | sha256 `ff0f0ce5…a453` (full hash in `REPRODUCIBILITY.md`) |
| Held-out item-level outputs | private run folders; not in the public repository | `FINAL_EVAL.sha256`, 58 files |
| Checkpoint lock | `results/run_records/a5_final/FINAL_CHECKPOINTS.json` | 11 weight hashes, freeze and code hash |
| Training run records | `results/run_records/` (`run.json`, `DONE.json`); full run folders are private | `SESSION_A/B.sha256`, `RECOVERY.sha256` |
| Generated report and statistics | `results/final_eval/` | reproduced independently, 0 mismatches |
| Research analysis, figures, audit | `results/research_analysis/` | `PHASE1_FREEZE.sha256` (62 files) |
| Post-evaluation integrity | `results/research_analysis/post_final_integrity/` | `TRAINED_RUNS_UNCHANGED`; 43 checks, 0 failed |
| Code identity | `docs/PACKAGE_MIGRATION.md`, `scripts/verify_code_lineage.py` | the release reconstructs the locked evaluation code |
| This report's tables and figures | `report/tables/`, `report/figures/` | generated by `report/scripts/` from the frozen files |

**Software and compute**
- **Software:** PyTorch 2.5.1, Transformers 4.51.3, vLLM 0.7.2, math-verify 0.7.0 (`requirements-cloud.lock.txt`).
- **Hardware:** 2× NVIDIA RTX A6000 (48 GB).
- **Total GPU time:** about 34 GPU-h for training and tuning (all 17 runs, including the failed GRPO-A) and about 3 GPU-h for held-out evaluation.

**Determinism.** GPU training is not bit-deterministic: the A1 branch reproduced the original step 81 exactly from the saved state, and its later steps diverged from the original run.

### Source of every number

| Section | Source file |
|---|---|
| 1, 6, 9 | `results/final_eval/report.json`, `results/research_analysis/{consolidated_results,statistical_comparisons}.csv`, `results/research_analysis/data/*.csv`, `analysis.json` |
| 4 | `data/stage0/MANIFEST.json`, `data/stage0/subsets/`, `EXPERIMENT_PLAN.md` §1, `results/verifier_v2/crosscheck.json` |
| 5.2, 10.1 | run.json / DONE.json of each training run; `final/freeze_v1.yaml` |
| 8 | `results/grpo_a_incident_analysis/REPORT.md`, `GRPO_A_INCIDENT.md`, `results/research_analysis/data/grpo_dynamics.csv` |
| Audit | `results/research_analysis/AUDIT.md`, `audit.json`, `post_final_integrity/` |

**Audits of this report**
- `report/AUDIT_KEY_CLAIMS.md` recomputes each key claim from its named source.
- `report/AUDIT_REPORT_NUMBERS.md` checks that every number in the deliverables exists in the sources at the stated rounding.
- Two small errors found in the frozen Phase I prose are recorded in `results/research_analysis/ERRATA_2026-11-27.md`. This report uses the corrected values.
