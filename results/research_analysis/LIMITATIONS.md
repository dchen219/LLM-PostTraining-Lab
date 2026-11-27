# Limitations

These limit how far the findings in `RESEARCH_FINDINGS.md` can be taken. The most consequential come first.

## 1. One training seed per arm

Every checkpoint was trained once. McNemar tests and CIs treat the trained weights as fixed and quantify only the sampling of evaluation questions, so they answer "would this checkpoint score differently on other questions?", not "would retraining give a different answer?".

**What we know about seed variance.** It comes from one pair: SFT-reverified_n512 trained with seed 0 and with seed 100, on identical data and learning rate. The pair differed by 4.1 pp on GSM8K dev (p = 1e-4) and 3.2 pp on MATH dev (n.s.).
- With 1 degree of freedom, the implied seed SD of a contrast is about 4.0 pp (GSM8K) and 2.5 pp (MATH).
- The true value could plausibly be several times smaller or larger.
- No GRPO seed replicate exists. GRPO-seed variance is unknown.

**Consequences**

| Effect | Size | Status under seed noise |
|---|---|---|
| C3 (GRPO over SFT parent) | 12–13 pp | Large relative to any plausible seed SD |
| Prompt effect | 41–48 pp | Large relative to any plausible seed SD |
| SFT 128 → 512 | 11 pp | Large relative to any plausible seed SD |
| C1, C2 on GSM8K | 9 pp | Detected beyond question noise; about 2 combined SEs once seed variance is added. Suggestive, not robust. |
| All MATH data-quality and scaling effects | 0–2 pp | Within seed noise |
| Post-hoc direct-vs-sequential MATH gap | 2 pp | Within seed noise |

**The pre-registered design asked for more seeds.** It planned 3 seeds for the matched-512 arms and seed means with SD (EXPERIMENT_PLAN.md sections 3–4). Tier A ran one seed. The seed-level questions in that plan therefore remain unanswered.

## 2. The pre-registered direct-GRPO run failed

GRPO-A_base_s2026 was stopped by the rollout log-probability gate at step 98, after visible instability at steps 96–97. As a result:
- confirmatory comparison C4 (pathway A vs base) has no result;
- the registered direct-vs-sequential question is unanswered.

A1 and A2 are post-hoc branches from a run that later collapsed. Their results describe "GRPO-A's step-80 state plus 20 steps", not a clean direct run. One collapse in one seed says nothing about how often collapse happens, and neither the trigger nor the frequency was isolated.

## 3. Prompt sensitivity

- The base model's accuracy changes by 41–48 pp between the two frozen prompts.
- SFT and GRPO models are evaluated only with r1_zero, the prompt they were trained on.
- The base model is compared using its better prompt (question-only), as pre-registered. Other prompts (few-shot, chat template) were not tried; they could narrow or widen the SFT-vs-base gap.
- The SFT-vs-base comparison crosses prompts and output formats. Under relaxed scoring, which credits misformatted right answers, the re-verified SFT arms' GSM8K advantage over base question-only falls to between −0.8 and +3.0 pp.
- "SFT improves reasoning over the base model" is therefore weaker than "SFT improves strict accuracy".

## 4. Verifier limitations

**One verifier for reward and metric.** The strict verifier 2.0.0 is both the GRPO reward and the evaluation metric. Reward hacking of verifier quirks would raise both together and stay invisible in the metric. The second verifier (math-verify) and relaxed scoring are partial checks; both moved GRPO-B by 0.6 pp or less.

**Known false rejects.** Stage 0 found that 118 of 291 rejected traces are equal under an independent check, and conflicting units are ignored. Arms whose outputs use non-canonical answer styles are penalised. About 2 pp of the C2 GSM8K gap disappears under the second verifier.

**Lenient format check.** A2's doubled `</think>` passes it (4,398 of 4,406 MATH responses with the doubled tag counted as formatted). Format compliance overstates protocol adherence for that model.

**Truncation is scored as wrong.** Responses cut at 1,024 tokens count as incorrect. This penalises GRPO-B, which writes longer answers: truncation accounts for 21% of its MATH errors.

## 5. Possible dataset contamination

- **Our own pipeline is clean on GSM8K.**
  - Stage 0 found 0 exact, 0 13-gram and 0 fuzzy overlaps between GSM8K test and our SFT/RL data.
  - In this audit, held-out item ids are disjoint from the development and SFT ids.
- **MATH val is less clean.** It had 6 exact and 119 fuzzy near-duplicates in the SFT source, which were removed from training. Fuzzy matching can miss paraphrases.
- **Pretraining contamination cannot be ruled out.** Qwen2.5-Math was pretrained on a large mathematical corpus.
  - The base model reaches 58.6% on MATH val with a bare question-only prompt, far above its 7.2% on our difficulty-selected MATH dev set.
  - Some memorisation of public GSM8K and MATH items is plausible.
- **What contamination would and would not affect.**
  - It inflates absolute accuracies.
  - It would mainly affect between-arm contrasts if arms differ in how much they elicit memorised answers. That is possible, and it is not tested.
- **GSM8K test has prior history.** It was used for hyperparameter selection by the tutorial and by earlier runs on this project. No hyperparameter here was inherited from those results, but the benchmark is not pristine.

## 6. Numerical and engine effects

- **Weights matched exactly.** Exact weight-sync checks passed at every step of every run, with 0 mismatched parameters.
- **Engine and policy still differ during rollouts.** The rollout log-prob gap is non-zero (mean about 0.004, max token gap up to 20 nats at GRPO-A's step 97). Policy-gradient estimates therefore use slightly off-policy samples.
  - The gate bounds the mean gap, not the tail.
  - Whether the gap contributed to GRPO-A's collapse or merely reflected it cannot be separated from these logs.
- **Evaluation determinism holds only within one setup.** Re-evaluating identical weights on dev gave 0 of 1,500 changed answers under the frozen settings.
  - On historical data a different setup changed 39 of 1,319 GSM8K answers. Results are reproducible only on the same engine version, GPU type and settings (vLLM 0.7.2, A6000, prefix caching off).
- **fp32 master weights** were used throughout, because pure-bf16 updates would have left most weights unchanged (Stage 0). Conclusions may not hold for other precision setups.

## 7. Design confounds

- **Epochs fixed at 2.** Data size is confounded with optimizer steps (32 steps at n = 512).
- **Recovered-trace selection.** The re-verified pools are the exact pool plus recovered traces. Those traces were selected because their answers failed string-matching, which may correlate with problem type and answer style. Label correctness, answer format and problem mix are not separated.
- **Learning rates.** They were tuned on dev for SFT from one 512-example re-verified run, and for GRPO from 30 base-model steps, then applied to every arm. Arms other than the tuned configuration may be at a sub-optimal rate.
- **No GRPO KL penalty, one update per batch.** The ratio is 1, so clipping never acts. Results are specific to this GRPO variant.
- **Only the final step was evaluated on held-out data.** This was by design, to prevent selection. Claims about early stopping rest on dev curves only.

## 8. External validity

- One model family and size (Qwen2.5-Math-1.5B, a math-specialised base).
- One teacher for SFT traces (gpt-oss-120b on math12k).
- One verifier and one RL training pool (6,832 problems).
- Two English benchmarks of short-answer math.
- Greedy decoding with a 1,024-token budget. Sampling, larger budgets or majority voting could change the rankings, especially for longer-answer models such as GRPO-B.

The direction "exact-match beats re-verified" is a statement about this teacher, this filter and this verifier. It should not be read as a general claim about data cleaning.

## 9. Post-evaluation evidence (resolved)

The trained-run integrity check after evaluation was supplied on 2026-11-27 and verified (`AUDIT.md` section 4). It is no longer a limitation.

What remains is attested rather than re-runnable:
- the server-side `sha256sum -c` itself, whose pass is recorded by the recovered log;
- the launcher log `final_eval.log`, which was not recovered.
