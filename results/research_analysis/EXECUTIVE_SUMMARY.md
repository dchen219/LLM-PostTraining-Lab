# Executive summary

Post-training Qwen2.5-Math-1.5B with SFT and GRPO, evaluated once on held-out GSM8K test (n = 1,319) and MATH val (n = 5,000) under a plan frozen before any held-out result existed.
Single training seed per arm. CIs and p-values cover sampling of evaluation questions, not training-seed variance.

## Audit

- **0 of 217 checks failed.** Every number in the generated report was reproduced by independent code (0 mismatches).
- Checkpoint, dataset, verifier, engine and code identity all match the lock and the frozen plan.
- **Post-evaluation integrity closed (2026-11-27):** the server check printed `TRAINED_RUNS_UNCHANGED`. Its manifest (349 files) covers every locked checkpoint, and every entry matches a hash recorded at training time (43 checks, 0 failed).

## Main results (strict accuracy)

| | GSM8K | MATH |
|---|---|---|
| Base, r1_zero / question-only prompt | 17.7 / 65.7 | 17.4 / 58.6 |
| Best SFT arm (exact-match, 512 or 3,420) | 81.3 | 61.9 / 62.5 |
| GRPO-B (SFT re-verified 512 + 100 GRPO steps) | **86.1** | **72.5** |
| GRPO-A (direct, pre-registered) | FAILED at step 98 | – |

## Verified findings

**Confirmatory**
1. **GRPO improves on its SFT parent (C3):** +13.4 pp GSM8K and +11.7 pp MATH (Holm p < 1e-24). The gain is in reasoning: 234 of 237 newly solved GSM8K items were already well-formatted but wrong.
2. **Exact-match SFT data beat re-verified data on GSM8K (C1 −9.3 pp, C2 −8.6 pp).**
   - On MATH the gap is −2.0 pp (C1) and not detected (C2).
   - This is the opposite of what recovering traces was meant to achieve.
   - It is robust to question sampling but only about 2 SEs once an estimate of training-seed variance is added.

**Confirmatory-secondary**

3. **SFT with 512 or more examples beats the base model's question-only prompt** by 6–16 pp (GSM8K) and 2–4 pp (MATH).
   - Much of the re-verified arms' GSM8K margin comes from making answers extractable.
   - **SFT with 128 examples is worse** than question-only.

## Negative and null results

- **Direct vs sequential GRPO:** the pre-registered run failed (C4 unavailable), so the registered question is unanswered.
  - Post-hoc branches A1/A2 match GRPO-B on GSM8K (no difference detected) and trail it by about 2 pp on MATH.
- **Re-verified (recovered) training traces did not help;** they hurt on GSM8K.
- **No detectable SFT gain beyond 512 examples.** 2,048 → 4,423 was lower, within seed noise.
- **GRPO dev gains stopped after about 25 of 100 steps.** About 75% of GRPO compute bought no measurable dev improvement. Two-thirds of rollout groups were all-correct or all-wrong, so they carried no signal.

## Engineering contributions (code written by Claude at the user's request; see EXPERIMENT_PLAN.md section 0)

- **Freeze-then-lock evaluation:**
  - a hashed plan;
  - a checkpoint lock written before evaluation;
  - weights checked against training lineage;
  - refusal on mismatch.
- **Exact weight-sync gate and rollout log-prob gate.** The rollout gate caught GRPO-A's collapse before a corrupted update.
- **Incident and recovery protocol:** the failed run was preserved, and branches were run in new folders without substitution.
- **Read-only audit and independent reproduction scripts:** `final_audit.py`, `research_analysis.py`, `research_figures.py`.

## Recommendations for the technical report

1. **Lead with C3** (GRPO over SFT) and the formatting-vs-reasoning decomposition. These are the strongest results.
2. **Report C1/C2 as "exact-match data scored higher in one seed"**, with the seed sensitivity shown (figure 2), not as a general claim about data cleaning.
3. **State the GRPO-A failure plainly.** Keep A1/A2 in a separate post-hoc section.
4. **Present SFT gains against the question-only baseline,** with relaxed accuracy alongside.
5. **Next experiments, in priority order:**
   - 2–3 more seeds for the matched-512 arms and GRPO-B;
   - a clean direct-GRPO run;
   - zero-advantage filtering or early stopping at about 25 steps. The pre-registered Stage 6 trigger was met.

Files:
- `AUDIT.md`
- `RESEARCH_FINDINGS.md`
- `LIMITATIONS.md`
- `consolidated_results.csv`
- `statistical_comparisons.csv`
- `figures/`: 10 figures as PNG and PDF, with plotting data in `figures/data/`
- `data/`
- `scripts/research_analysis.py`
- `scripts/research_figures.py`
