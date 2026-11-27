# Executive summary: LLM Reasoning Post-Training with SFT and GRPO

**Project.** The experimental framework investigates mathematical reasoning through supervised fine-tuning and reinforcement learning with verifiable rewards. Existing mathematical verification components and prompt templates are incorporated with their original attribution and licensing, as documented in THIRD_PARTY_NOTICES.md. It was developed with AI coding assistance under the author's direction.

**Setup.** Qwen2.5-Math-1.5B, post-trained with supervised fine-tuning (SFT) on teacher reasoning traces and with GRPO (reinforcement learning with verifiable rewards) on 2× RTX A6000.
- **Scale:** 17 documented training runs.
- **Evaluation:** 11 model groups, each evaluated once on held-out GSM8K test (n = 1,319) and MATH val (n = 5,000).
- **Protocol:** the evaluation plan was frozen and hashed before any held-out result existed.

Full report: `TECHNICAL_REPORT.md` / `TECHNICAL_REPORT.pdf`.

## Headline

**Sequential GRPO improved held-out accuracy over its own SFT parent by +13.4 pp on GSM8K and +11.7 pp on MATH** (pre-registered confirmatory test, Holm p = 1.9e-25 and 5.7e-73).

It is not a formatting effect:
- The parent was already formatted correctly on 99.6% of GSM8K items.
- 234 of the 237 GSM8K items GRPO newly solved were well-formatted but wrong before.
- The gain persists under relaxed extraction and a second verifier (+11.2 / +10.5 pp).

| Model | GSM8K | MATH |
|---|---|---|
| Base, R1-Zero prompt | 17.7 | 17.4 |
| Base, question-only prompt (strongest baseline) | 65.7 | 58.6 |
| SFT parent (re-verified data, 512 examples) | 72.7 | 60.8 |
| Best SFT arms (exact-match data, 512 / 3,420 examples) | 81.3 | 61.9 / 62.5 |
| **GRPO-B: SFT → 100 GRPO steps** | **86.1** | **72.5** |
| GRPO-A: direct GRPO from base (pre-registered) | FAILED at step 98 | — |

## Seven contributions

1. **Reasoning data curation and re-verification.**
   - **Input:** used the public gpt-oss-120b trace dataset (4,836 traces; not generated here).
   - **Re-verification:** re-scored every trace with a strict math verifier, recovering 1,049 traces the dataset's exact-match filter had rejected (4,545 vs 3,496).
   - **Cleaning:** decontaminated against the evaluation sets.
   - **Controls:** built nested, matched subsets (128 to 4,423 examples) with a size control.
2. **Supervised fine-tuning.** Response-masked SFT with exact gradient accumulation; learning rate chosen on dev data by a rule fixed in advance.
3. **Online RL with verifiable rewards.**
   - GRPO with online vLLM rollouts (32 prompts × 8 samples), binary verifier rewards and group-mean advantages.
   - Clipping is inactive at one update per batch, so the update is REINFORCE with a group baseline.
4. **Dual-GPU training and inference architecture.** A PyTorch policy and a vLLM rollout engine on separate GPUs, with:
   - bit-exact weight synchronisation every step;
   - fp32 master weights with bf16 autocast;
   - token-budget micro-batching;
   - atomic lineage checkpoints;
   - bit-exact resume (on CPU).
5. **Numerical stability and checkpoint recovery.**
   - A rollout log-probability gate stopped the pre-registered direct-GRPO run after its collapse (reward 0.62 → 0.05 in one update), before any corrupted update.
   - The failed run was preserved.
   - Two post-hoc branches from its step-80 state (same and halved learning rate) completed without collapse. They are diagnostics only, not a completion of the registered run.
6. **Statistical benchmark evaluation.**
   - Frozen plan, checkpoint lock, strict / second-verifier / relaxed scoring.
   - Exact McNemar tests with Holm families.
7. **Reproducible experimentation.**
   - A run registry, a hash-locked verifier and per-run records of config, code and verifier hashes.
   - A 217-check audit, an independent reproduction of every statistic, and 174 tests.

## Further findings

**Confirmatory**
- **Data quality:** exact-match SFT data beat verifier-recovered data on GSM8K: −9.3 pp (full sets) and −8.6 pp (512 examples). On MATH: −2.0 pp and not detected. This is the opposite of the motivating hypothesis. It is robust to question sampling but rests on one training seed per arm.

**Confirmatory-secondary**
- SFT with 512 or more examples beats the base model's best prompt by 6–16 pp (GSM8K) and 2–4 pp (MATH).
- SFT with 128 examples is worse than it.

**Descriptive and exploratory**
- The prompt alone moves the base model by +47.9 pp on GSM8K.
- SFT returns flatten after 512 examples.
- On dev data, GRPO's measurable gain came in the first 25 of 100 steps, as zero-advantage groups rose to about two-thirds.

## Limitations

- **Seeds:** one training seed per arm, so intervals cover question sampling only.
- **Missing comparison:** the registered direct-vs-sequential comparison is unavailable.
- **Verifier:** one verifier serves as both reward and metric.
- **Contamination:** possible pretraining exposure to GSM8K/MATH.
- **Scope:** one model, one teacher, greedy decoding at 1,024 tokens.
- **Not claimed:**
  - a novel algorithm;
  - a measured speedup;
  - multi-seed robustness;
  - production deployment.

## Provenance

- **Third-party inputs:**
  - the SFT traces: `garg-aayush/sft-cs336-assign5-datasets`, gpt-oss-120b;
  - the strict grader: Dr. GRPO, Apache-2.0;
  - the R1-Zero prompt;
  - the base model: Qwen2.5-Math-1.5B.

  See `THIRD_PARTY_NOTICES.md`.
- **This project:** the system, experiments and analysis, developed with AI coding assistance (Claude Code) under the author's direction.
