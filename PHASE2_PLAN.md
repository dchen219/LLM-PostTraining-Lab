# Phase II plan: DRAFT, not pursued

> **Status (2026-11-27): not pursued.** The user decided that Phase I is the complete scope of the project. This document is kept only as a sketch of optional future work (technical report §15). Nothing in it was registered, configured or run.
>
> **Correction:** both pre-registered learning-rate rules selected **1e-5**, for SFT and for GRPO (run.json of every Tier A run). The configs' default values are not what ran. The "2e-5 / 5e-6" stated in §2 below is wrong.

Date: 2026-11-27. Status: **proposal, not pursued**. Nothing here is registered, configured or run. No registry row, config or code has been changed.

- Phase I is frozen: `results/research_analysis/PHASE1_FREEZE.md`. Manifest `PHASE1_FREEZE.sha256` = 466b2bd8…c145.
- Phase I inputs: `PHASE1_INPUTS.sha256` = 6cb13f12…4940.

Phase II answers the questions Phase I left open, using registry rows that were written before any held-out result wherever possible. Section 9 lists what the user must decide before any of it becomes binding.

## 1. What Phase I left open

| Open question | Phase I evidence | What would close it |
|---|---|---|
| **Q1.** Is exact-match data really better than re-verified data, or is that seed noise? | C2 GSM8K −8.6 pp, C1 −9.3 pp, one seed each. The one same-data seed pair moved 4.1 pp, so C1/C2 sit about 2 combined SEs from zero | More seeds of the matched-512 arms |
| **Q2.** Does GRPO's gain over its SFT parent hold across GRPO seeds? | C3 +13.4 / +11.7 pp, one GRPO seed. GRPO-seed variance is unknown | GRPO-B seeds 2027, 2028 |
| **Q3.** Direct (base → GRPO) vs sequential (base → SFT → GRPO) | The registered run failed at step 98; C4 unavailable; A1/A2 are post-hoc only | Fresh, unchanged GRPO-A seeds, with failures counted, not replaced |
| **Q4.** Does GRPO need 100 steps? | Dev gains stopped after step 25 in GRPO-B (dev only; held-out saw step 100 only) | Held-out evaluation of step-25 checkpoints, on new seeds |
| **Q5.** Can the wasted compute be cut without changing training? | 65–72% of groups had zero advantage after step 25; update took 54% of step time. The Stage 6 trigger was met | The registered `GRPO-X_filter_compute` run |

## 2. Rules carried over unchanged

- Every gate stays as it is: exact weight sync, the 0.05 rollout log-prob gate, verifier 2.0.0 lock, code hash recorded.
- **A run that fails a gate is FAILED.** It is never resumed, replaced or re-seeded. Post-hoc branches may be run as diagnostics, but are never evaluated on held-out data for a registered comparison.
- No checkpoint, hyperparameter, stopping step or seed is chosen with held-out data. Final step only, except the pre-registered step-25 checkpoints (Q4).
- Learning rates stay as tuned in Phase I (SFT 2e-5 per the registry configs, GRPO 5e-6). No re-tuning.
- Phase I files are not edited (`PHASE1_FREEZE.md`). New outputs go under `results/phase2/`; new server folders go under new run names.
- Claude prepares files; the user runs everything on the server. No GPU work, registry edit, commit or push happens without the user's approval.

## 3. Held-out reuse

GSM8K test (1,319) and MATH val (5,000) have now been looked at once. Reusing them is defensible only under these safeguards:

1. **The hypotheses are fixed now, before any Phase II run.** They come from Phase I, but they test *new* seeds whose results do not exist yet.
2. **Confirmatory tests use only the new seeds.** The Phase I seed (s0 / s2026) motivated the question, so including it would bias the test toward replication. All-seed pooled estimates are reported, labelled descriptive.
3. **The checkpoints are chosen by the registry alone.** A `freeze_v2.yaml` is written before the first Phase II held-out evaluation, in the same way as `freeze_v1.yaml`.
4. **Anchor re-evaluation.** Comparisons against Phase I checkpoints (base question-only, the SFT parent) reuse the Phase I item files only if a Phase II re-evaluation of the parent `SFT-reverified_n512_s0` gives 0 changed answers on both sets. If any answer changes, every comparison partner is re-evaluated under the Phase II setup and those outputs are used.

A fresh benchmark (for example a perturbed-GSM8K set) would add external validity, but needs data acquisition and a new decontamination pass. It is listed as optional (E8).

## 4. Experiments

GPU-h = 2 × wall-clock on the 2× A6000 node, using measured Phase I costs.

| ID | Runs | Registry | Answers | GPU-h | Tier |
|---|---|---|---|---|---|
| **E1** | `SFT-exact_n512_s1, _s2`, `SFT-reverified_n512_s1, _s2` | exist, HOLD_B | Q1 | 1.8 | Core |
| **E2** | `SFT-exact_n512_s3, _s4`, `SFT-reverified_n512_s3, _s4` | **new rows** (same generator, seeds 3, 4) | Q1, with usable power | 1.8 | Recommended |
| **E3** | `GRPO-B_sft512_s2027, _s2028` (parent `SFT-reverified_n512_s0`, unchanged) | exist, HOLD_B | Q2, Q3 partner | 14.7 | Core |
| **E4** | `GRPO-A_base_s2027, _s2028`, config identical to the failed s2026 except seed | exist, HOLD_B | Q3 | about 13.6 if both complete | Core |
| **E5** | Step-25 checkpoints of E3 (see section 6) | instrumentation change or 2 new rows | Q4 | 0 extra training (option a) / 3.3 (option b) | Recommended |
| **E6** | `GRPO-X_filter_compute` (GRPO-B s2026 with `group_filter: compute_only`) | exists, GATED; trigger met | Q5 | 4.5–7.3 | Core (dev only) |
| **E7** | `SFT-exact_full_s1, _s2`, `SFT-reverified_full_s1, _s2`, `SFT-reverified_sizecontrol_s1, _s2` | exist, HOLD_B | Q1 at full size, plus composition vs volume | 12.2 | Optional |
| **E8** | Evaluation only: GRPO-B s2026 with 2,048 new tokens (truncation); base model few-shot prompt; a fresh benchmark | new, exploratory | Limitations 3, 4, 8 | about 1–3 | Optional |
| – | Held-out evaluation of the new checkpoints (12–14, plus the anchor) | `freeze_v2.yaml` | all | about 4 | Core |

**Totals.** Core plus recommended (E1–E6 and evaluation) is about 40–44 GPU-h, or about 20–22 h of wall-clock on the node. Optional work adds about 13–15 GPU-h.

**Not proposed, and why**
- **Difficulty-aware sampling.** Its trigger also needs prompts to be revisited. GRPO-B drew 3,200 prompts from a 6,832-prompt pool, so none were (to be confirmed from the rollouts).
- **Off-policy reuse.** Its trigger is rollout time above 50% of step time. Rollouts took at most 46%, because the update took 54%.
- **`renormalize` filtering.** It changes the effective learning rate and needs its own learning-rate control.
- **Raw-data arms, EI and Tier C.** No Phase I result motivates them over the above.
- **More GRPO-A branches from step 80.** They answer nothing about the registered question.

## 5. Hypotheses and analysis (to be frozen as `freeze_v2.yaml` before the first held-out evaluation)

Statistics are the same as Phase I unless stated: strict accuracy primary; exact McNemar; Wald CI on the paired difference; Holm within each family. Second-verifier and relaxed accuracy are reported alongside every result. "Supported" means Holm p < 0.05 **in the stated direction**. Phase I's direction-agnostic label is not reused.

**F1. Data-quality replication (Q1), confirmatory, m = 2 (GSM8K, MATH)**
- Unit: training seed. d_s = acc(re-verified_n512_s) − acc(exact_n512_s) on held-out, for the new seeds s1–s4 (E1+E2).
- Test: two-sided one-sample t-test on d_s (df = 3). Report the mean with its 95% t-interval.
- Interpretation:
  - "Exact-match advantage replicated": Holm p < 0.05 with mean d < 0.
  - "Re-verified advantage": Holm p < 0.05 with mean d > 0.
  - Otherwise "not detected". This is **not** evidence of no difference: the interval half-width is about 6.6 pp on GSM8K, so no equivalence claim is possible.
- If E2 is not approved, F1 falls back to s1 and s2 only. "Replicated" then requires both new seeds individually significant (McNemar p < 0.05) in the Phase I direction, and no seed-level CI is reported (df = 1).
- Descriptive: per-seed McNemar tests; the mean over all 5 seeds (including s0); the seed SD of each arm and of the contrast, which replaces Phase I's 1-df heuristic.

**F2. GRPO over its SFT parent, replication (Q2), confirmatory, m = 4**
- `GRPO-B_s2027` and `_s2028`, each vs `SFT-reverified_n512_s0`, on 2 benchmarks. Item-level McNemar.
- Supported: Holm p < 0.05, positive, and ≥ 1.4 pp on GSM8K (the Phase I threshold).
- Descriptive: the GRPO-seed SD across s2026–s2028.

**F3. Direct GRPO (Q3), confirmatory, two families of m = 4 each (2 seeds × 2 benchmarks)**
- F3a: `GRPO-A_base_s` vs base question-only.
- F3b: `GRPO-A_base_s` vs `GRPO-B_s` (same GRPO seed): direct vs sequential.
- **A seed that fails a gate counts p = 1 in both families**, as C4 did in Phase I. It is not replaced.
- Descriptive: completion count k of 3 registered direct runs (s2026–s2028), with a Clopper–Pearson interval. With 3 runs this interval is necessarily wide (1 of 3 gives about 1%–91%), so no collapse-rate claim is made.
- A1/A2 remain post-hoc and are not part of F3.

**F4. Early stopping (Q4), confirmatory non-inferiority, m = 4 (2 seeds × 2 benchmarks)**
- Contrast: step 25 − step 100 within each E3 run.
- H0: difference ≤ −2.0 pp. One-sided paired test.
- Non-inferior: Holm-adjusted p < 0.05, equivalently a lower 95% one-sided bound above −2.0 pp.
- The 2.0 pp margin is fixed here.
- Phase I dev curves predict about 0 pp on GSM8K and about −2 pp on MATH. MATH non-inferiority is therefore likely to fail, and that would be reported as such.
- Descriptive: GPU-h saved; accuracy per GPU-h.

**E6 (Q5), dev only, not a held-out comparison**
- Primary: mean `update_s` per step, filtered vs Phase I GRPO-B s2026.
- Guard: the gradient is identical by construction (unit-tested). GPU non-determinism means trajectories will not match bit-for-bit. Dev accuracy at steps 25–100 is reported, and a step-100 difference beyond the dev seed noise (4 pp) is investigated as a possible bug, not reported as a filtering effect.

**Power**, using the seed SD heuristic from Phase I (GSM8K contrast SD about 4.1 pp, MATH about 2.6 pp; uncertain):

| True GSM8K effect | F1 with 4 new seeds (t-test) | Fallback: both of 2 seeds significant |
|---|---|---|
| 9 pp (Phase I estimate) | 0.82 | 0.90 |
| 6 pp | 0.51 | 0.68 |
| 4.5 pp | 0.33 | 0.51 |

- The Phase I estimate is probably inflated (winner's curse). A non-replication would therefore be uninformative unless the true effect is large. This is the main reason to add E2. Seven seeds would give 0.89 power at 6 pp, if the user wants a stronger test (4 more SFT runs, about 1.8 GPU-h).
- **F2:** with an effect of 12–13 pp, power is near 1 unless GRPO-seed variance is very large.
- **F4:** if the true difference is 0, GSM8K power is about 0.7 per seed. This is approximate, from Phase I discordance rates.

## 6. Step-25 checkpoints (E5): two options

**(a) Recommended: instrumentation in E3.**
- Add `train.keep_checkpoints: [25]`. At step 25 it exports a full checkpoint with its own `lineage.json` and `weights_sha256` into `checkpoint_step25/`.
- No update, random number or data-order change. Unit test: a 30-step CPU run with and without the option gives bitwise-identical final weights. The readiness run is repeated.
- Gives a within-run comparison (same trajectory) on 2 new seeds at no extra training cost.

**(b) Fallback: separate runs.**
- `GRPO-B25_sft512_s2027, _s2028` with `steps: 25`. The learning rate is constant after 5 warmup steps, so the schedule is identical. No code change.
- GPU non-determinism means each one is a different trajectory from its 100-step partner. This adds run-to-run noise to F4.

## 7. Order and stopping points

Each block is approved and launched separately. A block's dev results are reviewed before the next block starts. Nothing in a block is changed by an earlier block's results; failures are reported, not worked around.

| Block | Contents | GPU-h | Notes |
|---|---|---|---|
| 0 | Approvals (section 9); registry edits; configs from `scripts/make_configs.py`; option (a) code and tests if chosen; Phase II plan appended to `EXPERIMENT_PLAN.md` and hashed | 0 | Claude prepares; user approves |
| 1 | E1 (+E2) | 1.8–3.6 | Cheapest; answers the most fragile Phase I claim |
| 2 | E3 (+E5a) | 14.7 | |
| 3 | E4 | about 13.6 | A failed seed ends that seed only |
| 4 | E6 | 4.5–7.3 | Dev only |
| 5 | `freeze_v2.yaml` → lock → anchor check → held-out evaluation → report | about 4 | Same scripts as Phase I, new run root |
| 6 (optional) | E7, E8 | 13–15 | Only after block 5 is reported |

If server time runs short, stop after block 2 and evaluate what exists. F1 and F2 are the most valuable families. F3 and F4 are then reported as not run, not as null.

## 8. Failure handling

| Event | Response |
|---|---|
| A gate stops a run | FAILED.json; incident record as for GRPO-A; never resumed or replaced; counts p = 1 in its family |
| A run crashes without a gate (OOM, node loss) | May be resumed from its last state under the existing resume rules; recorded in the run's notes; not a failure |
| Anchor re-evaluation changes answers | Re-evaluate all comparison partners under the Phase II setup; Phase I item files are not used for Phase II tests |
| E6 diverges from GRPO-B beyond 4 pp dev | Stop interpreting E6; open a bug investigation; no held-out use |

## 9. Decisions needed from the user

1. **Scope.** Core (E1, E3, E4, E6), plus recommended (E2, E5), plus optional (E7, E8)? Recommendation: core and recommended, about 40–44 GPU-h.
2. **Held-out reuse** under section 3 (same sets, `freeze_v2`, new seeds only, anchor check)? Recommendation: yes.
3. **Step-25 checkpoints:** option (a) instrumentation or option (b) separate runs? Recommendation: (a).
4. **Registry edits** (needs approval):
   - HOLD_B → PLANNED for the E1, E3 and E4 rows (6 rows);
   - GATED → PLANNED for `GRPO-X_filter_compute`;
   - new rows for E2 (4) and, under option (b), E5 (2);
   - bookkeeping: Tier A rows still read PLANNED in `EXPERIMENT_REGISTRY.csv`. Mark them DONE, and GRPO-A_base_s2026 FAILED.
5. **Server time** for about 20–22 h of wall-clock on research-server, in the blocks above.

Once these are answered, Block 0 can be prepared. It involves no GPU work: configs, the registry diff, the `EXPERIMENT_PLAN.md` section 15 text, the option (a) code and tests if chosen, and the server command sheet. All of it is returned for review before anything is uploaded.
