# GRPO-A_base_s2026: failure record, diagnosis and post-hoc recovery branches

> **Update, 2026-11-26: the logs have been analysed** (`results/grpo_a_incident_analysis/REPORT.md`, figures there):
> - No duplicate rows and no resumes.
> - Verification ruled out: 0 errors, all 2,304 saved rewards reproduce, and format held at 87.5% at the collapse.
> - One-update collapse at update 96, from a low-entropy plateau, with the largest gradient of the run.
> - The earliest false-alarm-free warning is the rollout log-prob gap rising against its own 20-step baseline at steps 94-95, at about 0.005, far below the 0.05 gate.
> - Section 2 below predates the logs; where they differ, the analysis report wins.
>
> **V5, 2026-11-26 (approved):** every step's rollouts are saved with token-level detail, diagnostic criteria are registered before training (section 3), and the update is built against the deployed V3 (section 4).

Recorded 2026-11-26 from the user's report. The server logs had not yet been copied to the laptop, so every statement about the run's trajectory before step 95 is open until `scripts/recovery_report.py` has read them.

## 1. Record

`GRPO-A_base_s2026` (direct GRPO from the base model, lr 1e-5 by the pre-registered rule) **failed**: 97 updates completed; the batch for step 98 was rejected by the rollout log-probability gate (mean gap 0.0794 > 0.05) **before** `grpo_update()`, so it changed nothing. No `DONE.json`. Last saved state: step 80, sha256 `73114307…e1bb`, backed up in `incident_backups/A_base_s2026_failed_step97/`.

Tier A therefore stands at 14 completed runs and 1 failed pre-registered run. The failed run is reported as failed. It is not replaced, resumed or completed. `recovery/incident_GRPO-A_base_s2026.json` records it, and the code now refuses to open that folder (`Run`) and the Tier A launcher skips it.

## 2. Diagnosis

| Reported value | Step 95 | 96 | 97 | Attempted 98 |
|---|---|---|---|---|
| Training reward | 0.707 | 0.617 | **0.055** | not trained |
| Mean rollout log-prob gap | | | 0.0388 | **0.0794** |
| Largest single-token gap | | | 19.83 | |

For comparison, steps 1-10 from the base model (pilot and A6000 validation): mean gap 0.0095-0.0118, p99 about 0.11, largest token 0.4-7.3.

The reward logged at step k is measured on rollouts from the weights after k−1 updates. The collapse from 0.617 to 0.055 therefore appeared in the weights produced by update 96, or by drift accumulated up to it.

### Confirmed defects (none caused the failure; all fixed)

| Defect | Effect | Fix |
|---|---|---|
| The gate exited without saving the rejected batch | The step-98 rollouts, token IDs and per-token gaps are lost; only the message remains | Before exiting, the gate writes `failed_batch_step<N>/` (`rollouts.jsonl` with prompt IDs, token IDs, both log-probs per token, worst tokens, rewards and advantages; `meta.json`) and a `FAILED.json`. A run with `FAILED.json` is never resumed |
| An ordinary resume appended duplicate step rows (steps between the last save and the crash) and overwrote rollout and evaluation files of those steps | Means over steps could double-count | On resume, later rows are moved to `metrics.superseded.jsonl` and later files to `superseded/`; nothing is deleted. `session_report.py` now keeps one row per step and reports `duplicated_step_rows`. Check `resumes` for the 14 completed runs |
| `server_tier_a.sh grpo-main auto` would have resumed the failed run from step 80 (it sees `state.pt`), appending to its logs and eventually writing `DONE.json` | Original evidence overwritten | Incident guard in `Run` and in the launcher |
| No measure of update size over all parameters (the probe samples 6 tensors) | The question "how large was each step" could not be answered | `update_l2`, `update_rms`, `update_max_abs`, `update_rel`, `param_l2`, `adam_m_l2`, `adam_sqrt_v_mean` every step, computed exactly from the AdamW state (tested against the real change) |

### Examined and not supported by the evidence

| Hypothesis | Why not |
|---|---|
| Stale or partial weights in vLLM, including CUDA graphs after the in-place sync | The exact check passed at step 97 and before the attempted step 98. It compares vLLM's live parameter tensors, which are the tensors the captured graphs read, because `load_weights` copies in place. Qwen2 holds no weight-derived buffers (the rotary cache depends only on positions) |
| Prompt/response token misalignment, position IDs, padding | Lengths are checked per rollout. A shifted alignment would produce gaps of several nats on most tokens, not a mean of 0.01-0.04 with p99 near 0.11. Right padding with an attention mask; real tokens get positions 0..n−1; covered by the padding-invariance and mask-alignment tests |
| A6000 micro-batch 2 and the 3,300-token cap | Tested to give the same gradient, loss and update count |
| Clipping failed to act | It cannot act in this configuration (next section), so it neither caused nor could have prevented the failure |

### Plausible explanations (need the logs)

1. **Policy collapse in the last updates.** Reward fell by 0.56 between two consecutive weight versions. Gradient norm at step 97 (0.208) is about 7-20 times the norms seen in validation (0.006-0.034). Adam normalises the gradient, though, so the per-element step is set by lr and m/√v, not by the norm. Whether update size grew is now logged, but it was not logged in the original run. **Open:** whether this is format collapse (check `format_rate` and `truncated_rate` at 96-97) or answer collapse, and whether `verifier_errors` (timeouts and grader errors score 0) rose.
2. **A growing mismatch between training and inference numerics.** Weights agree exactly after rounding to bf16, but activations do not. Under autocast, PyTorch keeps an fp32 residual stream, fp32 RMSNorm weights and fp32 embeddings, and uses SDPA kernels. vLLM runs everything in bf16 with its own kernels. The mean gap grew from about 0.01 to 0.039 at step 97 and 0.079 one step later. With one update per batch, the update never sees the sampler's probabilities: rollouts drawn by vLLM are treated as on-policy for PyTorch's model, and nothing corrects that bias. **Open:** cause or consequence. The step-97 gap was still inside tolerance while reward had already collapsed. That favours "collapse first, then degenerate text with a larger gap" but does not prove it. The per-step gap trajectory (in `metrics.jsonl`) decides whether it rose before step 96.
3. **The learning rate was chosen at a shorter horizon.** lr 1e-5 is the top of the grid and was selected from 30-step tuning runs, then used for 100 steps. From the base model it had never been observed beyond step 30. The same rate completed 100 steps from the SFT parent (GRPO-B). This is a limitation of the tuning design, not evidence that the rate caused the failure; branches A1 and A2 test it.

### Unresolved

What the rejected batch contained (not saved). Whether the gap trended upward before step 96. Whether the collapse is reproducible: GPU kernels are not bit-deterministic, so A1 is a statistical replay. Its step 81 should be close to the original step 81 (same weights, prompts and generation seeds), and later steps may diverge. If A1 does not fail, the original dynamics are not ruled out. If it fails, a learning-rate cause is still not shown unless A2 behaves differently, and one seed per branch cannot separate learning-rate effects from noise.

### Verifier warnings

`score()` never raises. Timeouts and grader exceptions score 0, are flagged per rollout and counted per step (`verifier_errors`). Parser warnings from SymPy or LaTeX handling do not change a reward unless they become such an error. Read `verifier_errors` at steps 95-97 before attributing the reward drop to the policy.

### Why clipping is inactive

`epochs: 1` and `update_batch: 256` = 32 prompts × 8 rollouts give exactly one optimizer step per rollout batch. The "old" log-probs are recomputed by the same PyTorch model just before that step, so the ratio π_θ/π_old is exactly 1 at the only point where it is evaluated. clip(1, 0.8, 1.2) = 1, so min(r·A, clip(r)·A) = r·A and the gradient is A·∇log π: REINFORCE with a group-mean baseline, normalised by the constant 1,024. Clipping acts only with several updates per batch. Because the old log-probs come from PyTorch rather than from vLLM's sampler, it would not react to the sampler mismatch either. The rollout gate is the only protection against that mismatch.

## 3. Recovery branches (post-hoc, diagnostic; final design, V5)

| ID | Folder | Learning rate | Everything else |
|---|---|---|---|
| `GRPO-A-replay80-lr1e-5-s2026` (A1) | `runs/recovery/A_replay80_lr1e-5_s2026` | 1e-5 (as the original) | identical |
| `GRPO-A-recovery80-lr5e-6-s2026` (A2) | `runs/recovery/A_recovery80_lr5e-6_s2026` | 5e-6, from the first update after step 80 | identical |

**Both branches:**
- Start from the same verified step-80 state and run steps 81-100.
- Each configuration *is* the original's recorded `run.json` configuration (hash re-checked). It changes only the keys on an explicit allowlist (`posttrain/fork_grpo.py: ALLOWED`); anything else is refused.
- Allowed changes, all recorded in each branch's `run.json`:
  - `train.lr` (A2 only): the intervention.
  - `train.save_every`: 20 → 5. Operational, so a branch can resume before step 100.
  - `grpo.save_rollouts_every`: 10 → 1. Instrumentation: every step's rollouts are written.
- Unchanged: optimizer (AdamW, betas 0.9/0.95, no weight decay), loss and clipping (`grpo_clip`, 0.2, one update per batch), no KL term, sampling (temperature 1.0, 32 prompts × 8, 1,024 tokens), reward function and verifier 2.0.0, 100 steps, the 0.05 rollout gate and every other gate. The tuned learning rate is not re-chosen, and the registry is not modified.

`python -m posttrain.fork_grpo`:
- Before creating anything, it checks the source state's sha256, that its step is 80, and that it holds model, optimizer, step, sampler and all RNG states.
- It restores all of them; on the CPU tests the two branches start from bit-identical states.
- It evaluates the step-80 state on GSM8K dev (1,000) and MATH dev (500) before any update.
- It trains with every gate unchanged and re-checks the source state and the original folder at the end.

**What every attempted step leaves on disk (V5):**

| Where | Content |
|---|---|
| `rollouts/step<N>.jsonl`, every completed step | Per rollout: step, prompt ID, reference answer, response, token IDs, token count, finish reason and truncation flag, reward, format and answer rewards, advantage, verifier rule and error, mean entropy, mean and max log-prob gap, and the full per-token sampling and policy log-probs |
| `metrics.jsonl`, every step | Reward, format, truncation, entropy, gradient norm, update size over all parameters (`update_l2`, `update_rms`, `update_max_abs`, `update_rel`, `param_l2`, Adam state), gap mean, p99 and max, tokens over 1 and 5 nats, worst token, informative groups, lengths, timing, memory |
| `failed_batch_step<N>/`, a batch rejected by the rollout gate or by a non-finite loss or gradient | The same per-rollout records plus the 5 worst tokens of each rollout, and `meta.json` (mismatch statistics, sync record, learning rate the update would have used, `optimizer_updated: false`). A weight-check failure stops before any rollout exists, so it has no batch |

Saving draws no random numbers and touches neither the model nor the optimizer. On the CPU tests, a run with rollouts saved every step equals a run without saving in every logged number, in the model state before every update, in the final weights, and in the Python, NumPy and PyTorch random streams.

**Storage and overhead:**
- About 5.4 MB per step, measured on the original run's real response lengths (99k tokens).
- Up to 14 MB per step if every response hit 1,024 tokens.
- About 220 MB for both branches (at most 0.6 GB), plus up to about 14 MB per rejected batch.
- Writing takes well under a second, under 1% of a 124 s step.
- State saves every 5 steps: one 18.5 GB `state.pt` per branch, about 37 GB transiently while it is replaced; deleted when the branch finishes. A finished branch keeps a 3 GB bf16 checkpoint.

### Registered diagnostic criteria (`recovery/diagnostic_criteria.yaml`, frozen 2026-11-26, before any branch ran)

| Criterion | Exact rule |
|---|---|
| Early warning | At step t, for both the mean and the p99 rollout log-prob gap: z = (x_t − median of steps t−20..t−1) / (1.4826 × MAD of those 20 steps). Warning when **both** z ≥ 3.0. For steps 81-100 the window includes the original's steps up to 80 |
| Severe reward deterioration | Any training step with mean reward < 0.30 |
| Sustained reward deterioration | At least two consecutive steps < 0.30. If a rollout-gate stop follows the first low step, this is "not observable", not "absent" |
| Numerical failure | A rollout batch with mean gap > 0.05: the existing gate, which stops the branch before any update on it |
| Successful completion | Step 100 reached and `DONE.json` written, with every correctness gate passed at every step and no `FAILED.json` |

- **Diagnostic only.** Nothing stops, slows or restarts a branch because of reward or a warning. The only stops are the existing correctness gates.
- **Recorded and tested.** Each branch records the file's sha256 in `run.json`. `scripts/recovery_report.py` evaluates exactly these rules, and the unit tests check them against hand computation.
- **Caveat:** the early-warning rule was derived retrospectively from the one failed run (it fired at 94-97 and at no step from 30 to 93 there) and has not been validated on any other run. With one run per branch, a hit or a miss is descriptive.

Compared in `results/grpo_a_incident/REPORT.md`, together with the criteria table:
- Dev accuracy at step 80 and at the end (strict, second verifier, relaxed), with paired McNemar tests and Holm correction.
- Format rate, reward, entropy, gradient norms and update size.
- Gap mean, p99 and max; informative groups; length and truncation.
- Completion, wall time and GPU-hours.
- A1's steps 81-97 against the original's.

## 4. Commands (V5)

**Which update to install:** `a5_update_v5.tar.gz` is built against the **V3** package that is deployed, so V4 is not needed. If V4 had been applied, V5 still applies on top, and both paths were rehearsed. V5 changes no configuration, data or registry file, and no file in `sft/`, `grpo/` or `validation/`.

Training needs `A5_AUTHORIZED=yes` (owner agreement and allocated GPUs; replace `0,1` with your GPUs) and `A5_RECOVERY_APPROVED=yes`. Expected about 55-60 min per branch, about 2 h and 4 GPU-hours for both. This is based on the original run's measured 124 s per step and 80-150 s per evaluation, plus state loading and hashing.

**1. WSL: check and send the update**
```bash
cd <repo>/dist && sha256sum -c a5_update_v5.tar.gz.sha256
scp a5_update_v5.tar.gz a5_update_v5.tar.gz.sha256 <host>:/srv/research/posttrain-v2/
```

**2. SERVER (`ssh <host>`): apply it (replaced files are kept in `a5_before_v5`), and record checksums of everything that must not change**
```bash
tmux new -s a5-recovery
export A5_AUTHORIZED=yes A5_GPUS=0,1 A5_ROOT=/srv/research/posttrain-v2
cd $A5_ROOT && sha256sum -c a5_update_v5.tar.gz.sha256 && mkdir a5_before_v5 && tar tzf a5_update_v5.tar.gz | while read -r f; do [ ! -f "$f" ] || cp --parents -p "$f" a5_before_v5/; done && tar xzf a5_update_v5.tar.gz
cd $A5_ROOT/a5 && source scripts/server_env.sh
cd $PT_RUNS && find sft grpo -type f -exec sha256sum {} + > $A5_ROOT/BEFORE_RECOVERY.sha256; cd $A5_ROOT/a5      # a few minutes; includes the step-80 state
echo "73114307c664b55c05b6b6eddb326d0179525c89944dac10daef48f2a424e1bb  $A5_ROOT/incident_backups/A_base_s2026_failed_step97/state.pt" | sha256sum -c
```

**3. SERVER: checks (start nothing)**
```bash
python3 scripts/verify_upload.py              # 188 expected, 0 missing/changed
bash scripts/server_preflight.sh              # PREFLIGHT_OK
bash scripts/server_recovery.sh ready         # 101 passed; READY (recovery)
bash scripts/server_recovery.sh check         # per branch: source_state_sha256_ok true; differences: run_dir, train.save_every, grpo.save_rollouts_every (+ train.lr for A2); CHECK_OK
bash scripts/server_tier_a.sh plan            # GRPO-A_base_s2026 shows FAILED
```

**4. SERVER: run A1, then A2**
```bash
export A5_RECOVERY_APPROVED=yes
set -o pipefail
bash scripts/server_recovery.sh run 2>&1 | tee -a $A5_ROOT/recovery.log
bash scripts/server_recovery.sh status
cd $PT_RUNS && sha256sum -c --quiet $A5_ROOT/BEFORE_RECOVERY.sha256 && echo ORIGINALS_UNCHANGED
echo "73114307c664b55c05b6b6eddb326d0179525c89944dac10daef48f2a424e1bb  $A5_ROOT/incident_backups/A_base_s2026_failed_step97/state.pt" | sha256sum -c
find recovery -type f ! -name '*.safetensors' ! -name 'state.pt*' -exec sha256sum {} + > RECOVERY.sha256; cd $A5_ROOT/a5; unset A5_RECOVERY_APPROVED
```

**5. WSL: recover and report**
```bash
mkdir -p ~/a5_recovery/grpo ~/a5_recovery/recovery
rsync -a --exclude 'state.pt*' <host>:/srv/research/posttrain-v2/runs/grpo/A_base_s2026 ~/a5_recovery/grpo/
rsync -a --exclude '*.safetensors' --exclude 'state.pt*' <host>:/srv/research/posttrain-v2/runs/recovery/ ~/a5_recovery/recovery/
rsync -a <host>:/srv/research/posttrain-v2/runs/RECOVERY.sha256 ~/a5_recovery/
rsync -a <host>:/srv/research/posttrain-v2/recovery.log ~/a5_recovery/
(cd ~/a5_recovery && sha256sum -c --quiet RECOVERY.sha256 && echo RECOVERED_OK)
cd <repo>
<repo>/.venv/bin/python scripts/recovery_report.py --runs ~/a5_recovery --out results/grpo_a_incident
```

| Message | Action |
|---|---|
| `<branch> stopped at the rollout log-prob gate … continuing` | A result (the numerical-failure criterion). The batch is in `failed_batch_step<N>/` |
| `STOPPED at <branch>: weight-sync check failed` or `non-finite …` | Infrastructure. Nothing else runs; send the log. A non-finite stop also saves its batch |
| `STOPPED at <branch> (not a gate …)` after an interruption | Run the same `run` command; the branch resumes from its last 5-step state |
| Out of memory, or any `REFUSED` | Do not retry. Send the log |
