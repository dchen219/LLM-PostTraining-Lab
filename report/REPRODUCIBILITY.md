# Reproducibility

> **Public repository.**
> - **What is not included:** model weights, item-level outputs, rollouts and dataset text.
> - **What is:** `results/run_records/` holds sanitised copies of every training run's `run.json` and `DONE.json`, and of the checkpoint lock. With them, the table and key-claim scripts in section 3 run from the repository alone.
> - **Commands that cannot run from this repository alone:** those that need item-level outputs, namely `final_report.py`, `research_analysis.py`, `verify_post_final_integrity.py` and the `PHASE1_INPUTS.sha256` check. They are shown as they were run, with the private `~/a5_*` folders.
>
> **Code layout.** This release reorganises the package: the grader is now `posttrain/verifiers/` and the prompt templates are `posttrain/prompts/`.
> - The frozen results were produced by the original layout.
> - `python scripts/verify_code_lineage.py` reverses the documented edits and reproduces the original code hash recorded in the checkpoint lock, and the reward-code hashes recorded by every run.
>
> See `docs/PACKAGE_MIGRATION.md` and `docs/PUBLIC_EXPORT.md`.

This file describes how every result in `TECHNICAL_REPORT.md` was produced and how to verify or regenerate it. The study is complete; no further training or evaluation is needed to check the report. Sections 1–3 verify and regenerate everything from the frozen files on a CPU. Sections 4–6 describe how the GPU results were produced.

## 0. Installation

```bash
# GPU environment used for the runs (Python 3.12 via uv; versions pinned in requirements-cloud.lock.txt)
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -r requirements-cloud.lock.txt          # torch 2.5.1, transformers 4.51.3, vllm 0.7.2, math-verify 0.7.0
python -m pytest tests -q                              # CPU tests: 168 passed, 6 skipped until the data is rebuilt (174 in the full environment)
python scripts/verify_code_lineage.py                  # release code -> original code identity (lock and verifier hashes)
# base model, pinned revision; its weights must hash to PT_BASE_SHA256 (scripts/server_env.sh)
python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen2.5-Math-1.5B", revision="4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"))'
```

On the research server, `setup_server.sh` performs these steps with extra checks: package verification, no credential-like files, exact version assertions, two GPUs and the base-weight hash. The analysis and report scripts in sections 1–3 need only the CPU packages plus scipy and matplotlib.

## 1. Verify the frozen record (CPU, minutes)

```bash
# project root
sha256sum final/freeze_v1.yaml          # ff0f0ce574bcb47c9dafedc77bab1654362bb955a81c593960108ca682b7a453
sha256sum -c --quiet results/research_analysis/PHASE1_FREEZE.sha256 && echo PHASE1_OUTPUTS_UNCHANGED
# with the recovered server outputs in the WSL/Linux home directory
cd ~ && sha256sum -c --quiet <project>/results/research_analysis/PHASE1_INPUTS.sha256 && echo PHASE1_INPUTS_UNCHANGED
cd ~/a5_final && sed 's#  final_eval/#  #' FINAL_EVAL.sha256 | sha256sum -c --quiet && echo RECOVERED_OK
```

| Record | Content |
|---|---|
| `final/freeze_v1.yaml` | Frozen evaluation plan: checkpoints (registry only), prompts, engine settings, families, thresholds |
| `~/a5_final/FINAL_CHECKPOINTS.json` | Lock: freeze sha256, code sha256, weights sha256 of all 11 checkpoints, written before the first evaluation |
| `~/a5_final/<checkpoint>/` | Item-level outputs (every response, score, category) for both held-out sets |
| `~/a5_final/BEFORE_FINAL.sha256`, `post_final_integrity.log` | Trained-run hashes before evaluation and the post-evaluation check (`TRAINED_RUNS_UNCHANGED`) |
| `~/a5_sessions`, `~/a5_recovery` | Run records of every training run (config, metrics, dev evaluations, rollouts, lineage) with manifests |

## 2. Regenerate the statistics, audit and figures (CPU)

```bash
V=path/to/venv/bin/python
$V scripts/final_report.py --final ~/a5_final --runs ~/a5_sessions --runs ~/a5_recovery --out results/final_eval_rerun
$V scripts/research_analysis.py --final ~/a5_final --runs ~/a5_sessions --runs ~/a5_recovery --out results/research_analysis_rerun
$V scripts/research_figures.py --data results/research_analysis_rerun          # needs matplotlib
$V scripts/verify_post_final_integrity.py --final ~/a5_final --runs ~/a5_sessions --runs ~/a5_recovery --out results/research_analysis_rerun/post_final_integrity
```

Write reruns to a new folder; the frozen `results/research_analysis/` is read-only. `research_analysis.py` recomputes every accuracy, interval, McNemar p-value and Holm p-value from the item files, using code independent of the generated report, and compares them with `report.json`. The expected result is 217 checks with 0 failed, and 0 mismatches. The only differences are Wilson bounds of about 5e-7, because scipy uses z = 1.959964 and the report uses 1.96.

## 3. Regenerate this report (CPU)

```bash
R="--runs results/run_records/a5_sessions --runs results/run_records/a5_recovery"   # or the private ~/a5_* folders
$V report/scripts/make_tables.py $R                                            # report/tables/*.csv|md from the frozen files
PYTHONPATH=<matplotlib> $V report/scripts/make_figures.py                       # copies frozen figures; draws the two diagrams
PYTHONPATH=<matplotlib> $V report/scripts/build_report.py                       # report/TECHNICAL_REPORT.md + build/TECHNICAL_REPORT.html
chrome --headless=new --no-pdf-header-footer --print-to-pdf=report/TECHNICAL_REPORT.pdf report/build/TECHNICAL_REPORT.html
$V report/scripts/key_claims.py $R --final results/run_records/a5_final         # report/AUDIT_KEY_CLAIMS.md
$V report/scripts/audit_numbers.py $R --final results/run_records/a5_final      # report/AUDIT_REPORT_NUMBERS.md
```

`TECHNICAL_REPORT.md` is assembled from `report/src/TECHNICAL_REPORT.src.md`. Every table is inserted from `report/tables/`, which is generated from the frozen files; no table is typed by hand. The number audit compares every number in the deliverables with the original artifacts.

## 4. Environment used for the GPU runs

| Component | Version |
|---|---|
| GPUs | 2× NVIDIA RTX A6000 (48 GB), driver ≥ 550 |
| Python packages | `requirements-cloud.lock.txt`: torch 2.5.1, transformers 4.51.3, vllm 0.7.2, math-verify 0.7.0, accelerate 1.6.0 |
| Base model | Qwen/Qwen2.5-Math-1.5B, revision `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`, weights sha256 `f7aa8d66…857b` |
| Data | `data/stage0/` built by `scripts/stage0_build_data.py` (sha256 of every file in `MANIFEST.json`) |
| Server settings | `scripts/server_env.sh` (`PT_HW=a6000`: micro-batch 2, 3,300-token GRPO cap; memory only, gradients unchanged) |

CPU tests: `python -m pytest tests -q`. In the full environment, with the datasets built, all 174 pass; that was the count on 2026-11-27 with the original layout. In a clean clone of this repository, 168 pass and 6 skip until the data is rebuilt.

## 5. How the training runs were produced

All runs are defined in `EXPERIMENT_REGISTRY.csv` and `configs/`, both generated by `scripts/make_configs.py`. They are launched only through gated scripts, which refuse to run without explicit authorization variables.

| Session | Command (inside `tmux`, after `source scripts/server_env.sh`) | Runs |
|---|---|---|
| A: SFT tuning, then 7 SFT arms | `bash scripts/server_tier_a.sh sft-tune`; `bash scripts/server_tier_a.sh sft-main auto` | 3 + 7 |
| B: GRPO tuning, then GRPO-B and GRPO-A | `bash scripts/server_tier_a.sh grpo-tune`; `bash scripts/server_tier_a.sh grpo-main auto` | 3 + 2 |
| Post-hoc branches of the failed GRPO-A | `bash scripts/server_recovery.sh run` | 2 |

`auto` applies the pre-registered learning-rate rule (`scripts/choose_lr.py`). Both stages selected 1e-5. The GRPO-A failure, its incident record and the branch protocol are documented in `GRPO_A_INCIDENT.md` and `recovery/`. The failed run folder is refused by the code and is never resumed. The detailed server command sheets are private and are not part of the public repository.

GPU training is not bit-deterministic (vLLM sampling and CUDA kernels). Re-running a configuration gives a statistically similar, not identical, trajectory. Resume is bit-exact on CPU (tested).

## 6. How the held-out evaluation was produced

`FINAL_EVAL_PLAN.md` steps 3–5:
1. `bash scripts/server_final_eval.sh ready`, then `check`: identity of all 11 checkpoints against DONE.json lineage.
2. `bash scripts/server_final_eval.sh run`. This writes the lock first, then evaluates on both GPUs with greedy decoding.
3. `sha256sum -c --quiet BEFORE_FINAL.sha256 && echo TRAINED_RUNS_UNCHANGED`, then `FINAL_EVAL.sha256` of the outputs.

Evaluation is reproducible on the same engine, GPU type and settings: re-evaluating identical weights on dev changed 0 of 1,500 answers. It is not guaranteed across engine versions or GPU types.

## 7. GPU cost

| Item | GPU-h (2 × wall-clock) |
|---|---|
| SFT tuning (3 runs) | 1.43 |
| SFT arms (7 runs) | 8.13 |
| GRPO tuning (3 runs, 30 steps) | 5.71 |
| GRPO-B (100 steps) | 7.35 |
| GRPO-A (failed after 97 updates) | 6.61 |
| Post-hoc branches A1, A2 (20 steps each) | 4.79 |
| **Training and tuning total** | **34.01** |
| Held-out evaluation (11 checkpoints, 1.49 h on 2 GPUs) | 2.98 |

Source: `wall_s` in each finished run's DONE.json. GRPO-A has no DONE.json; its 6.61 GPU-h is `gpu_h_total` in `results/research_analysis/analysis.json`, 2 × the last wall-clock value logged in its `metrics.jsonl`. Training wall-clock includes in-training dev evaluations.
