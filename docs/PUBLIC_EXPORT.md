# About this public repository

This repository is a **curated public export** of a private working repository. It was built from an explicit file allowlist; nothing outside the list was copied. The private working copy, its frozen Phase I record and its run folders were not modified. The original working directory was not a Git repository, so this repository starts with a fresh history.

## What is included

| Area | Content |
|---|---|
| Code | `posttrain/` (the post-training system), `scripts/` (data build and verification, gated launchers, diagnostics, recovery, analysis and audit), `report/scripts/` |
| Third-party files used by the pipeline | `posttrain/verifiers/drgrpo_grader.py` and the R1-Zero prompt template (see `THIRD_PARTY_NOTICES.md`) |
| Tests and configuration | `tests/`, `configs/`, `EXPERIMENT_REGISTRY.csv`, `final/freeze_v1.yaml`, `recovery/` |
| Data (no text) | `data/stage0/MANIFEST.json` (hashes, counts), `contamination.json`, `subsets/*.json` (trace ids), `recovery_decisions.jsonl` (id, recovered, tier for the 1,340 audited traces) |
| Results | `results/final_eval/` (held-out report), `results/research_analysis/` (frozen analysis, statistics, figures, audits), `results/grpo_a_incident_analysis/`, `results/stage0/` and `results/verifier_v2/` (verifier and precision checks) |
| Run records | `results/run_records/`: `run.json` and `DONE.json` of every training run, plus the checkpoint lock `FINAL_CHECKPOINTS.json` |
| Report | `report/` (Markdown and PDF technical report, figures, tables, audits, build scripts) |
| Plans | `EXPERIMENT_PLAN.md`, `FINAL_EVAL_PLAN.md`, `GRPO_A_INCIDENT.md`, `PHASE2_PLAN.md` (an unexecuted follow-up, marked as not pursued) |

## What is not included, and why

| Category | Examples | Reason |
|---|---|---|
| Credentials and private access notes | key files, server access notes | Never published |
| Personal documents | resume and interview material | Personal |
| Model weights and optimizer states | `*.safetensors`, `state.pt` | Size. Weights are identified by sha256 in `results/run_records/a5_final/FINAL_CHECKPOINTS.json` |
| Dataset text | SFT traces, RL pool, evaluation sets | Third-party text; rebuild it from public sources (below) |
| Item-level outputs | every held-out response and score, rollouts, metrics logs | Size and benchmark text. Aggregates and statistics are included |
| Third-party tutorial and course material | the `datawhalechina/diy-llm` tutorial's code, notebook and images; course handouts, starter code and tests | Not this project's work (the tutorial is CC BY-NC-SA 4.0) |
| Server command sheets and archival notes | server migration and session command sheets, cloud-pilot material, superseded plan versions | Infrastructure details; not needed to read or check the results |
| Build artifacts | upload packages, generated HTML | Regenerable |

Some documents still mention files that are not included, such as `SERVER_MIGRATION.md`, `SESSIONS_A_B.md`, `RESEARCH_SERVER_AUDIT.md` and `~/a5_*` run folders. Those mentions are kept so that the frozen record stays as written.

## Sanitisation

Server paths, host names and user names were replaced in every shipped text file:

| Placeholder | Stands for |
|---|---|
| `/srv/research/…` | the project's working directory on the research server |
| `research-server`, `<host>` | the research server's host name |
| `<server-user>` | the server account name |
| `<repo>` | the local path of the working repository |
| `~` | the local home directory |

13 repository files changed. 6 of them belong to the frozen Phase I record, so `sha256sum -c results/research_analysis/PHASE1_FREEZE.sha256` reports those 6 as FAILED in this repository. The other 56 frozen files verify unchanged, and `final/freeze_v1.yaml` is byte-identical (sha256 `ff0f0ce5…a453`). The original hashes are kept in the freeze manifest; the table gives both hashes.

| File | In the frozen record | Original sha256 | Public sha256 |
|---|---|---|---|
| `FINAL_EVAL_PLAN.md` | no | `f8ffd61a5e0ee82e…` | `4bbba4ff683628ac…` |
| `GRPO_A_INCIDENT.md` | no | `52a46b445efa7d7d…` | `7060c166420ba473…` |
| `PHASE2_PLAN.md` | no | `81fbd33fbdfbe447…` | `de88b175b5acb5b7…` |
| `results/final_eval/report.json` | yes | `b452dd46d22e5633…` | `62290f96191de9e1…` |
| `results/research_analysis/AUDIT.md` | yes | `d6a58997abbfe687…` | `6c5a9276503974e9…` |
| `results/research_analysis/PHASE1_FREEZE.md` | yes | `16460e648b84a363…` | `b1ecfc4da2c3d58b…` |
| `results/research_analysis/post_final_integrity/post_final_integrity.json` | yes | `4f813d7ccc80512a…` | `bf439479151a2350…` |
| `results/research_analysis/training_audit/AUDIT.md` | yes | `5bcd32a862a9d21c…` | `e7c9eb0cf73739b3…` |
| `results/research_analysis/training_audit/audit.json` | yes | `a569400b1743ede1…` | `16715d9b2fe454b3…` |
| `scripts/server_env.sh` | no | `cc390425a49115b3…` | `059f7a6fc3f5f1e5…` |
| `scripts/stage0_base_eval_hf.py` | no | `aeeb89698e79dc5f…` | `0788824497c5f242…` |
| `scripts/stage0_failure_decomposition.py` | no | `59be702ea4506430…` | `6004bc1e4230a957…` |
| `tests/test_posttrain.py` | no | `88150a349b019bba…` | `cdf070b4c06c006b…` |

**Run records.** All 35 files in `results/run_records/` had their server paths replaced. The hashes recorded inside them (`config_sha256`, `code_sha256`, weight hashes) are the original values. `config_sha256` was computed over the original, unsanitised paths, so recomputing it from a sanitised `run.json` gives a different value. Every other value is unchanged. As a check, `report/scripts/make_tables.py` regenerates all 15 report tables byte-identically from these records.

**Code identity.** This release reorganises the package (`docs/PACKAGE_MIGRATION.md`), so its own code hash differs from the one the runs recorded. `scripts/verify_code_lineage.py` reverses the documented edits and reproduces the locked evaluation-code hash (`0e0b00141307…`) and the verifier hashes recorded by every run.

## Export-only edits

These edits exist only in the public copy:

| File | Change |
|---|---|
| `tests/test_posttrain.py` | Tests that read dataset text skip when `data/stage0/sft_all_labeled_4836.jsonl` is absent. Before, they checked only for `MANIFEST.json`, which is shipped |
| `report/scripts/key_claims.py` | Checks the root `README.md` instead of the draft. Takes `--final`. Skips claims about the personal documents, which are not shipped |
| `report/scripts/audit_numbers.py` | Audits the shipped documents only |
| Package layout | Grader and prompt templates moved into `posttrain/verifiers/` and `posttrain/prompts/`, with the import edits listed in `docs/PACKAGE_MIGRATION.md` |
| `pyproject.toml`; `uv.lock` | Standalone package metadata; the legacy lock file is not shipped (`requirements-cloud.lock.txt` pins the environment) |
| `report/` | The report, executive summary and reproducibility guide are organised around the project's seven engineering contributions. The PDF was re-rendered. Results, tables and figures are unchanged |
| `scripts/stage0_*.py` | The project root is located relative to the script, not a fixed local path |
| `report/AUDIT_KEY_CLAIMS.md`, `report/AUDIT_REPORT_NUMBERS.md` | Regenerated for the shipped documents |

## Tests

| Environment | Result |
|---|---|
| Original layout, full working environment (2026-11-27) | 174 passed |
| This repository as cloned | 168 passed, 6 skipped, 1 warning |
| This repository with the private inputs restored (datasets and one historical audit file) | 174 passed, 1 warning |

The 6 skipped tests check dataset hashes, train/eval separation, config resolution and the registry against the dataset text. To run them, rebuild the data. You need:
- `sft-reason/` from `garg-aayush/sft-cs336-assign5-datasets`;
- the GSM8K train parquet and test jsonl.

The expected source hashes are listed in `data/stage0/MANIFEST.json` under `source_sha256`.

```bash
python scripts/stage0_build_data.py --src <sft-reason dir> --gsm8k-train <train parquet> --gsm8k-test <test jsonl> \
       --audit data/stage0/recovery_decisions.jsonl --out data/stage0_rebuild
# compare data/stage0_rebuild/MANIFEST.json output_sha256 with data/stage0/MANIFEST.json, then copy the *.jsonl files into data/stage0/
```

The builder reads only `id`, `recovered` and `tier` from the audit file, and it does not hash that file. `recovery_decisions.jsonl` should therefore give the same outputs as the private audit file. This rebuild was not run as part of the export.

## Reproducibility limits of the public repository

- **Item-level recomputation is not possible from this repository alone.** `scripts/final_report.py`, `scripts/research_analysis.py`, `scripts/verify_post_final_integrity.py` and the `PHASE1_INPUTS.sha256` check need the item-level outputs and run folders, which are not shipped. The statistics and audits they produced are included, frozen.
- **Report tables and key claims regenerate from the repository.**
  - Tables: `make_tables.py` with `results/run_records/`.
  - Key claims: `key_claims.py` re-checks every claim in the report, the executive summary, the reproducibility guide and this README.
- **The data must be rebuilt** from public sources before training or running the data tests (above).
- **Training and evaluation need two GPUs and the pinned stack** (`requirements-cloud.lock.txt`: torch 2.5.1, vLLM 0.7.2, CUDA 12.4). The gated launchers expect the research-server layout and refuse to start without explicit authorisation variables.
- **`report/src/TECHNICAL_REPORT.src.md` is a build template.** Its image paths resolve from `report/` after `report/scripts/build_report.py` runs. Read `report/TECHNICAL_REPORT.md` or the PDF instead.
- **Frozen-manifest check:** 6 frozen files differ from their original hashes because of sanitisation (above).
