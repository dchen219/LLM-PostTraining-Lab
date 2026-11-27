# Artifact audit and consolidated development results (generated)

Trees: ~/a5_sessions, ~/a5_recovery

**Unresolved problems: 0**


Explained (not problems): 2

- SFT-reverified_full_s0: 25 duplicated step rows = steps 151-175 replayed after one resume from the step-150 save (before the rewind fix); first replayed step identical, later losses differ by at most 7.1e-04 (GPU nondeterminism); reports keep the last row per step, which is the trajectory that produced the checkpoint
- BEFORE_RECOVERY.sha256: 15 entries are model weights or optimizer states (model.safetensors, state.pt), excluded from the download by design; their recorded hashes are cross-checked against DONE.json below and must be confirmed on the server

Manifest provenance: SESSION_MANIFEST_PROVENANCE.txt: Created retrospectively for artifact transfer. Historical session manifests were missing. Historical integrity checked against BEFORE_RECOVERY.sha256. Created at: 2026-11-26T16:29:05-08:00
SESSION_A.sha256 and SESSION_B.sha256 were generated retrospectively for transfer verification (after the sessions, not during them). Their evidence that the transferred files are the pre-recovery files is the agreement with BEFORE_RECOVERY.sha256, written on the server before the recovery branches ran.

| run | expected | status | config hash | verifier | data file | data ids | ckpt lineage | weights = server manifest | code | rows | dup | last | resumes | GPU-h | weights | chain | config differences |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT-tune_lr1e-5 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 32 | 0 | 32 | 0 | 0.37 | e5957bb836ca0cff | base>sft | - |
| SFT-tune_lr2e-5 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 32 | 0 | 32 | 0 | 0.44 | 5a366b209d2801c9 | base>sft | - |
| SFT-tune_lr5e-5 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 32 | 0 | 32 | 0 | 0.62 | aae6c56fb1c426b3 | base>sft | - |
| SFT-exact_full_s0 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 214 | 0 | 214 | 0 | 1.83 | 02c6ba5396a4d113 | base>sft | train.lr 2e-05->1e-05 |
| SFT-reverified_full_s0 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 303 | 25 | 278 | 1 | 2.69 | 02ecf90239277d41 | base>sft | train.lr 2e-05->1e-05 |
| SFT-reverified_sizecontrol_s0 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 214 | 0 | 214 | 0 | 1.57 | 765bcbcbc2daa37d | base>sft | train.lr 2e-05->1e-05 |
| SFT-exact_n512_s0 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 32 | 0 | 32 | 0 | 0.44 | c4f4b4e94efa551a | base>sft | train.lr 2e-05->1e-05 |
| SFT-reverified_n512_s0 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 32 | 0 | 32 | 0 | 0.44 | f57b1af68b655397 | base>sft | train.lr 2e-05->1e-05 |
| SFT-reverified_n128_s0 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 8 | 0 | 8 | 0 | 0.12 | 622f1201d8ae8f7b | base>sft | train.lr 2e-05->1e-05 |
| SFT-reverified_n2048_s0 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 128 | 0 | 128 | 0 | 1.03 | 65f61e4c0f5e830b | base>sft | train.lr 2e-05->1e-05 |
| GRPO-tune_lr2e-6 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 30 | 0 | 30 | 0 | 1.95 | c75ca51e83a5efbc | base>grpo | - |
| GRPO-tune_lr5e-6 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 30 | 0 | 30 | 0 | 1.83 | fb03a43fd99f8a35 | base>grpo | - |
| GRPO-tune_lr1e-5 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 30 | 0 | 30 | 0 | 1.93 | 41ce3ff86d3dbac9 | base>grpo | - |
| GRPO-A_base_s2026 | failed | failed | yes | 2.0.0 | - | - | - | - | 739a2df9e838 | 97 | 0 | 97 | 0 | - | - | - | train.lr 5e-06->1e-05 |
| GRPO-B_sft512_s2026 | done | done | yes | 2.0.0 | yes | yes | yes | yes | 739a2df9e838 | 100 | 0 | 100 | 0 | 7.35 | 83dd404abf349f3a | base>sft>grpo | train.lr 5e-06->1e-05 |
| GRPO-A-replay80-lr1e-5-s2026 | done | done | yes | 2.0.0 | yes | yes | yes | - | c80348598ca8 | 20 | 0 | 100 | 0 | 2.28 | bf3453c6c71da3a3 | base>grpo | - |
| GRPO-A-recovery80-lr5e-6-s2026 | done | done | yes | 2.0.0 | yes | yes | yes | - | c80348598ca8 | 20 | 0 | 100 | 0 | 2.51 | a4055bd25156cf73 | base>grpo | - |

| manifest | entries | verified locally | missing | changed | weights not downloaded |
|---|---|---|---|---|---|
| ~/a5_sessions/BEFORE_RECOVERY.sha256 | 273 | 258 | 0 | 0 | 15 |
| ~/a5_sessions/SESSION_A.sha256 | 151 | 151 | 0 | 0 | 0 |
| ~/a5_sessions/SESSION_B.sha256 | 107 | 107 | 0 | 0 | 0 |
| ~/a5_recovery/RECOVERY.sha256 | 74 | 74 | 0 | 0 | 0 |

Learning-rate rule re-derived from the tuning runs: SFT: tuning {'1e-05': 0.861, '2e-05': 0.85, '5e-05': 0.818} -> 1e-05; main runs [1e-05]; GRPO: tuning {'2e-06': 0.676, '5e-06': 0.846, '1e-05': 0.875} -> 1e-05; main runs [1e-05]

Cross-checks: {"SESSION_A.sha256": {"entries": 151, "also_in_before_recovery": 151, "only_in_session_manifest": [], "differing": []}, "SESSION_B.sha256": {"entries": 107, "also_in_before_recovery": 107, "only_in_session_manifest": [], "differing": []}, "before_recovery_not_in_session_manifests": [], "original_state_pt": "73114307c664b55c05b6b6eddb326d0179525c89944dac10daef48f2a424e1bb", "original_copies": {"files": 20, "compared": 20, "differing": []}}

## Development results (strict = frozen verifier 2.0.0; %)

| run | kind | point | lr | GSM8K strict (95% CI) | second | relaxed | format | clean | doubled </think> | tokens | MATH strict (95% CI) | second | relaxed | format | trunc | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT-tune_lr1e-5 | registered | final_step | 1e-05 | 86.1 (83.8-88.1) | 87.4 | 87.4 | 99.9 | 99.9 | 0 | 141 | 33.4 (29.4-37.6) | 35.2 | 36.4 | 80.8 | 82 | 0.37 |
| SFT-tune_lr2e-5 | registered | final_step | 2e-05 | 85.0 (82.7-87.1) | 85.6 | 85.6 | 99.7 | 99.7 | 0 | 139 | 27.8 (24.1-31.9) | 28.6 | 28.6 | 90.4 | 44 | 0.44 |
| SFT-tune_lr5e-5 | registered | final_step | 5e-05 | 81.8 (79.3-84.1) | 82.4 | 82.4 | 99.9 | 99.9 | 0 | 141 | 30.2 (26.3-34.4) | 31.2 | 31.4 | 89.2 | 47 | 0.62 |
| SFT-exact_full_s0 | registered | final_step | 1e-05 | 88.8 (86.7-90.6) | 88.8 | 88.8 | 99.9 | 99.9 | 0 | 133 | 31.6 (27.7-35.8) | 32.0 | 32.0 | 95.4 | 23 | 1.83 |
| SFT-reverified_full_s0 | registered | final_step | 1e-05 | 81.9 (79.4-84.2) | 82.9 | 82.9 | 100.0 | 100.0 | 0 | 128 | 29.6 (25.8-33.7) | 30.2 | 30.2 | 95.4 | 23 | 2.69 |
| SFT-reverified_sizecontrol_s0 | registered | final_step | 1e-05 | 83.6 (81.2-85.8) | 84.0 | 84.0 | 99.7 | 99.7 | 0 | 137 | 30.2 (26.3-34.4) | 30.2 | 30.2 | 95.0 | 24 | 1.57 |
| SFT-exact_n512_s0 | registered | final_step | 1e-05 | 88.9 (86.8-90.7) | 89.1 | 89.1 | 99.9 | 99.9 | 0 | 130 | 32.4 (28.4-36.6) | 33.4 | 34.6 | 82.6 | 49 | 0.44 |
| SFT-reverified_n512_s0 | registered | final_step | 1e-05 | 82.0 (79.5-84.3) | 83.9 | 83.9 | 99.9 | 99.9 | 0 | 129 | 30.2 (26.3-34.4) | 31.6 | 32.0 | 84.2 | 64 | 0.44 |
| SFT-reverified_n128_s0 | registered | final_step | 1e-05 | 68.0 (65.0-70.8) | 68.5 | 80.5 | 83.6 | 83.6 | 0 | 176 | 23.6 (20.1-27.5) | 24.4 | 36.8 | 54.4 | 95 | 0.12 |
| SFT-reverified_n2048_s0 | registered | final_step | 1e-05 | 84.2 (81.8-86.3) | 84.7 | 84.7 | 100.0 | 100.0 | 0 | 130 | 32.6 (28.6-36.8) | 33.4 | 33.6 | 90.4 | 45 | 1.03 |
| GRPO-tune_lr2e-6 | registered | final_step | 2e-06 | 67.6 (64.6-70.4) | 70.7 | 75.2 | 93.7 | 93.7 | 1 | 137 | 17.4 (14.3-21.0) | 19.8 | 28.0 | 69.0 | 59 | 1.95 |
| GRPO-tune_lr5e-6 | registered | final_step | 5e-06 | 84.6 (82.2-86.7) | 85.0 | 87.5 | 96.5 | 96.5 | 0 | 142 | 29.6 (25.8-33.7) | 30.6 | 33.0 | 88.0 | 43 | 1.83 |
| GRPO-tune_lr1e-5 | registered | final_step | 1e-05 | 87.5 (85.3-89.4) | 87.7 | 87.8 | 99.7 | 99.7 | 0 | 147 | 27.8 (24.1-31.9) | 29.8 | 30.2 | 93.6 | 28 | 1.93 |
| GRPO-A_base_s2026 | FAILED run, diagnostic | step75 | 1e-05 | 92.9 (91.1-94.3) | 92.9 | 92.9 | 100.0 | 86.8 | 132 | 202 | 39.8 (35.6-44.2) | 40.6 | 41.0 | 90.6 | 47 | - |
| GRPO-B_sft512_s2026 | registered | final_step | 1e-05 | 92.8 (91.0-94.2) | 92.8 | 92.8 | 99.8 | 99.8 | 0 | 247 | 47.8 (43.5-52.2) | 48.6 | 50.2 | 85.0 | 75 | 7.35 |
| GRPO-A-replay80-lr1e-5-s2026 | post-hoc branch | final_step | 1e-05 | 92.7 (90.9-94.2) | 92.7 | 92.7 | 100.0 | 100.0 | 0 | 193 | 41.6 (37.4-46.0) | 42.2 | 42.4 | 93.0 | 35 | 2.28 |
| GRPO-A-recovery80-lr5e-6-s2026 | post-hoc branch | final_step | 5e-06 | 91.9 (90.0-93.4) | 91.9 | 91.9 | 100.0 | 5.6 | 944 | 213 | 41.2 (37.0-45.6) | 41.4 | 42.2 | 87.2 | 64 | 2.51 |
