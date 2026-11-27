| Comparison (a → b) | Set | a only | b only | b − a, pp [95% CI] | relaxed b − a | p (exact McNemar) | Holm p | Reading |
|---|---|---|---|---|---|---|---|---|
| A1 (lr 1e-5) vs base r1_zero | GSM8K | 6 | 900 | +67.8 [+65.2, +70.4] | +40.6 | 2.8e-258 | 1.1e-257 | difference detected: b > a |
| A2 (lr 5e-6) vs base r1_zero | GSM8K | 9 | 889 | +66.7 [+64.1, +69.3] | +39.6 | 9.6e-250 | 2.9e-249 | difference detected: b > a |
| A1 vs base question-only | GSM8K | 45 | 307 | +19.9 [+17.3, +22.4] | +12.3 | 4.4e-49 | 8.9e-49 | difference detected: b > a |
| A2 vs A1 | GSM8K | 63 | 49 | −1.1 [−2.6, +0.5] | −1.1 | 0.219 | 0.219 | no difference detected (not evidence of equivalence) |
| A1 (lr 1e-5) vs base r1_zero | MATH | 72 | 2725 | +53.1 [+51.6, +54.5] | +27.7 | <1e-300 | <1e-300 | difference detected: b > a |
| A2 (lr 5e-6) vs base r1_zero | MATH | 64 | 2725 | +53.2 [+51.8, +54.7] | +28.0 | <1e-300 | <1e-300 | difference detected: b > a |
| A1 vs base question-only | MATH | 309 | 904 | +11.9 [+10.6, +13.2] | +9.7 | 5.0e-68 | 1.0e-67 | difference detected: b > a |
| A2 vs A1 | MATH | 325 | 333 | +0.2 [−0.8, +1.2] | +0.3 | 0.785 | 0.785 | no difference detected (not evidence of equivalence) |
| Direct (A1, post-hoc branch) vs sequential (GRPO-B) | GSM8K | 80 | 88 | +0.6 [−1.3, +2.5] | +0.6 | 0.589 | 1.000 | exploratory: no difference detected, b > a |
| Direct (A2, post-hoc branch) vs sequential (GRPO-B) | GSM8K | 71 | 93 | +1.7 [−0.2, +3.6] | +1.7 | 0.101 | 0.302 | exploratory: no difference detected, b > a |
| GRPO-B vs SFT exact n512 (other data source, same size) | GSM8K | 66 | 130 | +4.9 [+2.8, +6.9] | +4.8 | 5.7e-6 | 2.3e-5 | exploratory: difference (Holm p < 0.05), b > a |
| Exact-match scaling 512 -> 3,420 | GSM8K | 89 | 90 | +0.1 [−1.9, +2.1] | +0.2 | 1.000 | 1.000 | exploratory: no difference detected, b > a |
| Composition at 512 vs at full size: exact n512 -> re-verified n2048 | GSM8K | 149 | 79 | −5.3 [−7.5, −3.1] | −5.1 | 4.1e-6 | 2.1e-5 | exploratory: difference (Holm p < 0.05), b < a |
| Direct (A1, post-hoc branch) vs sequential (GRPO-B) | MATH | 322 | 426 | +2.1 [+1.0, +3.2] | +2.1 | 1.6e-4 | 6.5e-4 | exploratory: difference (Holm p < 0.05), b > a |
| Direct (A2, post-hoc branch) vs sequential (GRPO-B) | MATH | 330 | 426 | +1.9 [+0.8, +3.0] | +1.8 | 5.4e-4 | 0.002 | exploratory: difference (Holm p < 0.05), b > a |
| GRPO-B vs SFT exact n512 (other data source, same size) | MATH | 238 | 769 | +10.6 [+9.4, +11.8] | +9.7 | 8.8e-66 | 4.4e-65 | exploratory: difference (Holm p < 0.05), b > a |
| Exact-match scaling 512 -> 3,420 | MATH | 503 | 533 | +0.6 [−0.7, +1.9] | −0.5 | 0.368 | 0.561 | exploratory: no difference detected, b > a |
| Composition at 512 vs at full size: exact n512 -> re-verified n2048 | MATH | 508 | 544 | +0.7 [−0.6, +2.0] | −0.2 | 0.281 | 0.561 | exploratory: no difference detected, b > a |

Post-hoc. 'frozen' = the A1/A2 family listed in freeze_v1.yaml (Holm m = 4); 'unregistered' = chosen after the held-out results (Holm m = 5). Exploratory only.
