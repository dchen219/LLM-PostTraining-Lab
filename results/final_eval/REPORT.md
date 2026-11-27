# Final held-out evaluation (generated)

Freeze sha256 ff0f0ce574bcb47c9dafedc77bab1654362bb955a81c593960108ca682b7a453. Not evaluated (missing): none.
Problems: none.

GRPO-A_base_s2026 (pre-registered) FAILED at step 98 and has no final checkpoint; comparison C4 is unavailable and counted with p = 1. Rows marked post-hoc are diagnostic branches started from its step-80 state; they do not replace it.

| checkpoint | class | prompt | set | strict % (95% CI) | second | relaxed | format | clean | doubled </think> | tokens mean/median/p90 | correct/incorrect tokens | truncated | train GPU-h | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BASE | preregistered | r1_zero | gsm8k_test_final | 17.7 (15.8-19.9) | 32.1 | 44.9 | 51.1 | 80.0 | 0 | 152/129/239 | 121/159 | 8 (0.6%) | - | 4019 |
| BASE | preregistered | r1_zero | math_val_final | 17.4 (16.4-18.5) | 23.3 | 43.6 | 42.6 | 57.2 | 7 | 322/198/897 | 153/358 | 407 (8.1%) | - | 3914 |
| BASE | preregistered | question_only | gsm8k_test_final | 65.7 (63.1-68.2) | 65.7 | 73.2 | 100.0 | - | 0 | 349/329/518 | 349/348 | 15 (1.1%) | - | 2934 |
| BASE | preregistered | question_only | math_val_final | 58.6 (57.2-59.9) | 58.6 | 61.6 | 100.0 | - | 0 | 566/514/1024 | 496/665 | 634 (12.7%) | - | 3334 |
| SFT-exact_full_s0 | preregistered | r1_zero | gsm8k_test_final | 81.3 (79.2-83.4) | 81.6 | 81.6 | 99.8 | 99.8 | 0 | 139/130/204 | 131/176 | 2 (0.2%) | 1.83 | 3904 |
| SFT-exact_full_s0 | preregistered | r1_zero | math_val_final | 62.5 (61.2-63.9) | 63.1 | 63.2 | 98.0 | 98.0 | 0 | 272/207/533 | 198/395 | 95 (1.9%) | 1.83 | 4038 |
| SFT-reverified_full_s0 | preregistered | r1_zero | gsm8k_test_final | 72.0 (69.5-74.4) | 72.6 | 72.6 | 99.9 | 99.9 | 0 | 134/124/195 | 125/157 | 1 (0.1%) | 2.69 | 3135 |
| SFT-reverified_full_s0 | preregistered | r1_zero | math_val_final | 60.5 (59.2-61.9) | 61.1 | 61.2 | 98.4 | 98.4 | 0 | 264/202/519 | 191/377 | 81 (1.6%) | 2.69 | 2600 |
| SFT-reverified_sizecontrol_s0 | preregistered | r1_zero | gsm8k_test_final | 74.9 (72.5-77.2) | 75.4 | 75.4 | 99.8 | 99.8 | 0 | 142/131/208 | 132/171 | 2 (0.2%) | 1.57 | 2205 |
| SFT-reverified_sizecontrol_s0 | preregistered | r1_zero | math_val_final | 62.1 (60.8-63.5) | 63.0 | 63.0 | 98.2 | 98.2 | 0 | 284/218/562 | 206/410 | 87 (1.7%) | 1.57 | 2781 |
| SFT-exact_n512_s0 | preregistered | r1_zero | gsm8k_test_final | 81.3 (79.1-83.3) | 81.3 | 81.3 | 99.5 | 99.5 | 0 | 139/126/199 | 127/190 | 7 (0.5%) | 0.44 | 2120 |
| SFT-exact_n512_s0 | preregistered | r1_zero | math_val_final | 61.9 (60.6-63.3) | 63.1 | 63.7 | 92.9 | 92.9 | 0 | 302/215/662 | 215/445 | 230 (4.6%) | 0.44 | 2585 |
| SFT-reverified_n512_s0 | preregistered | r1_zero | gsm8k_test_final | 72.7 (70.2-75.0) | 74.9 | 74.9 | 99.6 | 99.6 | 0 | 138/124/201 | 126/168 | 5 (0.4%) | 0.44 | 2812 |
| SFT-reverified_n512_s0 | preregistered | r1_zero | math_val_final | 60.8 (59.4-62.1) | 62.7 | 62.9 | 94.4 | 94.4 | 0 | 295/207/641 | 208/429 | 223 (4.5%) | 0.44 | 3245 |
| SFT-reverified_n128_s0 | preregistered | r1_zero | gsm8k_test_final | 61.5 (58.8-64.1) | 61.7 | 72.5 | 82.5 | 82.5 | 0 | 188/138/318 | 135/271 | 38 (2.9%) | 0.12 | 3818 |
| SFT-reverified_n128_s0 | preregistered | r1_zero | math_val_final | 49.7 (48.4-51.1) | 50.4 | 61.8 | 74.0 | 74.0 | 0 | 342/239/832 | 216/467 | 358 (7.2%) | 0.12 | 4175 |
| SFT-reverified_n2048_s0 | preregistered | r1_zero | gsm8k_test_final | 76.0 (73.6-78.2) | 76.3 | 76.3 | 100.0 | 100.0 | 0 | 134/126/199 | 128/156 | 0 (0.0%) | 1.03 | 2626 |
| SFT-reverified_n2048_s0 | preregistered | r1_zero | math_val_final | 62.6 (61.3-64.0) | 63.4 | 63.5 | 97.2 | 97.2 | 0 | 285/214/574 | 206/417 | 125 (2.5%) | 1.03 | 3119 |
| GRPO-B_sft512_s2026 | preregistered | r1_zero | gsm8k_test_final | 86.1 (84.2-87.9) | 86.1 | 86.1 | 99.9 | 99.9 | 0 | 253/240/359 | 244/310 | 1 (0.1%) | 7.35 | 2171 |
| GRPO-B_sft512_s2026 | preregistered | r1_zero | math_val_final | 72.5 (71.3-73.8) | 73.1 | 73.4 | 94.3 | 94.3 | 0 | 460/395/867 | 384/662 | 283 (5.7%) | 7.35 | 3547 |
| GRPO-A-replay80-lr1e-5-s2026 | posthoc | r1_zero | gsm8k_test_final | 85.5 (83.5-87.3) | 85.5 | 85.5 | 100.0 | 100.0 | 0 | 198/184/278 | 193/232 | 0 (0.0%) | 2.28 | 3100 |
| GRPO-A-replay80-lr1e-5-s2026 | posthoc | r1_zero | math_val_final | 70.5 (69.2-71.7) | 71.3 | 71.3 | 97.6 | 97.6 | 1 | 384/324/703 | 326/522 | 119 (2.4%) | 2.28 | 2615 |
| GRPO-A-recovery80-lr5e-6-s2026 | posthoc | r1_zero | gsm8k_test_final | 84.5 (82.4-86.3) | 84.5 | 84.5 | 99.9 | 6.4 | 1233 | 225/212/312 | 217/268 | 1 (0.1%) | 2.51 | 5810 |
| GRPO-A-recovery80-lr5e-6-s2026 | posthoc | r1_zero | math_val_final | 70.6 (69.3-71.9) | 71.4 | 71.6 | 95.9 | 8.0 | 4406 | 431/364/804 | 360/600 | 204 (4.1%) | 2.51 | 5952 |

| family | comparison | set | a only | b only | diff pp (95% CI) | p | Holm p | second diff | relaxed diff | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| pre-registered confirmatory | C1 data quality, full | gsm8k_test_final | 182 | 59 | -9.3 [-11.6, -7.1] | 8.88e-16 | 2.66e-15 | -9.0 | -9.0 | supported |
| pre-registered confirmatory | C2 data quality, matched 512 | gsm8k_test_final | 171 | 58 | -8.6 [-10.8, -6.4] | 4.02e-14 | 8.04e-14 | -6.4 | -6.4 | supported |
| pre-registered confirmatory | C3 pathway B vs its SFT parent | gsm8k_test_final | 60 | 237 | +13.4 [+11.0, +15.9] | 4.84e-26 | 1.94e-25 | +11.2 | +11.2 | supported |
| pre-registered confirmatory | C4 pathway A vs base | gsm8k_test_final | - | - | - | - | 1 | - | - | unavailable: run failed; counted with p = 1 so the family is not shrunk after the failure |
| pre-registered confirmatory | C1 data quality, full | math_val_final | 492 | 392 | -2.0 [-3.2, -0.8] | 0.000859 | 0.00258 | -2.0 | -2.0 | supported |
| pre-registered confirmatory | C2 data quality, matched 512 | math_val_final | 470 | 414 | -1.1 [-2.3, +0.0] | 0.0643 | 0.129 | -0.4 | -0.8 | not detected |
| pre-registered confirmatory | C3 pathway B vs its SFT parent | math_val_final | 256 | 843 | +11.7 [+10.5, +13.0] | 1.43e-73 | 5.71e-73 | +10.5 | +10.5 | supported |
| pre-registered confirmatory | C4 pathway A vs base | math_val_final | - | - | - | - | 1 | - | - | unavailable: run failed; counted with p = 1 so the family is not shrunk after the failure |
| pre-registered secondary | SFT exact_full vs base question-only | gsm8k_test_final | 93 | 300 | +15.7 [+12.9, +18.5] | 1.67e-26 | 9.99e-26 | +15.9 | +8.3 | supported |
| pre-registered secondary | SFT reverified_full vs base question-only | gsm8k_test_final | 169 | 253 | +6.4 [+3.3, +9.4] | 5.06e-05 | 0.000101 | +6.9 | -0.7 | supported |
| pre-registered secondary | SFT sizecontrol vs base question-only | gsm8k_test_final | 140 | 262 | +9.2 [+6.3, +12.2] | 1.2e-09 | 4.78e-09 | +9.8 | +2.2 | supported |
| pre-registered secondary | SFT exact_n512 vs base question-only | gsm8k_test_final | 90 | 296 | +15.6 [+12.8, +18.4] | 9.52e-27 | 6.66e-26 | +15.7 | +8.1 | supported |
| pre-registered secondary | SFT reverified_n512 vs base question-only | gsm8k_test_final | 169 | 262 | +7.1 [+4.0, +10.1] | 8.68e-06 | 2.6e-05 | +9.2 | +1.7 | supported |
| pre-registered secondary | SFT reverified_n128 vs base question-only | gsm8k_test_final | 268 | 213 | -4.2 [-7.4, -0.9] | 0.0137 | 0.0137 | -3.9 | -0.8 | supported |
| pre-registered secondary | SFT reverified_n2048 vs base question-only | gsm8k_test_final | 146 | 282 | +10.3 [+7.3, +13.3] | 4.69e-11 | 2.35e-10 | +10.6 | +3.0 | supported |
| pre-registered secondary | SFT exact_full vs base question-only | math_val_final | 535 | 733 | +4.0 [+2.6, +5.4] | 2.97e-08 | 1.49e-07 | +4.6 | +1.6 | supported |
| pre-registered secondary | SFT reverified_full vs base question-only | math_val_final | 612 | 710 | +2.0 [+0.5, +3.4] | 0.00761 | 0.00761 | +2.6 | -0.4 | supported |
| pre-registered secondary | SFT sizecontrol vs base question-only | math_val_final | 551 | 730 | +3.6 [+2.2, +5.0] | 6.32e-07 | 2.53e-06 | +4.4 | +1.4 | supported |
| pre-registered secondary | SFT exact_n512 vs base question-only | math_val_final | 562 | 730 | +3.4 [+2.0, +4.8] | 3.28e-06 | 9.85e-06 | +4.5 | +2.1 | supported |
| pre-registered secondary | SFT reverified_n512 vs base question-only | math_val_final | 617 | 729 | +2.2 [+0.8, +3.7] | 0.00247 | 0.00494 | +4.1 | +1.3 | supported |
| pre-registered secondary | SFT reverified_n128 vs base question-only | math_val_final | 968 | 527 | -8.8 [-10.3, -7.3] | 2e-30 | 1.4e-29 | -8.1 | +0.3 | supported |
| pre-registered secondary | SFT reverified_n2048 vs base question-only | math_val_final | 525 | 729 | +4.1 [+2.7, +5.5] | 9.2e-09 | 5.52e-08 | +4.9 | +2.0 | supported |
| descriptive | composition at matched size | gsm8k_test_final | 144 | 59 | -6.4 [-8.5, -4.4] | 2.19e-09 | - | -6.1 | -6.1 | descriptive (p < alpha) |
| descriptive | volume | gsm8k_test_final | 111 | 73 | -2.9 [-4.9, -0.9] | 0.00622 | - | -2.9 | -2.9 | descriptive (p < alpha) |
| descriptive | sample efficiency 128 -> 512 | gsm8k_test_final | 139 | 287 | +11.2 [+8.2, +14.2] | 6.26e-13 | - | +13.2 | +2.4 | descriptive (p < alpha) |
| descriptive | sample efficiency 512 -> 2048 | gsm8k_test_final | 92 | 135 | +3.3 [+1.0, +5.5] | 0.0052 | - | +1.4 | +1.4 | descriptive (p < alpha) |
| descriptive | sample efficiency 2048 -> full | gsm8k_test_final | 123 | 71 | -3.9 [-6.0, -1.9] | 0.000232 | - | -3.7 | -3.7 | descriptive (p < alpha) |
| descriptive | pathway B vs base question-only | gsm8k_test_final | 50 | 320 | +20.5 [+17.8, +23.1] | 2.59e-49 | - | +20.5 | +12.9 | descriptive (p < alpha) |
| descriptive | pathway B vs base r1_zero | gsm8k_test_final | 11 | 913 | +68.4 [+65.8, +71.0] | 1.41e-253 | - | +54.0 | +41.2 | descriptive (p < alpha) |
| descriptive | base prompts: question-only vs r1_zero | gsm8k_test_final | 57 | 689 | +47.9 [+44.8, +51.0] | 9.01e-139 | - | +33.5 | +28.4 | descriptive (p < alpha) |
| descriptive | composition at matched size | math_val_final | 455 | 436 | -0.4 [-1.6, +0.8] | 0.547 | - | -0.2 | -0.2 | descriptive |
| descriptive | volume | math_val_final | 487 | 406 | -1.6 [-2.8, -0.4] | 0.00739 | - | -1.8 | -1.8 | descriptive (p < alpha) |
| descriptive | sample efficiency 128 -> 512 | math_val_final | 430 | 983 | +11.1 [+9.6, +12.5] | 4.33e-50 | - | +12.2 | +1.1 | descriptive (p < alpha) |
| descriptive | sample efficiency 512 -> 2048 | math_val_final | 473 | 565 | +1.8 [+0.6, +3.1] | 0.00471 | - | +0.8 | +0.6 | descriptive (p < alpha) |
| descriptive | sample efficiency 2048 -> full | math_val_final | 512 | 406 | -2.1 [-3.3, -0.9] | 0.000522 | - | -2.3 | -2.4 | descriptive (p < alpha) |
| descriptive | pathway B vs base question-only | math_val_final | 231 | 930 | +14.0 [+12.7, +15.3] | 9.61e-100 | - | +14.6 | +11.8 | descriptive (p < alpha) |
| descriptive | pathway B vs base r1_zero | math_val_final | 61 | 2818 | +55.1 [+53.7, +56.6] | 0 | - | +49.9 | +29.8 | descriptive (p < alpha) |
| descriptive | base prompts: question-only vs r1_zero | math_val_final | 169 | 2227 | +41.2 [+39.6, +42.7] | 0 | - | +35.3 | +18.0 | descriptive (p < alpha) |
| post-hoc exploratory | A1 (lr 1e-5) vs base r1_zero | gsm8k_test_final | 6 | 900 | +67.8 [+65.2, +70.4] | 2.81e-258 | 1.12e-257 | +53.4 | +40.6 | exploratory (p < alpha) |
| post-hoc exploratory | A2 (lr 5e-6) vs base r1_zero | gsm8k_test_final | 9 | 889 | +66.7 [+64.1, +69.3] | 9.61e-250 | 2.88e-249 | +52.3 | +39.6 | exploratory (p < alpha) |
| post-hoc exploratory | A1 vs base question-only | gsm8k_test_final | 45 | 307 | +19.9 [+17.3, +22.4] | 4.44e-49 | 8.87e-49 | +19.9 | +12.3 | exploratory (p < alpha) |
| post-hoc exploratory | A2 vs A1 | gsm8k_test_final | 63 | 49 | -1.1 [-2.6, +0.5] | 0.219 | 0.219 | -1.1 | -1.1 | exploratory |
| post-hoc exploratory | A1 (lr 1e-5) vs base r1_zero | math_val_final | 72 | 2725 | +53.1 [+51.6, +54.5] | 0 | 0 | +48.0 | +27.7 | exploratory (p < alpha) |
| post-hoc exploratory | A2 (lr 5e-6) vs base r1_zero | math_val_final | 64 | 2725 | +53.2 [+51.8, +54.7] | 0 | 0 | +48.1 | +28.0 | exploratory (p < alpha) |
| post-hoc exploratory | A1 vs base question-only | math_val_final | 309 | 904 | +11.9 [+10.6, +13.2] | 5.03e-68 | 1.01e-67 | +12.7 | +9.7 | exploratory (p < alpha) |
| post-hoc exploratory | A2 vs A1 | math_val_final | 325 | 333 | +0.2 [-0.8, +1.2] | 0.785 | 0.785 | +0.1 | +0.3 | exploratory |
