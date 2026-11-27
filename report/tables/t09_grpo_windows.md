| Run | Steps | Zero-advantage groups % | Entropy (nats) | Reward | Mean length | GSM8K dev gain pp | MATH dev gain pp | GPU-h |
|---|---|---|---|---|---|---|---|---|
| GRPO-B (sequential) | 0-25 | 54 | 0.29 | 0.65 | 373 | +10.7 | +15.6 | 1.65 |
| GRPO-B (sequential) | 25-50 | 70 | 0.06 | 0.76 | 413 | +1.0 | −2.8 | 1.71 |
| GRPO-B (sequential) | 50-75 | 72 | 0.06 | 0.77 | 432 | −0.6 | +1.4 | 1.68 |
| GRPO-B (sequential) | 75-100 | 65 | 0.11 | 0.73 | 472 | −0.3 | +3.4 | 1.75 |
| GRPO-A (direct, FAILED at 98) | 0-25 | 48 | 0.29 | 0.40 | 259 | +67.7 | +23.6 | 1.45 |
| GRPO-A (direct, FAILED at 98) | 25-50 | 62 | 0.05 | 0.68 | 408 | +2.9 | +8.4 | 1.65 |
| GRPO-A (direct, FAILED at 98) | 50-75 | 66 | 0.04 | 0.74 | 423 | +0.5 | +0.6 | 1.66 |

Development sets only (exploratory). Source: `data/grpo_compute_windows.csv`.
