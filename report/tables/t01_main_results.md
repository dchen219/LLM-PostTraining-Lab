| Checkpoint | Class | GSM8K strict % (95% CI) | MATH strict % (95% CI) | GSM8K relaxed % | MATH relaxed % | GSM8K format % | MATH format % | Cumulative GPU-h |
|---|---|---|---|---|---|---|---|---|
| Base, r1_zero prompt | pre-registered | 17.7 (15.8–19.9) | 17.4 (16.4–18.5) | 44.9 | 43.6 | 51.1 | 42.6 | 0.00 |
| Base, question-only prompt | pre-registered | 65.7 (63.1–68.2) | 58.6 (57.2–59.9) | 73.2 | 61.6 | 100.0 | 100.0 | 0.00 |
| SFT re-verified n128 | pre-registered | 61.5 (58.8–64.1) | 49.7 (48.4–51.1) | 72.5 | 61.8 | 82.5 | 74.0 | 0.12 |
| SFT re-verified n512 | pre-registered | 72.7 (70.2–75.0) | 60.8 (59.4–62.1) | 74.9 | 62.9 | 99.6 | 94.4 | 0.44 |
| SFT exact n512 | pre-registered | 81.3 (79.1–83.3) | 61.9 (60.6–63.3) | 81.3 | 63.7 | 99.5 | 92.9 | 0.44 |
| SFT re-verified n2048 | pre-registered | 76.0 (73.6–78.2) | 62.6 (61.3–64.0) | 76.3 | 63.5 | 100.0 | 97.2 | 1.03 |
| SFT re-verified size control | pre-registered | 74.9 (72.5–77.2) | 62.1 (60.8–63.5) | 75.4 | 63.0 | 99.8 | 98.2 | 1.57 |
| SFT exact full | pre-registered | 81.3 (79.2–83.4) | 62.5 (61.2–63.9) | 81.6 | 63.2 | 99.8 | 98.0 | 1.83 |
| SFT re-verified full | pre-registered | 72.0 (69.5–74.4) | 60.5 (59.2–61.9) | 72.6 | 61.2 | 99.9 | 98.4 | 2.69 |
| GRPO-B (sequential) | pre-registered | 86.1 (84.2–87.9) | 72.5 (71.3–73.8) | 86.1 | 73.4 | 99.9 | 94.3 | 7.79 |
| A1 (post-hoc, lr 1e-5) | post-hoc | 85.5 (83.5–87.3) | 70.5 (69.2–71.7) | 85.5 | 71.3 | 100.0 | 97.6 | 7.37 |
| A2 (post-hoc, lr 5e-6) | post-hoc | 84.5 (82.4–86.3) | 70.6 (69.3–71.9) | 84.5 | 71.6 | 99.9 | 95.9 | 7.59 |
| GRPO-A, direct (pre-registered) | pre-registered | FAILED at step 98 | FAILED | – | – | – | – | 6.61 (sunk) |

Source: `results/research_analysis/consolidated_results.csv`. Wilson 95% CI. GPU-h = 2 × training wall-clock on 2× RTX A6000, cumulative over the lineage (GRPO-B includes its SFT parent; A1/A2 include GRPO-A steps 1–80).
