# Package migration and code identity

The public release reorganises the code into a standalone package. Nothing about the training or evaluation logic changed. This page records:
- the mapping from the original layout to the release;
- the exact edits;
- how to check that the release is the code that produced the results.

## Layout

| Original layout (used for every run) | Public release | Change |
|---|---|---|
| `cs336_alignment/drgrpo_grader.py` | `posttrain/verifiers/drgrpo_grader.py` | moved; bytes unchanged |
| `cs336_alignment/prompts/r1_zero.prompt` | `posttrain/prompts/r1_zero.prompt` | moved; bytes unchanged |
| `cs336_alignment/prompts/question_only.prompt` | `posttrain/prompts/question_only.prompt` | moved; bytes unchanged |
| `cs336_alignment/__init__.py` (empty) | not shipped | the legacy package directory no longer exists |
| `posttrain/*.py` | `posttrain/*.py` | import lines and the code-hash helper edited (below) |
| (none) | `posttrain/verifiers/__init__.py`, `posttrain/prompts/__init__.py` | new: package markers and the template loader |
| `pyproject.toml` (legacy metadata) | `pyproject.toml` | replaced: package `llm-posttraining-lab`, `posttrain*` packages |
| `uv.lock` (legacy metadata) | not shipped | `requirements-cloud.lock.txt` pins the exact environment of every run |

## Code identities

| Identity | Value | Meaning |
|---|---|---|
| **Original code hash, evaluation version** | `0e0b00141307ce2f33118c7a918ae2566b10b05e0a5d65078ad466dbd1e51af9` | Recorded in the checkpoint lock. It is the code that ran the held-out evaluation, and the version this release reconstructs |
| Original code hash, training of the 15 pre-registered runs | `739a2df9e838…` | Recorded in their `run.json` (`results/run_records/a5_sessions/`) |
| Original code hash, the two diagnostic branches | `c80348598ca8…` | Recorded in their `run.json` (`results/run_records/a5_recovery/recovery/`) |
| **Release code hash** | `4a7b0e9846bb897427f168fd318fb6156a520dd84a18ab5d341f2fb54236280a` | `posttrain.common.code_hash()` in this repository: every `.py` under `posttrain/`, keyed by path |
| Verifier, original (every run) | `verify.py` `b905bcaa4945…`, grader `f4fa06fed495…` | `posttrain/verifier_lock.py` `ORIGINAL_SHA256` |
| Verifier, release | `verify.py` `22ce23b849fb…`, grader `f4fa06fed495…` | `posttrain/verifier_lock.py` `SHA256`. Verifier version 2.0.0 is unchanged; only `verify.py`'s import lines differ |

### What the training-time code versions differ in

The three original code versions differ as follows.
- **Unchanged from training to evaluation:** the files that define the training objectives and rewards. These are `losses.py`, `train_sft.py`, `batching.py`, `engines.py`, `evaluate.py`, `verify.py`, `verifier_lock.py` and the grader. Every run recorded the same verifier hashes.
- **Changed between the training version (`739a2df9…`) and the evaluation version (`0e0b0014…`):** the changes add failure handling and recovery, and leave the update step unchanged.
  - `train_grpo.py`: failed-batch capture at the gates, fork support for diagnostic branches, extra per-step diagnostics.
  - `common.py`: refusal to resume a failed run, rewind on resume.
  - `model_io.py`: a state-completeness check on resume.
  - `diagnostics.py`: an AdamW update-size statistic computed from optimizer state.
  - New files: `fork_grpo.py`, `final_eval.py`.
- **Changed between the branch version (`c8034859…`) and the evaluation version:** only `final_eval.py` was added.

These comparisons were made against the private deployment packages of each version, which are not published. In this repository, the evaluation version can be checked directly; the statements about the earlier versions are documented, not reproducible here.

## Checks

**1. Reconstruction.** `python scripts/verify_code_lineage.py` uses `docs/code_lineage.json`. It reverses every edit below, in memory, and checks:
- that every file of the original layout regains its original sha256;
- that the original code hash equals the checkpoint lock;
- that the reward code equals the verifier hashes recorded by all training runs.

**2. Semantic equivalence (run at release time).** Both layouts were imported side by side, and the same inputs went through:
- the reward functions (131 answer/reference cases, including all golden verifier cases);
- the prompt templates;
- the group advantages and every policy-loss variant;
- the SFT loss and micro-batching.

| Item | Original vs release |
|---|---|
| adv | identical |
| clean_format | identical |
| fns | identical |
| golden_ok | identical |
| microbatches | identical |
| policy_loss | identical |
| score_question_only | identical |
| score_r1_zero | identical |
| sft_loss_sum | identical |
| templates | identical |

**3. CPU regression tests (release layout).** This repository as cloned: 168 passed, 6 skipped, 1 warning. With the private inputs restored (datasets and one historical audit file): 174 passed, 1 warning.

**What these checks do not show.** They do not show that this release trained the reported checkpoints. The checkpoints were produced by the original layout, at the code versions in the table above. The release is a reorganised copy of the evaluation version.

## Exact edits

All edits replace import paths, or the code-hash helper that enumerates files. The verifier lock gains the release hash and keeps the original.

**`posttrain/verify.py`**

```diff
- sys.path.insert(0, os.path.join(ROOT, "cs336_alignment"))
- from drgrpo_grader import extract_answer, grade, question_only_reward_fn, r1_zero_reward_fn  # noqa: E402
+ from .verifiers.drgrpo_grader import extract_answer, grade, question_only_reward_fn, r1_zero_reward_fn  # noqa: E402
```

**`posttrain/verify.py`**

```diff
- grader_py=h(os.path.join(ROOT, "cs336_alignment", "drgrpo_grader.py"))
+ grader_py=h(os.path.join(here, "verifiers", "drgrpo_grader.py"))
```

**`posttrain/evaluate.py`**

```diff
- from .verify import clean_format, score
+ from .verify import clean_format, score
+ from .prompts import load as load_prompt
```

**`posttrain/evaluate.py`**

```diff
-     return open(os.path.join(ROOT, "cs336_alignment", "prompts", name + ".prompt"), encoding="utf-8").read().strip()
+     return load_prompt(name)
```

**`posttrain/evaluate.py`**

```diff
-     from drgrpo_grader import extract_answer, grade
+     from .verifiers.drgrpo_grader import extract_answer, grade
```

**`posttrain/common.py`**

```diff
-     for d in ("posttrain", "cs336_alignment"):
-         for f in sorted(os.listdir(os.path.join(ROOT, d))):
-             if f.endswith(".py"):
-                 h.update(f.encode() + open(os.path.join(ROOT, d, f), "rb").read().replace(b"\r\n", b"\n"))
+     # Release layout: every .py under posttrain/ (subpackages included), keyed by its path. The hash of the original layout,
+     # recorded by the runs and the checkpoint lock, is reproduced by scripts/verify_code_lineage.py.
+     fs = sorted(os.path.relpath(os.path.join(d, n), ROOT).replace(os.sep, "/") for d, _, ns in os.walk(os.path.join(ROOT, "posttrain")) for n in ns if n.endswith(".py"))
+     for f in fs:
+         h.update(f.encode() + open(os.path.join(ROOT, f), "rb").read().replace(b"\r\n", b"\n"))
```

**`posttrain/__init__.py`**

```diff
- """Configurable SFT / Expert Iteration / GRPO extension.
+ """Configurable SFT / Expert Iteration / GRPO post-training system.
```

**`posttrain/__init__.py`**

```diff
- Third-party code reused: cs336_alignment/drgrpo_grader.py (sail-sg/understand-r1-zero, Apache-2.0).
+ Third-party code reused: posttrain/verifiers/drgrpo_grader.py (Dr. GRPO grader, Apache-2.0); see THIRD_PARTY_NOTICES.md.
```

**`posttrain/verifier_lock.py`**

```diff
- verify_py="b905bcaa4945be2e94286664a5fe9659c30c364ca973b4d90007463de89df106"
+ verify_py="22ce23b849fb03bf71cf73e2fa86865dd8edf9a898c63e684b7baa11ffdf7639"
```

**`posttrain/verifier_lock.py`**

```diff
- VERSION = "2.0.0"
+ VERSION = "2.0.0"
+ # Release identity. Every run in results/run_records was produced with ORIGINAL_SHA256 (original package layout). The two differ only
+ # because posttrain/verify.py now imports the unchanged grader from posttrain/verifiers/; scripts/verify_code_lineage.py
+ # reconstructs the original files from this release and checks both hashes.
+ ORIGINAL_SHA256 = dict(verify_py="b905bcaa4945be2e94286664a5fe9659c30c364ca973b4d90007463de89df106", grader_py="f4fa06fed4950d6bd66a2fc40faaa67943327c2064e9d7ced1accdaa19ed25f0")
```

**`scripts/stage0_build_data.py`**

```diff
- sys.path.insert(0, os.path.join(HERE, "..", "cs336_alignment"))
+ sys.path.insert(0, os.path.join(HERE, ".."))
```

**`scripts/stage0_build_data.py`**

```diff
- from drgrpo_grader import r1_zero_reward_fn  # noqa: E402
+ from posttrain.verifiers.drgrpo_grader import r1_zero_reward_fn  # noqa: E402
```

**`scripts/stage0_verify_audit.py`**

```diff
- sys.path.insert(0, os.path.join(HERE, "..", "cs336_alignment"))
+ sys.path.insert(0, os.path.join(HERE, ".."))
```

**`scripts/stage0_verify_audit.py`**

```diff
- from drgrpo_grader import grade, r1_zero_reward_fn  # noqa: E402
+ from posttrain.verifiers.drgrpo_grader import grade, r1_zero_reward_fn  # noqa: E402
```

**`scripts/stage0_failure_decomposition.py`**

```diff
- PROJ = os.path.expanduser("<repo>")
- sys.path.insert(0, PROJ + "/cs336_alignment")
- from drgrpo_grader import grade, extract_answer
+ PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
+ sys.path.insert(0, PROJ)
+ from posttrain.verifiers.drgrpo_grader import grade, extract_answer
```

**`scripts/stage0_base_eval_hf.py`**

```diff
- PROJ = os.path.expanduser("<repo>")
- sys.path.insert(0, PROJ + "/cs336_alignment")
+ PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
+ sys.path.insert(0, PROJ)
```

**`scripts/stage0_base_eval_hf.py`**

```diff
- from drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
+ from posttrain.verifiers.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
```

**`scripts/stage0_base_eval_hf.py`**

```diff
-     tpl = open(f"cs336_alignment/prompts/{name}.prompt", encoding="utf-8").read().strip()
+     from posttrain.prompts import load
+     tpl = load(name)
```

**`scripts/verifier_crosscheck.py`**

```diff
- from drgrpo_grader import extract_answer, r1_zero_reward_fn
+ from posttrain.verifiers.drgrpo_grader import extract_answer, r1_zero_reward_fn
```

**`tests/test_posttrain.py`**

```diff
-     from drgrpo_grader import r1_zero_reward_fn
+     from posttrain.verifiers.drgrpo_grader import r1_zero_reward_fn
```
