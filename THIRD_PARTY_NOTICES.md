# Third-party notices and provenance

This repository's own code and documents are covered by [`LICENSE`](LICENSE). The items below are third-party and keep their own terms. They are not this project's work.

## Included in this repository

### `posttrain/verifiers/drgrpo_grader.py`: strict math grader

- **Licence:** Apache License 2.0. The full text is in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
- **Copyright:** 2025 Garena Online Private Limited. The copyright and licence header at the top of the file is kept unchanged.
- **Origin:** the file's docstring says it comes from `sail-sg/understand-r1-zero` (`understand_r1_zero/math_grader.py`) and is "Based on HF math_verify, verl, open reasoner zero, etc."
  - That upstream repository is MIT-licensed: "Copyright (c) 2025-present Zichen". The full text is in [`LICENSES/MIT-understand-r1-zero.txt`](LICENSES/MIT-understand-r1-zero.txt).
- **How it reached this project:** the copy used here is the one distributed with the public Stanford CS336 (Spring 2025) alignment assignment materials.
- **Changes:** none to its content.
  - It is byte-identical to the copy used for every run; its sha256 `f4fa06fe…` is recorded by every run's verifier.
  - In this public release it was moved from the original package directory (`cs336_alignment/`) to `posttrain/verifiers/`. See [`docs/PACKAGE_MIGRATION.md`](docs/PACKAGE_MIGRATION.md).

### `posttrain/prompts/r1_zero.prompt`: R1-Zero prompt template

- **Content:** the DeepSeek-R1-Zero conversation format ("A conversation between User and Assistant. … `<think>` … `<answer>` …").
- **Licence basis:** the text is identical, up to surrounding whitespace, to `apply_r1_template` in the MIT-licensed `sail-sg/understand-r1-zero` (`train_zero_math.py`, commit `dfca49dd`). It is included under that licence ([`LICENSES/MIT-understand-r1-zero.txt`](LICENSES/MIT-understand-r1-zero.txt)). The loader strips surrounding whitespace.
- **The file used for the runs:** a byte-identical copy distributed with the Stanford CS336 alignment assignment materials (sha256 `56213732…`). It is the same file that produced the frozen results.

### `posttrain/prompts/question_only.prompt`

- **Content:** `{question}`, the bare question placeholder, with no other text. It is the template used for the base model's question-only baseline.

## Used but not redistributed

| Item | Licence (as stated by the source) | What this repository contains |
|---|---|---|
| `garg-aayush/sft-cs336-assign5-datasets` (Hugging Face): the gpt-oss-120b reasoning traces on math12k | Apache-2.0 | Trace ids, content hashes, counts and recovery decisions only. No trace text |
| Qwen2.5-Math-1.5B (Hugging Face) | Apache-2.0 | The pinned revision id and the weight hash only |
| GSM8K, MATH (via the dataset above and `hiyouga/math12k`) | See the original sources | Aggregate scores and contamination statistics only. No question text |
| Python dependencies (`requirements-cloud.lock.txt`) | Their own licences | Version pins only |

## Provenance of the experimental setting

- **Shared starting point:** the choice of base model, benchmarks, R1-Zero prompt and verifier-based SFT/GRPO recipe follows the public Stanford CS336 (Spring 2025) alignment materials. The grader and prompt above were obtained from those materials.
- **Not included:** no other course material (handouts, starter code, tests) is included.
- **Tutorial code:** the project's first working copy came from the public `datawhalechina/diy-llm` tutorial, which is licensed CC BY-NC-SA 4.0. None of the tutorial's code, notebook or images are in this repository. A line-by-line comparison found no shared code with `posttrain/` beyond import statements.
- **This project's own work:**
  - the post-training system in `posttrain/` (except `posttrain/verifiers/drgrpo_grader.py`);
  - the scripts, tests and configuration;
  - the data re-verification, the experiments, the analysis and the report.

  It was developed with AI coding assistance (Claude Code) under the author's direction.

## Affiliation

This project is not affiliated with or endorsed by Stanford University, the CS336 course staff, Datawhale, Sea AI Lab, Alibaba/Qwen or OpenAI.
