"""Memory-only settings per GPU type. They change how a batch is split, never the batch, the loss divisor, the number of updates or the lr.

The experiment configs keep micro_batch 4, the setting validated on 80 GB in the pilot. scripts/run_experiment.py applies these as --set
overrides when PT_HW is set, so each run's run.json records what was used.

a6000 (48 GB): predicted peak = 25.4 GiB + 5.10 GiB per 1,000 padded tokens (results/pilot_analysis/memory_analysis.json, measured on A100).
GRPO and EI: at most 2 sequences and 3,300 padded tokens per micro-batch. Without the token cap, two 2,048-token sequences need about 46 GiB.
SFT: the longest training example is 1,267 tokens, so 2 x 1,267 needs about 38 GiB and no cap is required.
"""
HW = {"a6000": {"posttrain.train_sft": ["train.micro_batch=2"],
                "posttrain.train_grpo": ["train.micro_batch=2", "train.token_budget=3300"],
                "posttrain.train_ei": ["train.micro_batch=2", "train.token_budget=3300"],
                "posttrain.evaluate": []}}


def hw_overrides(module, name):
    if not name:
        return []
    if name not in HW:
        raise SystemExit(f"unknown PT_HW {name!r}; known: {sorted(HW)}")
    return list(HW[name][module])
