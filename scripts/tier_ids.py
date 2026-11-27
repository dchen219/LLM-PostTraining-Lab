"""Which registered experiments belong to which plan tier. Single source for make_configs.py (status) and plan_tiers.py (cost)."""
X = "SFT-"
A = [X + f"tune_lr{l}" for l in ("1e-5", "2e-5", "5e-5")] + [X + n + "_s0" for n in (
    "exact_full", "reverified_full", "reverified_sizecontrol", "exact_n512", "reverified_n512", "reverified_n128", "reverified_n2048")]
A += ["GRPO-tune_lr" + l for l in ("2e-6", "5e-6", "1e-5")] + ["GRPO-A_base_s2026", "GRPO-B_sft512_s2026"]
B = [X + "reverified_n256_s0", X + "reverified_n1024_s0"]  # middle points of the sample-efficiency curve; the pre-registered comparisons do not use them
B += [X + f"{n}_s{i}" for i in (1, 2) for n in ("exact_full", "reverified_full", "reverified_sizecontrol", "exact_n512", "reverified_n512",
                                                "reverified_n128", "reverified_n256", "reverified_n1024", "reverified_n2048")]
B += ["GRPO-B_sft512_s2027", "GRPO-B_sft512_s2028", "GRPO-A_base_s2027", "GRPO-A_base_s2028", "GRPO-X_filter_compute"]
C = [X + f"{n}_s{i}" for i in (0, 1, 2) for n in ("raw_n512", "exact_n128", "exact_n2048")] + ["EI-base"]
C += ["GRPO-X_" + n for n in ("filter_renorm", "difficulty", "offpolicy_e2", "onpolicy_u4", "lennorm_mean", "stdnorm")]
assert len(set(A + B + C)) == len(A + B + C)
