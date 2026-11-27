"""Automated readiness, separately for SFT and GRPO. Reads results only.   python scripts/readiness.py sft|grpo"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tier_ids import A  # noqa: E402

VALIDATED = "4a28e40b1372e0de81ab7b1744eb2570e83bdad680a6c12478bdaf268efa8f9c"  # code the GPU validation ran; later changes are listed in EXPERIMENT_PLAN.md section 12
COMMON = ("sync", "policy_memory_and_gradients", "policy_peak_below_46_gib")
GRPO = ("resume", "exact_check_ran_every_step", "no_weight_mismatch_in_any_step", "rollout_logprobs_agree")


def problems(kind, runs):
    out = []
    p = runs + "/validation/summary.json"
    if not os.path.exists(p):
        return ["validation/summary.json is missing"]
    g = json.load(open(p)).get("gates", {})
    out += [f"validation gate {k} has not passed" for k in (COMMON if kind == "sft" else COMMON + GRPO) if g.get(k) is not True]
    for d in ("sft",) if kind == "sft" else ("grpo_ref", "grpo_from_sft"):
        m = runs + f"/validation/{d}/run.json"
        if not os.path.exists(m) or json.load(open(m))["code_sha256"] != VALIDATED:
            out.append(f"validation/{d} did not run the validated code")
    if kind == "grpo":
        sft = [i for i in A if i.startswith("SFT")]
        todo = [i for i in sft if not os.path.exists(f"{runs}/sft/{i[4:]}/DONE.json")]
        if todo:
            out.append(f"Session A is not finished: {todo}")
        elif json.load(open(runs + "/sft/reverified_n512_s0/DONE.json"))["lineage"]["chain"] != ["base", "sft"]:
            out.append("the registered GRPO parent SFT-reverified_n512_s0 has an unexpected lineage")
    return out


if __name__ == "__main__":
    bad = problems(sys.argv[1], os.environ["PT_RUNS"])
    print("\n".join("NOT READY: " + b for b in bad) or f"{sys.argv[1]}: validation gates present and passed")
    sys.exit(1 if bad else 0)
