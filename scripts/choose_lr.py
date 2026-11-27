"""Apply the pre-registered learning-rate rule (EXPERIMENT_PLAN.md section 10).

python scripts/choose_lr.py sft|grpo [--value]      (PT_RUNS must be set; all three tuning runs must be finished)
Highest final-step GSM8K dev accuracy (n = 1,000); among the rates within 1.0 pp (10 answers) of it, the smallest. --value prints only the rate.
"""
import json, os, sys

kind = sys.argv[1]
rates = {"sft": ("1.0e-5", "2.0e-5", "5.0e-5"), "grpo": ("2.0e-6", "5.0e-6", "1.0e-5")}[kind]
n, ids = {}, set()
for r in rates:
    p = f"{os.environ['PT_RUNS']}/{kind}/tune_lr{r.replace('.0', '')}/DONE.json"
    if not os.path.exists(p):
        sys.exit(f"not finished: {p}. The rule needs all three runs.")
    e = json.load(open(p))["eval"]["gsm8k_dev"]
    if e["n"] != 1000:
        sys.exit(f"expected 1,000 dev items, found {e['n']} in {p}")
    n[r] = e["correct"]
    ids.add(e["data_sha256"])
    used = json.load(open(os.path.dirname(p) + "/run.json"))["config"]["train"]["lr"]
    if float(used) != float(r):
        sys.exit(f"{p} was trained at {used}, not {r}")
if len(ids) != 1:
    sys.exit("the tuning runs were evaluated on different files")
best = max(n.values())
lr = min((r for r in rates if best - n[r] <= 10), key=float)
print(lr if "--value" in sys.argv else f"correct of 1000: {n}\nchosen lr: {lr}")
