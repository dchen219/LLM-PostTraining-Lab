"""Gate: kill a GRPO run right after its first saved state, resume it, and compare with an uninterrupted run (same seed, same config).

python scripts/grpo_resume_test.py --ref <uninterrupted run dir> --run <new run dir> [--set key=value ...]   (the --set values of the reference)
Only its own child process is ever killed. Sampling is not bitwise reproducible, so the gate is structural: every step present once, the prompts
after resume are the uninterrupted run's prompts (sampler state restored), no weight mismatch, finished. Metric differences are reported, not gated.
"""
import argparse, json, os, signal, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/pilot/pilot_grpo_base.yaml")
ap.add_argument("--ref", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--timeout", type=int, default=1800)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
assert os.path.exists(a.ref + "/DONE.json"), "the reference run has not finished"
assert not os.path.exists(a.run) and not os.path.exists(a.out), "refusing to overwrite"
cmd = [sys.executable, "-m", "posttrain.train_grpo", "--config", os.path.join(ROOT, a.config)]
sets = ["--set", *a.set, f"run_dir={a.run}"]
p = subprocess.Popen(cmd + sets, cwd=ROOT)
t = time.time()
while not os.path.exists(a.run + "/state.pt") and p.poll() is None and time.time() - t < a.timeout:
    time.sleep(0.05)
if p.poll() is not None or not os.path.exists(a.run + "/state.pt"):
    p.kill()
    sys.exit("the run ended, or timed out, before a state was saved; nothing to resume")
p.send_signal(signal.SIGKILL)
p.wait()
time.sleep(10 if "cpu" not in " ".join(a.set) else 0)  # let the driver release GPU memory
subprocess.run(cmd + ["--resume"] + sets, cwd=ROOT, check=True)


def train(d):
    return {r["step"]: r for r in map(json.loads, open(d + "/metrics.jsonl")) if r["event"] == "train"}  # a repeated step: the later row wins


def prompts(d, s):
    return [json.loads(x)["id"] for x in open(f"{d}/rollouts/step{s}.jsonl")]


A, B = train(a.ref), train(a.run)
saved = [s for s in sorted(A) if os.path.exists(f"{a.ref}/rollouts/step{s}.jsonl") and os.path.exists(f"{a.run}/rollouts/step{s}.jsonl")]
last = [s for s in sorted(B) if B[s].get("params_checked")]
rep = dict(steps_ref=sorted(A), steps_run=sorted(B), resumed=os.path.exists(a.run + "/resumes.jsonl"), done=os.path.exists(a.run + "/DONE.json"),
           same_prompts={s: prompts(a.ref, s) == prompts(a.run, s) for s in saved}, params_checked_steps=last,
           params_mismatched=sum(B[s].get("params_mismatched", 0) for s in B),
           reported_only={s: {k: (A[s].get(k), B[s].get(k)) for k in ("loss", "grad_norm", "reward_mean", "informative_rate")} for s in A if s in B})
uses_vllm = "engine.name=hf" not in a.set
rep["prompts_distinguishable"] = any(prompts(a.ref, s) != prompts(a.ref, saved[0]) for s in saved[1:])  # otherwise equal prompts would prove nothing
rep["pass"] = bool(rep["resumed"] and rep["done"] and rep["steps_run"] == rep["steps_ref"] and saved == sorted(A) and all(rep["same_prompts"].values())
                   and rep["prompts_distinguishable"] and rep["params_mismatched"] == 0 and (last == sorted(B) or not uses_vllm))
json.dump(rep, open(a.out, "w"), indent=1)
print(json.dumps(rep, indent=1))
sys.exit(0 if rep["pass"] else 1)
