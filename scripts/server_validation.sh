#!/usr/bin/env bash
# The validation run (V1-V5 of FULL_TRAINING_DECISION.md) on two allocated RTX A6000s. Stops at the first failure. Starts no full training.
#   source scripts/server_env.sh && bash scripts/server_validation.sh          (inside tmux)
# It needs A5_AUTHORIZED=yes (checked in server_env.sh), idle GPUs, and a fresh output folder. It kills only processes it started itself.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "${PT_PLATFORM:-}" = research_server ] && [ "${A5_AUTHORIZED:-}" = yes ] || { echo "REFUSED: source scripts/server_env.sh with A5_AUTHORIZED=yes"; exit 1; }
[ "${PT_HW:-}" = a6000 ] || { echo "REFUSED: PT_HW must be a6000"; exit 1; }
V=$PT_RUNS/validation; L=$V/logs; [ ! -e "$V" ] || { echo "REFUSED: $V exists; results are never overwritten"; exit 1; }
MAX_MIN=${VALIDATION_MAX_MINUTES:-180}; T0=$(date +%s)
bash scripts/server_preflight.sh; mkdir -p "$L"
nvidia-smi -i "$A5_GPUS" --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,nounits -l 5 > "$L/gpu.csv" 2>/dev/null & MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT
step() { local n=$1; shift; local s=$(( MAX_MIN * 60 - $(date +%s) + T0 )); [ "$s" -gt 60 ] || { echo "time limit reached before $n"; exit 3; }
  echo "=== $n"; local rc=0 t; t=$(date +%s); timeout --signal=INT --kill-after=30 "$s" "$@" > "$L/$n.log" 2>&1 || rc=$?
  echo "{\"step\":\"$n\",\"rc\":$rc,\"seconds\":$(( $(date +%s) - t ))}" | tee -a "$L/steps.jsonl"
  [ "$rc" -eq 0 ] || { tail -25 "$L/$n.log"; echo "STOPPED at $n"; exit "$rc"; }; }
G=(eval.sets={} train.micro_batch=2 grpo.check_rollout_logprobs=true grpo.sync_check_every=1 grpo.save_rollouts_every=1)
PG=(python -m posttrain.train_grpo --config configs/pilot/pilot_grpo_base.yaml --set)
step V1_sync          python scripts/gpu_validation.py --part sync --model "$PT_MODEL" --out "$V/sync.json"
step V3_memory        python scripts/gpu_validation.py --part policy --model "$PT_MODEL" --out "$V/policy.json"
step V2a_grpo_ref     "${PG[@]}" "${G[@]}" train.token_budget=3300 grpo.steps=4 train.save_every=2 "run_dir=$V/grpo_ref"
step V2b_grpo_resume  python scripts/grpo_resume_test.py --ref "$V/grpo_ref" --run "$V/grpo_resume" --out "$V/grpo_resume.json" --set "${G[@]}" train.token_budget=3300 grpo.steps=4 train.save_every=2
step V4_sft_mb2       python -m posttrain.train_sft --config configs/sft/reverified_full_s0.yaml --set eval=null train.micro_batch=2 train.max_steps=10 train.save_every=null "run_dir=$V/sft"
step V5_from_sft      "${PG[@]}" "${G[@]}" train.token_budget=3300 grpo.steps=3 "init.path=$V/sft/checkpoint" init.expect_stage=sft "run_dir=$V/grpo_from_sft"
python - "$V" <<'PYEOF'
import glob, json, sys
V = sys.argv[1]; out = {}
for d in ("grpo_ref", "grpo_resume", "sft", "grpo_from_sft"):
    m = [json.loads(x) for x in open(f"{V}/{d}/metrics.jsonl")]
    t = [r for r in m if r["event"] == "train"]
    out[d] = dict(steps=len(t), params_checked=[r.get("params_checked") for r in t], rollout_tokens=[r.get("rollout_logprob_tokens") for r in t], step_s=[round(r["step_s"], 1) for r in t], peak={k: max(r.get(k, 0) for r in t) for k in t[0] if "peak" in k},
                  params_mismatched=sum(r.get("params_mismatched", 0) for r in t), probe=[r.get("sync_mean_abs_logprob_diff") for r in t], rollout_gap=[r.get("rollout_logprob_mad") for r in t], informative=[r.get("informative_rate") for r in t])
J = lambda p: json.load(open(f"{V}/{p}"))
G = [d for d in out if d.startswith("grpo")]
gates = dict(sync=J("sync.json")["pass"], policy_memory_and_gradients=J("policy.json")["pass"], resume=J("grpo_resume.json")["pass"],
             exact_check_ran_every_step=all(x and x > 0 for d in G for x in out[d]["params_checked"]),
             no_weight_mismatch_in_any_step=all(out[d]["params_mismatched"] == 0 for d in G),
             rollout_logprobs_agree=all(x is not None and x <= 0.05 for d in G for x in out[d]["rollout_gap"]),
             policy_peak_below_46_gib=all(out[d]["peak"].get("gpu1_peak_gib", 99) < 46 for d in out))
out["gates"], out["pass"] = gates, all(gates.values())
json.dump(out, open(f"{V}/summary.json", "w"), indent=1); print(json.dumps(out, indent=1))
sys.exit(0 if out["pass"] else 1)
PYEOF
echo "VALIDATION PASSED (review $V/summary.json before any training; this does not authorize training)"
