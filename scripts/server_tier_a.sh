#!/usr/bin/env bash
# Tier A (15 runs) in four sessions. Each session stops at the first failure and starts nothing else. Run inside tmux.
#   source scripts/server_env.sh
#   bash scripts/server_tier_a.sh plan                 print the runs, start nothing
#   bash scripts/server_tier_a.sh results              dev results so far, start nothing
#   bash scripts/server_tier_a.sh ready-sft | ready-grpo      automated readiness only, start nothing
# Session A:
#   bash scripts/server_tier_a.sh sft-tune
#   bash scripts/server_tier_a.sh sft-main  auto       auto = the pre-registered rule (EXPERIMENT_PLAN.md section 10); any other value must equal it
# Session B (refused until all ten SFT runs are finished):
#   bash scripts/server_tier_a.sh grpo-tune
#   bash scripts/server_tier_a.sh grpo-main auto
# A finished run is skipped; a run that stopped part-way is resumed from its last saved state.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "${PT_PLATFORM:-}" = research_server ] && [ "${PT_HW:-}" = a6000 ] || { echo "REFUSED: source scripts/server_env.sh first"; exit 1; }
ids() { case "$1" in
  sft-tune)  echo SFT-tune_lr1e-5 SFT-tune_lr2e-5 SFT-tune_lr5e-5 ;;
  sft-main)  echo SFT-reverified_n512_s0 SFT-exact_n512_s0 SFT-reverified_full_s0 SFT-exact_full_s0 SFT-reverified_sizecontrol_s0 SFT-reverified_n128_s0 SFT-reverified_n2048_s0 ;;
  grpo-tune) echo GRPO-tune_lr2e-6 GRPO-tune_lr5e-6 GRPO-tune_lr1e-5 ;;
  grpo-main) echo GRPO-B_sft512_s2026 GRPO-A_base_s2026 ;;
  *) return 1 ;; esac; }
dir() { case "$1" in SFT-*) echo "$PT_RUNS/sft/${1#SFT-}" ;; GRPO-*) echo "$PT_RUNS/grpo/${1#GRPO-}" ;; esac; }
# Automated readiness. SFT: server validation, code unchanged, frozen verifier (dev scores choose the learning rate). No reward gate.
# GRPO adds what its reward and rollouts need: verifier regression tests, the GRPO validation gates, and a finished Session A.
ready() {
  python3 scripts/verify_upload.py || { echo "REFUSED: packaged files are missing or changed"; exit 1; }
  python -m posttrain.verify || { echo "REFUSED: the verifier is not the frozen version or a golden case fails"; exit 1; }
  python scripts/readiness.py "$1" || { echo "REFUSED: $1 readiness"; exit 1; }
  [ "$1" = sft ] || python -m pytest tests -q -p no:cacheprovider -W ignore -k "verifier or grpo or sync or rollout" || { echo "REFUSED: GRPO reward/sync regression tests"; exit 1; }
  echo "READY ($1)"
}
S=${1:-}
[[ "$S" != ready-* ]] || S2=${S#ready-}
case "$S" in
  plan) for s in sft-tune sft-main grpo-tune grpo-main; do echo "$s:"; for i in $(ids $s); do [ -f "$(dir "$i")/DONE.json" ] && x=done || x=todo; [ -f "recovery/incident_$i.json" ] && x=FAILED; echo "   $x  $i"; done; done; exit 0 ;;
  results) python - <<'PYEOF'
import glob, json, os
R = os.environ["PT_RUNS"]
for f in sorted(glob.glob(R + "/sft/*/DONE.json") + glob.glob(R + "/grpo/*/DONE.json")):
    e = json.load(open(f)).get("eval") or {}
    print(os.path.relpath(os.path.dirname(f), R).ljust(34), " ".join(f"{k} {v['accuracy']:.3f} (format {v['format_rate']:.3f}, n {v['n']})" for k, v in e.items()))
PYEOF
    exit 0 ;;
  sft-tune|grpo-tune) [ $# -eq 1 ] || { echo "no learning rate here: these runs choose it"; exit 2; }; LR= ;;
  sft-main|grpo-main) [ "${2:-}" = auto ] || [[ "${2:-}" =~ ^[0-9.]+e-[0-9]+$ ]] || { echo "usage: $S auto"; exit 2; }; LR=$2 ;;
  ready-sft|ready-grpo) LR= ;;
  *) echo "plan | results | ready-sft | ready-grpo | sft-tune | sft-main auto | grpo-tune | grpo-main auto"; exit 2 ;;
esac
[ "${A5_AUTHORIZED:-}" = yes ] || { echo "REFUSED: server use not authorized"; exit 1; }
case "$S" in ready-*) ;; *) [ "${A5_TRAIN_APPROVED:-}" = yes ] || { echo "REFUSED: training has not been approved (A5_TRAIN_APPROVED is not 'yes')"; exit 1; } ;; esac
K=${S2:-${S%%-*}}; ready "$K"
case "$S" in ready-*) exit 0 ;; esac
if [ -n "$LR" ]; then
  W=$(python scripts/choose_lr.py "${S%%-*}" --value) || { echo "REFUSED: $W"; exit 1; }
  [ "$LR" = auto ] || python -c 'import sys; sys.exit(float(sys.argv[1]) != float(sys.argv[2]))' "$LR" "$W" || { echo "REFUSED: $LR is not the rate the rule selects ($W)"; exit 1; }
  LR=$W; echo "learning rate by the pre-registered rule: $LR"
fi
L=$PT_RUNS/logs; mkdir -p "$L"
for i in $(ids "$S"); do
  d=$(dir "$i"); [ -f "$d/DONE.json" ] && { echo "skip $i (done)"; continue; }
  [ -f "recovery/incident_$i.json" ] && { echo "skip $i (FAILED, kept as evidence: recovery/incident_$i.json; never resumed)"; continue; }
  bash scripts/server_preflight.sh
  a=(--id "$i" --go); [ -f "$d/state.pt" ] && a+=(--resume)
  [ ! -d "$d" ] || [ -f "$d/state.pt" ] || { echo "STOPPED: $d exists without a saved state; look at it before deciding"; exit 1; }
  [ -z "$LR" ] || a+=(--set "train.lr=$LR")
  echo "=== $i $(date +%F_%T)"; python scripts/run_experiment.py "${a[@]}" > "$L/$i.$(date +%s).log" 2>&1 || { tail -25 "$(ls -t "$L/$i".*.log | head -1)"; echo "STOPPED at $i"; exit 1; }
done
echo "SESSION $S DONE. Nothing else was started. Review: bash scripts/server_tier_a.sh results"
