#!/usr/bin/env bash
# Post-hoc recovery branches of the failed GRPO-A_base_s2026, from its verified step-80 state, in NEW folders under $PT_RUNS/recovery/.
# Diagnostic only: they are not registered experiments and do not replace the failed run, which stays untouched.
#   source scripts/server_env.sh
#   bash scripts/server_recovery.sh ready      tests, frozen verifier, package; starts nothing
#   bash scripts/server_recovery.sh check      verify the source state (sha256, step) and print every configuration difference; starts nothing
#   bash scripts/server_recovery.sh run        A1 (replay, lr 1e-5) then A2 (lr 5e-6); needs A5_RECOVERY_APPROVED=yes
#   bash scripts/server_recovery.sh status
# A finished branch, or one stopped by a correctness gate (FAILED.json: a result, not an error), is never run again.
# An interrupted branch resumes from its own last state (every 5 steps), or restarts from the source if it had not saved yet.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "${PT_PLATFORM:-}" = research_server ] && [ "${PT_HW:-}" = a6000 ] || { echo "REFUSED: source scripts/server_env.sh first"; exit 1; }
[ "${A5_AUTHORIZED:-}" = yes ] || { echo "REFUSED: server use not authorized"; exit 1; }
SPECS="recovery/A_replay80_lr1e-5_s2026.yaml recovery/A_recovery80_lr5e-6_s2026.yaml"
rdir() { python -c 'import sys, yaml; from posttrain.common import expand; print(expand(yaml.safe_load(open(sys.argv[1])))["run_dir"])' "$1"; }
ready() {
  python3 scripts/verify_upload.py || { echo "REFUSED: packaged files are missing or changed"; exit 1; }
  python -m posttrain.verify || { echo "REFUSED: the verifier is not the frozen version"; exit 1; }
  python -m pytest tests -q -p no:cacheprovider -W ignore -k "fork or recovery or rewind or incident or adam or rollout or grpo or sync or verifier or criteria or nonfinite" \
    || { echo "REFUSED: recovery/GRPO regression tests"; exit 1; }
  echo "READY (recovery)"
}
case "${1:-}" in
  ready) ready ;;
  check) for s in $SPECS; do python -m posttrain.fork_grpo --spec "$s" --check || { echo "REFUSED: $s"; exit 1; }; done; echo "CHECK_OK" ;;
  status) for s in $SPECS; do d=$(rdir "$s"); x=todo; [ -d "$d" ] && x=started; [ -f "$d/FAILED.json" ] && x="stopped at a gate"; [ -f "$d/DONE.json" ] && x=finished
          echo "$x  $s"; done
          [ -f "$PT_RUNS/grpo/A_base_s2026/DONE.json" ] && echo "WARNING: the failed original has a DONE.json" || echo "original GRPO-A_base_s2026: no DONE.json (failed, preserved)" ;;
  run)
    [ "${A5_RECOVERY_APPROVED:-}" = yes ] || { echo "REFUSED: recovery branches have not been approved (A5_RECOVERY_APPROVED is not 'yes')"; exit 1; }
    ready
    L=$PT_RUNS/logs; mkdir -p "$L"
    for s in $SPECS; do
      d=$(rdir "$s")
      [ -f "$d/DONE.json" ] && { echo "skip $s (finished)"; continue; }
      if [ -f "$d/FAILED.json" ]; then
        grep -q "rollout log-prob check failed" "$d/FAILED.json" || { echo "STOPPED: $s stopped at a non-rollout gate earlier (see $d/FAILED.json). Send the log."; exit 1; }
        echo "skip $s (stopped at the rollout gate; that is its result)"; continue; fi
      bash scripts/server_preflight.sh
      a=(--spec "$s"); [ -d "$d" ] && a+=(--resume)
      n=$(basename "$s" .yaml); echo "=== $n $(date +%F_%T)"
      if ! python -m posttrain.fork_grpo "${a[@]}" > "$L/recovery_$n.$(date +%s).log" 2>&1; then
        tail -25 "$(ls -t "$L/recovery_$n".*.log | head -1)"
        if [ -f "$d/FAILED.json" ] && grep -q "rollout log-prob check failed" "$d/FAILED.json"; then
          echo "$n stopped at the rollout log-prob gate; failed batch saved in $d. This is a result; continuing with the next branch."; continue; fi
        [ -f "$d/FAILED.json" ] && { echo "STOPPED at $n: $(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["error"])' "$d/FAILED.json"). Infrastructure, not a result. Send the log."; exit 1; }
        echo "STOPPED at $n (not a gate: interrupted, out of memory or another error). Send the log before retrying."; exit 1
      fi
    done
    echo "RECOVERY BRANCHES DONE. Nothing else was started." ;;
  *) echo "ready | check | status | run"; exit 2 ;;
esac
