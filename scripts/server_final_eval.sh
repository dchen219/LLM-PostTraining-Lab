#!/usr/bin/env bash
# Held-out evaluation of the checkpoints frozen in final/freeze_v1.yaml (inference only; trains nothing, changes no run folder).
#   source scripts/server_env.sh
#   bash scripts/server_final_eval.sh plan     the frozen job list; starts nothing
#   bash scripts/server_final_eval.sh ready    package, frozen verifier, tests; starts nothing
#   bash scripts/server_final_eval.sh check    every checkpoint's identity (DONE.json, weights sha256); starts nothing
#   bash scripts/server_final_eval.sh run      lock, then evaluate on two GPUs in parallel; needs A5_FINAL_EVAL_APPROVED=yes
#   bash scripts/server_final_eval.sh status
# Outputs go only to $PT_RUNS/final_eval/<id>/. A finished evaluation is never repeated; an interrupted one restarts from scratch on the same
# checkpoint (greedy decoding) and is recorded in its resumes.jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "${PT_PLATFORM:-}" = research_server ] && [ "${PT_HW:-}" = a6000 ] || { echo "REFUSED: source scripts/server_env.sh first"; exit 1; }
[ "${A5_AUTHORIZED:-}" = yes ] || { echo "REFUSED: server use not authorized"; exit 1; }
ready() {
  python3 scripts/verify_upload.py || { echo "REFUSED: packaged files are missing or changed"; exit 1; }
  python -m posttrain.verify || { echo "REFUSED: the verifier is not the frozen version"; exit 1; }
  python -m pytest tests -q -p no:cacheprovider -W ignore -k "final or verifier or lock or eval" || { echo "REFUSED: final-evaluation tests"; exit 1; }
  echo "READY (final)"
}
case "${1:-}" in
  plan) python -m posttrain.final_eval --plan --workers 2 ;;
  ready) ready ;;
  check) python -m posttrain.final_eval --check ;;
  status) for d in "$PT_RUNS"/final_eval/*/; do [ -d "$d" ] || continue; x=started; [ -f "$d/DONE.json" ] && x=evaluated; echo "$x  $(basename "$d")"; done
          [ -f "$PT_RUNS/final_eval/FINAL_CHECKPOINTS.json" ] && echo "lock: present" || echo "lock: none" ;;
  run)
    [ "${A5_FINAL_EVAL_APPROVED:-}" = yes ] || { echo "REFUSED: final evaluation has not been approved (A5_FINAL_EVAL_APPROVED is not 'yes')"; exit 1; }
    ready
    bash scripts/server_preflight.sh
    python -m posttrain.final_eval --lock
    IFS=, read -r -a G <<< "${A5_GPUS:?set A5_GPUS, e.g. 0,1}"
    [ "${#G[@]}" -ge 2 ] || G=("${G[0]}" "${G[0]}")
    L=$PT_RUNS/logs; mkdir -p "$L"; t=$(date +%s); pids=()
    if [ "${G[0]}" = "${G[1]}" ]; then
      CUDA_VISIBLE_DEVICES=${G[0]} python -m posttrain.final_eval --run --worker 0 --workers 1 > "$L/final_eval_w0.$t.log" 2>&1 || { tail -n 25 "$L/final_eval_w0.$t.log" || true; echo "STOPPED (final evaluation). Send the log."; exit 1; }
    else
      for w in 0 1; do CUDA_VISIBLE_DEVICES=${G[$w]} python -m posttrain.final_eval --run --worker $w --workers 2 > "$L/final_eval_w$w.$t.log" 2>&1 & pids+=($!); done
      rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
      [ $rc = 0 ] || { tail -n 15 "$L"/final_eval_w*."$t".log || true; echo "STOPPED (final evaluation). Send the logs."; exit 1; }
    fi
    echo "FINAL EVALUATION DONE. Nothing else was started." ;;
  *) echo "plan | ready | check | run | status"; exit 2 ;;
esac
