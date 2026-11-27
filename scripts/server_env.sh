# Source this on the research server, in every shell, AFTER access and GPUs have been confirmed:
#   export A5_AUTHORIZED=yes A5_GPUS=0,1 A5_ROOT=/srv/<your-own-new-folder>
#   source scripts/server_env.sh
# No billing, no shutdown, no watchdog, no sudo, no writes outside A5_ROOT.
_die() { echo "REFUSED: $1"; return 1; }
_a5() {
  [ "${A5_AUTHORIZED:-}" = yes ] || { _die "server use has not been confirmed. Set A5_AUTHORIZED=yes only once the owner has agreed."; return 1; }
  [[ "${A5_GPUS:-}" =~ ^[0-9]+,[0-9]+$ ]] || { _die "set A5_GPUS to the two GPU indices you were allocated, e.g. A5_GPUS=0,1"; return 1; }
  [ "${A5_GPUS%,*}" != "${A5_GPUS#*,}" ] || { _die "A5_GPUS needs two different GPUs"; return 1; }
  [ -n "${A5_ROOT:-}" ] && [ "${A5_ROOT#/}" != "$A5_ROOT" ] || { _die "set A5_ROOT to an absolute path of a NEW folder"; return 1; }
  local real old; real=$(readlink -m "$A5_ROOT")
  for old in "${A5_OLD_PROJECT:-/srv/research/legacy-workdir}" "${A5_OLD_STORAGE:-/srv/research/storage}"; do
    old=$(readlink -m "$old")
    case "$real/" in "$old/"*) _die "A5_ROOT is inside $old, which must stay untouched"; return 1 ;; esac
    case "$old/" in "$real/"*) _die "A5_ROOT contains $old; choose a separate folder"; return 1 ;; esac
  done
  unset GPU_USD_PER_HOUR POD_START_EPOCH RUNPOD_POD_ID RUNPOD_API_KEY A5_PLATFORM_RUNPOD WATCHDOG_MINUTE PILOT_MAX_USD ALLOW_FINAL_EVAL PYTHONPATH PYTHONHOME VIRTUAL_ENV 2>/dev/null
  export A5_ROOT=$real PT_PLATFORM=research_server PT_HW=${PT_HW:-a6000}
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$A5_GPUS
  export PT_RUNS=$A5_ROOT/runs PT_DATA=$A5_ROOT/a5/data/stage0 HF_HOME=$A5_ROOT/hf XDG_CACHE_HOME=$A5_ROOT/cache VLLM_CACHE_ROOT=$A5_ROOT/cache/vllm CUDA_CACHE_PATH=$A5_ROOT/cache/nv TRITON_CACHE_DIR=$A5_ROOT/cache/triton TORCH_HOME=$A5_ROOT/cache/torch
  export UV_CACHE_DIR=$A5_ROOT/cache/uv UV_PYTHON_INSTALL_DIR=$A5_ROOT/python UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TMPDIR=$A5_ROOT/tmp
  export HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_TELEMETRY=1 VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export WANDB_MODE=disabled
  # the pinned base model (revision 4a83ca6e): weights used from any folder must have exactly this hash
  export PT_BASE_SHA256=f7aa8d665ab44d3b461bed742fe93cdcedc15d2e31e50dc372c126e6f720857b
  mkdir -p "$PT_RUNS/pilot_logs" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR" || return 1
  [ -f "$A5_ROOT/a5/server_paths.sh" ] && . "$A5_ROOT/a5/server_paths.sh"
  echo "research server: GPUs $A5_GPUS, root $A5_ROOT, settings $PT_HW; billing and shutdown are off"
}
_a5; _rc=$?; unset -f _a5 _die; return $_rc 2>/dev/null || exit $_rc
