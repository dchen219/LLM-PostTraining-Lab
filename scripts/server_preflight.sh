#!/usr/bin/env bash
# Is it polite and safe to start on the allocated GPUs? Looks only. Run before every job; server_validation.sh runs it too.
set -euo pipefail
: "${A5_GPUS:?source scripts/server_env.sh first}" "${A5_ROOT:?}" "${PT_RUNS:?}"
[ "${PT_PLATFORM:-}" = research_server ] || { echo "FAIL: not configured for the research server"; exit 1; }
Q() { nvidia-smi -i "$1" --query-gpu="$2" --format=csv,noheader,nounits; }
for g in ${A5_GPUS/,/ }; do
  name=$(Q "$g" name); tot=$(Q "$g" memory.total); used=$(Q "$g" memory.used)
  echo "GPU $g: $name, $used / $tot MiB used"
  [[ "$name" == *A6000* ]] || { echo "FAIL: GPU $g is not an RTX A6000; the memory settings are for 48 GB"; exit 1; }
  [ "$tot" -ge 45000 ] || { echo "FAIL: GPU $g reports only $tot MiB"; exit 1; }
  [ "$used" -le 600 ] || { echo "FAIL: GPU $g is in use ($used MiB). Someone else may be running; do not start"; exit 1; }
  p=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | head -3 | tr '\n' ' ')
  [ -z "${p// }" ] || { echo "FAIL: GPU $g has running processes ($p)"; exit 1; }
done
for c in squeue qstat condor_q; do if command -v "$c" >/dev/null; then
  [ -n "${SLURM_JOB_ID:-}${PBS_JOBID:-}${_CONDOR_JOB_AD:-}" ] || { echo "FAIL: this server has a scheduler ($c). Request the GPUs through it"; exit 1; }; fi; done
D=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$D" -ge 550 ] || { echo "FAIL: driver $D is older than CUDA 12.4 needs"; exit 1; }
F=$(df -BG --output=avail "$A5_ROOT" | tail -1 | tr -dc 0-9)
[ "$F" -ge "${PT_MIN_FREE_GB:-150}" ] || { echo "FAIL: only ${F} GB free"; exit 1; }
R=$(free -g | awk '/Mem:/{print $7}'); [ "$R" -ge 40 ] || { echo "FAIL: only ${R} GB of memory available (state saves need about 20)"; exit 1; }
echo "PREFLIGHT_OK (driver $D, ${F} GB disk, ${R} GB memory)"
