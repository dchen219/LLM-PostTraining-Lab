#!/usr/bin/env bash
# One-time setup on the research server, inside A5_ROOT only. New environment; the old one is not touched. Starts no training.
#   source scripts/server_env.sh && bash setup_server.sh
# A5_BASE_MODEL (optional): an existing local snapshot of Qwen2.5-Math-1.5B to read instead of downloading. Its hash must match the pin.
set -euo pipefail
cd "$(dirname "$0")"
[ "${PT_PLATFORM:-}" = research_server ] || { echo "source scripts/server_env.sh first"; exit 1; }
[ "$(readlink -f .)" = "$A5_ROOT/a5" ] || { echo "unpack the package to $A5_ROOT/a5 and run this there"; exit 1; }
python3 scripts/verify_upload.py
if find . -maxdepth 3 \( -name '*id_rsa*' -o -name '*.ppk' -o -name '*key_pwd*' -o -name '.env' \) -not -path './.venv*' | grep -q .; then echo "credential-like files in the package; remove them"; exit 1; fi
# A5_UV may point at an existing uv program; running it changes nothing there (its cache and Python installs go under A5_ROOT).
UV=${A5_UV:-$(command -v uv || true)}; [ -x "$UV" ] || { echo "uv not found: set A5_UV to an existing uv program"; exit 1; }
[ -d .venv-server ] || "$UV" venv --no-config --python-preference only-managed --python 3.12 .venv-server
"$UV" pip install --no-config --python .venv-server/bin/python -r requirements-cloud.lock.txt
. .venv-server/bin/activate
python - <<'PYEOF'
import torch, transformers, vllm
assert torch.__version__.startswith("2.5.1") and vllm.__version__ == "0.7.2" and transformers.__version__ == "4.51.3", (torch.__version__, vllm.__version__, transformers.__version__)
assert torch.cuda.device_count() == 2, "expected exactly the two allocated GPUs"
assert torch.cuda.is_bf16_supported()
print("torch", torch.__version__, "cuda", torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(2)])
PYEOF
if [ -n "${A5_BASE_MODEL:-}" ]; then M=$(readlink -f "$A5_BASE_MODEL")
else M=$(python -c 'from huggingface_hub import snapshot_download as s; print(s("Qwen/Qwen2.5-Math-1.5B", revision="4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"))'); fi
echo "$PT_BASE_SHA256  $M/model.safetensors" | sha256sum -c -
{ echo "export PT_MODEL=\"$M\" HF_HUB_OFFLINE=1"; echo ". \"$A5_ROOT/a5/.venv-server/bin/activate\""; } > server_paths.sh
. ./server_paths.sh
python -m pytest tests -q -p no:cacheprovider -W ignore
echo "SERVER SETUP OK (no training was started)"
