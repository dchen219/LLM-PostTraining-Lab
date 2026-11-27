"""Runs vLLM 0.7.2's REAL Qwen2ForCausalLM.load_weights on CPU (no engine, no KV cache, no GPU) and checks every parameter exactly.

Sequence, as in training: load BASE weights -> overwrite in place with the GRPO step-10 weights (as fp32 tensors with a sub-bf16 perturbation,
like fp32 master weights) -> compare every engine parameter with policy.to(bf16), zero tolerance.
Positive controls: before the second load the check must FAIL; after skipping one tensor it must report exactly that parameter.
Not covered: the GPU copy of the weights, attention kernels, CUDA graphs.
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from safetensors import safe_open
from posttrain.common import ROOT, jdump
from posttrain.engines import compare_engine_state, expected_engine_state

BASE = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2")
NEW = os.path.expanduser("~/a5_pilot/pilot/pilot_grpo_base/checkpoint")


def build():
    """Construct vLLM's model class directly on CPU; only the attention-backend lookup (which probes a GPU) is replaced."""
    from unittest.mock import patch
    from vllm.config import CacheConfig, DeviceConfig, LoadConfig, ModelConfig, ParallelConfig, VllmConfig
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
    from vllm.platforms import current_platform
    init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29777", local_rank=0, backend="gloo")
    initialize_model_parallel(1, 1, backend="gloo")
    mc = ModelConfig(model=BASE, task="generate", tokenizer=BASE, tokenizer_mode="auto", trust_remote_code=False, dtype="bfloat16", seed=0, max_model_len=2048)
    cfg = VllmConfig(model_config=mc, cache_config=CacheConfig(16, 0.1, 0, "auto"), parallel_config=ParallelConfig(), device_config=DeviceConfig("cpu"),
                     load_config=LoadConfig(), **EXTRA)
    torch.set_default_dtype(torch.bfloat16)
    try:
        with patch.object(type(current_platform), "get_attn_backend_cls", return_value="vllm.attention.backends.xformers.XFormersBackend"):
            return Qwen2ForCausalLM(vllm_config=cfg)
    finally:
        torch.set_default_dtype(torch.float32)


EXTRA = {}


def stream(path, noise=0.0, skip=None):
    """Policy state_dict as train_grpo passes it: fp32 tensors, including the tied lm_head.weight duplicate."""
    g = torch.Generator().manual_seed(0)
    with safe_open(path + "/model.safetensors", "pt") as f:
        for k in f.keys():
            if k == skip:
                continue
            v = f.get_tensor(k).float()
            if noise:  # below half a bf16 ulp in relative terms, so rounding gives the stored value back
                v = v * (1 + noise * (2 * torch.rand(v.shape, generator=g) - 1))
            yield k, v
            if k == "model.embed_tokens.weight":
                yield "lm_head.weight", v


def check(m, path, noise=0.0):
    tot = dict(params_checked=0, params_mismatched=0, mismatch_names=[], elements=0)
    actual, part = dict(m.named_parameters()), {}
    tot["engine_parameters"] = len(actual)

    covered = set()

    def flush():
        e = expected_engine_state(part)
        covered.update(e)
        r = compare_engine_state(e, {k: actual[k] for k in e if k in actual})  # group by group to save memory; coverage is checked at the end
        tot["params_checked"] += r["params_checked"]
        tot["params_mismatched"] += r["params_mismatched"]
        tot["mismatch_names"] = (tot["mismatch_names"] + r["mismatch_names"])[:5]
        tot["elements"] += sum(x.numel() for x in e.values())
    groups = {}
    for k, v in stream(path, noise):
        key = ".".join(k.split(".")[:3]) if ".layers." in k else ("emb" if k in ("model.embed_tokens.weight", "lm_head.weight") else k)
        groups.setdefault(key, {})[k] = v.to(torch.bfloat16)
    for part in groups.values():
        flush()
    tot["engine_parameters_never_compared"] = sorted(set(actual) - covered)
    tot["params_mismatched"] += len(set(actual) - covered)
    tot["dtypes"] = sorted({str(v.dtype) for v in actual.values()})
    return tot


m = build()
rep = dict(vllm="0.7.2", model_class=type(m).__module__ + "." + type(m).__name__, dtype=str(next(m.parameters()).dtype))
m.load_weights(stream(BASE))
rep["1_base_loaded_vs_base"] = check(m, BASE)
rep["2_positive_control_base_loaded_vs_grpo10"] = check(m, NEW)
m.load_weights(stream(NEW, noise=1e-4))
rep["3_in_place_update_vs_grpo10"] = check(m, NEW, noise=1e-4)
m.load_weights(stream(BASE))
m.load_weights(stream(NEW, skip="model.layers.13.self_attn.k_proj.weight"))
rep["4_positive_control_one_tensor_skipped"] = check(m, NEW)
rep["pass"] = (rep["1_base_loaded_vs_base"]["params_mismatched"] == 0 and rep["2_positive_control_base_loaded_vs_grpo10"]["params_mismatched"] > 0
               and rep["3_in_place_update_vs_grpo10"]["params_mismatched"] == 0 and rep["3_in_place_update_vs_grpo10"]["params_checked"] == rep["3_in_place_update_vs_grpo10"]["engine_parameters"]
               and rep["4_positive_control_one_tensor_skipped"]["mismatch_names"] == ["model.layers.13.self_attn.qkv_proj.weight"])
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "pilot_analysis", "vllm_loader_check.json")
assert not os.path.exists(out), "refusing to overwrite"
jdump(out, rep)
print(json.dumps(rep, indent=1))
