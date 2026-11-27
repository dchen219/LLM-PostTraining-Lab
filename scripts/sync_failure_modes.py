"""Would the pilot's sync diagnostics have caught a real weight-sync failure? Local, inference only, no vLLM.

Policy side  : GRPO step-10 weights, fp32 parameters under bf16 autocast (as in training).
Engine side  : the same weights in bf16 (correct sync), or with part of them still at the BASE values (failed sync, 10 steps stale).
Metrics      : mean |dlogprob| on the pilot's 4 probe prompts (tolerance 0.05) and on step-10 rollout tokens (gate 0.05).
Also         : eager vs sdpa in fp32, to tell a precision effect from an implementation difference.
Linear-interpolated 'one step stale' weights (9/10 of the way) are an approximation: intermediate checkpoints do not exist.
Caveat       : HF bf16 sdpa stands in for vLLM flash attention.
"""
import gc, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from posttrain.common import ROOT, jdump, jload
from posttrain.evaluate import template
from posttrain.model_io import load_tokenizer
from transformers import AutoModelForCausalLM

P = os.path.expanduser("~/a5_pilot")
BASE = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2")
NEW = P + "/pilot/pilot_grpo_base/checkpoint"
tok, tpl, dev = load_tokenizer(BASE), template("r1_zero"), "cuda"
pool = jload(os.path.join(ROOT, "data", "stage0", "rl_train_pool.jsonl"))
Q = {r["id"]: r["question"] for r in pool}
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
probe = [enc(tpl.format(question=r["question"].strip())) for r in pool[:4]]
roll, st = [], []
for r in [r for r in jload(f"{P}/pilot/pilot_grpo_base/rollouts/step10.jsonl") if 20 < r["n_tokens"] <= 600][::6][:40]:
    p = enc(tpl.format(question=Q[r["id"]].strip()))
    roll.append(p + enc(r["response"]))
    st.append(len(p) - 1)
SETS = {"probe": (probe, [0] * 4), "rollout_step10": (roll, st)}


def load(path, dtype, attn="sdpa"):
    return AutoModelForCausalLM.from_pretrained(path, attn_implementation=attn, dtype=dtype).to(dev).eval()


@torch.no_grad()
def lps(m, name, autocast=False):
    out = []
    for ids, s in zip(*SETS[name]):
        x = torch.tensor([ids], device=dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=autocast):
            lg = m.lm_head(m.model(input_ids=x, use_cache=False).last_hidden_state[0, s:-1]).float()
        out.append(torch.log_softmax(lg, -1).gather(-1, x[0, s + 1:, None])[:, 0].cpu())
    return torch.cat(out)


def d(a, b):
    x = (a - b).abs()
    return dict(mean=round(x.mean().item(), 4), p99=round(x.quantile(.99).item(), 3), max=round(x.max().item(), 3))


def free():
    gc.collect()
    torch.cuda.empty_cache()


rep = {}
m = load(BASE, torch.float32)
ref = {s: lps(m, s) for s in SETS}
del m
free()
m = load(BASE, torch.float32, "eager")
rep["base_fp32_eager_vs_sdpa"] = {s: d(lps(m, s), ref[s]) for s in SETS}
del m
free()
m = load(NEW, torch.float32)
pol = {s: lps(m, s, True) for s in SETS}
del m
free()
new = load(NEW, torch.bfloat16)
base_sd = {k: v for k, v in AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).state_dict().items()}
good = {k: v.cpu().clone() for k, v in new.state_dict().items()}
VAR = {"correct_sync": lambda k: False, "nothing_synced": lambda k: True, "embeddings_stale": lambda k: "embed_tokens" in k or "lm_head" in k,
       "norms_stale": lambda k: "norm" in k, "all_mlp_stale": lambda k: ".mlp." in k, "all_attention_stale": lambda k: ".self_attn." in k,
       "layer0_stale": lambda k: ".layers.0." in k, "layer27_stale": lambda k: ".layers.27." in k,
       "last_7_layers_stale": lambda k: any(f".layers.{i}." in k for i in range(21, 28))}
rep["engine_variants"] = {}
for n, f in VAR.items():
    sd = {k: (base_sd[k] if f(k) else v) for k, v in good.items()}
    changed = sum(not torch.equal(sd[k], good[k]) for k in good)
    new.load_state_dict(sd)
    rep["engine_variants"][n] = dict(tensors_wrong=changed, **{s: d(lps(new, s), pol[s]) for s in SETS})
for frac in (0.9, 0.5):
    sd = {k: (base_sd[k].float() + frac * (v.float() - base_sd[k].float())).to(torch.bfloat16) for k, v in good.items()}
    new.load_state_dict(sd)
    rep["engine_variants"][f"interpolated_{frac}_of_10_steps"] = dict(tensors_wrong=sum(not torch.equal(sd[k], good[k]) for k in good),
                                                                    **{s: d(lps(new, s), pol[s]) for s in SETS})
out = os.path.join(ROOT, "results", "pilot_analysis", "sync_failure_modes.json")
assert not os.path.exists(out), "refusing to overwrite"
jdump(out, rep)
print(json.dumps(rep, indent=1))
