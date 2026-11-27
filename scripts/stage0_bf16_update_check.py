"""CPU diagnostic: what fraction of real Qwen2.5-Math-1.5B weights change when an Adam-sized step of +/-lr is
applied in pure bf16? (Adam's first step has magnitude ~lr per coordinate.) No training, read only."""
import json, sys, torch
from safetensors import safe_open

path, out = sys.argv[1], sys.argv[2]
g = torch.Generator().manual_seed(2026)
res = {lr: [0, 0] for lr in (1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4)}
with safe_open(path, "pt") as f:
    for k in f.keys():
        w = f.get_tensor(k).to(torch.bfloat16)
        if w.numel() > 2_000_000:
            w = w.flatten()[torch.randint(0, w.numel(), (2_000_000,), generator=g)]
        s = (torch.randint(0, 2, w.shape, generator=g) * 2 - 1).float()
        for lr in res:
            # p.add_(update) in bf16: rounded once after the addition
            n = (w.float() + lr * s).to(torch.bfloat16)
            res[lr][0] += int((n != w).sum()); res[lr][1] += w.numel()
rep = {f"{lr:g}": {"changed_pct": round(100 * a / b, 2), "unchanged_pct": round(100 - 100 * a / b, 2), "sampled": b} for lr, (a, b) in res.items()}
json.dump(rep, open(out, "w"), indent=1); print(json.dumps(rep, indent=1))
