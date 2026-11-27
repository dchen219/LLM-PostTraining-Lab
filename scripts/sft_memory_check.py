"""SFT micro-batch memory check. Calibrates padded tokens against the SFT pilot's measured peaks, then predicts the peak for the full-data arms.

Reconstructs train_sft batching exactly: subset order, epoch shuffle seeded by f"{seed}-{epoch}", batch_size, then micro-batches after the
character-length sort used by sft_step.
"""
import json, math, os, random, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from posttrain.common import ROOT, jdump, jload
from posttrain.evaluate import template
from posttrain.model_io import load_tokenizer

BASE = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2")
P = os.path.expanduser("~/a5_pilot")
OUT = os.path.join(ROOT, "results", "pilot_analysis", "sft_memory.json")
tok = load_tokenizer(BASE)
tpl = template("r1_zero")
rows = {r["id"]: r for r in jload(os.path.join(ROOT, "data", "stage0", "sft_all_labeled_4836.jsonl"))}
enc = {}


def ntok(r):
    if r["id"] not in enc:
        a = len(tok(tpl.format(question=r["problem"].strip()), add_special_tokens=False)["input_ids"])
        b = len(tok(r["reasoning_trace"], add_special_tokens=False)["input_ids"])
        enc[r["id"]] = (a, b + 1)  # + EOS
    return enc[r["id"]]


def batches(ids, seed, bs, mb, epochs):
    ex = [rows[i] for i in ids]
    for e in range(epochs):
        order = list(range(len(ex)))
        random.Random(f"{seed}-{e}").shuffle(order)
        for k in range(math.ceil(len(ex) / bs)):
            b = [ex[i] for i in order[k * bs:(k + 1) * bs]]
            b = sorted(b, key=lambda r: len(r["problem"]) + len(r["reasoning_trace"]))
            for i in range(0, len(b), mb):
                T = max(sum(ntok(r)) - 1 for r in b[i:i + mb])  # encode drops the last position in the shifted batch
                yield dict(epoch=e, batch=k, B=len(b[i:i + mb]), T=T, padded=len(b[i:i + mb]) * T)


pilot_ids = json.load(open(f"{ROOT}/data/stage0/subsets/reverified_n128_s0.json"))["ids"]
mb_rows = list(batches(pilot_ids, 0, 32, 8, 2))
per_step = {}
for r in mb_rows:
    per_step.setdefault((r["epoch"], r["batch"]), []).append(r["padded"])
pk = {r["step"]: r["gpu1_peak_gib"] for r in jload(P + "/sft/pilot_sft/metrics.jsonl") if r["event"] == "train"}
steps = []
per_epoch = math.ceil(128 / 32)
for step in range(1, 7):
    e, k = divmod(step - 1, per_epoch)
    steps.append(dict(step=step, max_padded=max(per_step[(e, k)]), peak_gib=pk[step]))
use = [s for s in steps if s["step"] > 1]
mx, my = st.mean(s["max_padded"] for s in use), st.mean(s["peak_gib"] for s in use)
b = sum((s["max_padded"] - mx) * (s["peak_gib"] - my) for s in use) / sum((s["max_padded"] - mx) ** 2 for s in use)
a0 = my - b * mx
fits = [dict(step=s["step"], max_padded=s["max_padded"], peak_gib=s["peak_gib"], pred=a0 + b * s["max_padded"]) for s in steps]
arms = {}
for name in ("raw_full", "reverified_full", "exact_full", "reverified_sizecontrol_s0"):
    f = f"{ROOT}/data/stage0/subsets/{name}.json"
    if not os.path.exists(f):
        continue
    ids = json.load(open(f))["ids"]
    L = sorted(sum(ntok(rows[i])) - 1 for i in ids)
    seen = list(batches(ids, 0, 32, 8, 1))
    worst = max(seen, key=lambda r: r["padded"])
    arms[name] = dict(n=len(ids), len_p50=L[len(L) // 2], len_p99=L[int(.99 * (len(L) - 1))], len_max=L[-1], over_1536=sum(x > 1535 for x in L),
                      worst_microbatch_padded=worst["padded"], worst_T=worst["T"], predicted_peak_gib_mb8=a0 + b * worst["padded"],
                      microbatches=sum(1 for _ in seen))
rep = dict(method="linear fit of SFT pilot peak (steps 2-6) on the largest padded micro-batch of each step", intercept_gib=a0, gib_per_1000_padded=1000 * b,
           pilot_steps=fits, arms=arms)
assert not os.path.exists(OUT), "refusing to overwrite"
jdump(OUT, rep)
print(json.dumps(rep, indent=1))
