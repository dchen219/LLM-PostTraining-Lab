"""Inference-only base-model verification with HF Transformers. No training.
Pre-declared: greedy, max_new_tokens=1024, full GSM8K test, left padding, bf16, sdpa.
Uses the project's own prompts, grader and GSM8K answer extraction. Writes only to OUT."""
import json, os, re, sys, time
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
os.chdir(PROJ)
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from posttrain.verifiers.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

OUT = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BS = int(sys.argv[3]) if len(sys.argv) > 3 else 24
os.makedirs(OUT, exist_ok=True)
MID = "Qwen/Qwen2.5-Math-1.5B"
tok = AutoTokenizer.from_pretrained(MID)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="cuda").eval()
rows = [json.loads(l) for l in open("data/gsm8k/test.jsonl", encoding="utf-8") if l.strip()]
if LIMIT:
    rows = rows[:LIMIT]
gold = [re.search(r"####\s*(.+)\s*$", r["answer"].strip()).group(1).strip() for r in rows]
summary = {}
for name, rf, stop in [("r1_zero", r1_zero_reward_fn, ["</answer>"]),
                       ("question_only", question_only_reward_fn, None)]:
    from posttrain.prompts import load
    tpl = load(name)
    prompts = [tpl.format(question=r["question"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i]))
    res = [None] * len(rows)
    t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    for s in range(0, len(order), BS):
        idx = order[s:s + BS]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        kw = dict(stop_strings=stop, tokenizer=tok) if stop else {}
        with torch.inference_mode():
            g = model.generate(**enc, max_new_tokens=1024, do_sample=False,
                               temperature=None, top_p=None, top_k=None,
                               pad_token_id=tok.pad_token_id, **kw)
        for i, ids in zip(idx, g[:, enc["input_ids"].shape[1]:]):
            txt = tok.decode(ids, skip_special_tokens=True)
            if stop and stop[0] in txt:
                txt = txt[:txt.index(stop[0]) + len(stop[0])]
            n = int((ids != tok.pad_token_id).sum())
            res[i] = dict(id=i, question=rows[i]["question"], ground_truth=gold[i], model_response=txt,
                          n_tokens=n, rewards=rf(txt, gold[i], fast=True))
        print(name, min(s + BS, len(order)), len(order), round(time.time() - t0), flush=True)
    with open(f"{OUT}/base_{name}.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = sum(r["rewards"]["answer_reward"] for r in res); fm = sum(r["rewards"]["format_reward"] for r in res)
    summary[name] = dict(n=len(res), correct=int(c), accuracy=c / len(res), format_ok=int(fm),
                         format_rate=fm / len(res), seconds=round(time.time() - t0, 1),
                         peak_mem_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                         model=MID, revision=model.config._commit_hash, batch_size=BS,
                         torch=torch.__version__, transformers=transformers.__version__,
                         gpu=torch.cuda.get_device_name(0), decoding="greedy, max_new_tokens=1024, bf16")
    json.dump(summary, open(f"{OUT}/base_metrics.json", "w"), indent=2)
    print("METRICS", name, json.dumps(summary[name]), flush=True)
