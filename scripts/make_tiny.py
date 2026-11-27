"""Build a tiny random Qwen2 checkpoint (real tokenizer) and toy data for the end-to-end smoke test."""
import json, os, sys
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

tokdir, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(tokdir)
torch.manual_seed(0)
m = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tok) + (-len(tok)) % 64, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=2, max_position_embeddings=512, tie_word_embeddings=True, bos_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id))
m.save_pretrained(f"{out}/base")
tok.save_pretrained(f"{out}/base")
sft = [dict(id=f"s{i}", problem=f"What is {i % 4} plus {i // 4 % 4}?", reasoning_trace=f" ok </think> <answer>{4 if i % 2 else 5}</answer>") for i in range(32)]
pool = [dict(id=f"p{i}", question=f"What is {i % 4} plus {i // 4}?", answer="4") for i in range(16)]
for n, rows in (("sft.jsonl", sft), ("pool.jsonl", pool), ("eval_toy.jsonl", pool[:8])):
    with open(f"{out}/{n}", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
