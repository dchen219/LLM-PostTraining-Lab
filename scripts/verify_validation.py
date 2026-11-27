"""Re-derive the A6000 validation results from raw artifacts; summary.json decides nothing. Run from inside the unpacked package that was uploaded.

Env: A5_VAL (artifacts), REAL_DATA (package data/stage0), TOKDIR (pinned tokenizer), OUT (new json), plus the server's PT_RUNS, PT_DATA and PT_MODEL paths.
"""
import collections, csv, glob, hashlib, json, math, os, random, statistics as st, sys
V, DATA = os.environ["A5_VAL"], os.environ["REAL_DATA"]
from posttrain.common import code_hash, load_config, sha_obj
from posttrain.verify import score
J = lambda p: json.load(open(p))
JL = lambda p: [json.loads(x) for x in open(p)]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rep, bad = {}, []


def ck(name, ok, detail=""):
    rep[name] = (bool(ok), detail)
    print("OK  " if ok else "BAD ", name, detail)


# 1 provenance
runs = ["grpo_ref", "grpo_resume", "sft", "grpo_from_sft"]
meta = {d: J(f"{V}/{d}/run.json") for d in runs}
ch = code_hash()
ck("code hash of package == hash in every run", all(m["code_sha256"] == ch for m in meta.values()), ch[:12])
ck("config hash recomputes from stored config", all(sha_obj(m["config"]) == m["config_sha256"] for m in meta.values()))
for d, m in meta.items():
    av = m["argv"]
    c = load_config(av[av.index("--config") + 1].split("/a5/")[-1], av[av.index("--set") + 1:])
    same = sha_obj(c) == m["config_sha256"]
    ck(f"{d}: config rebuilt from package yaml + recorded command is identical", same,
       "" if same else str([(k, c.get(k), m["config"].get(k)) for k in c if c.get(k) != m["config"].get(k)]))
    print("     env", {k: m["env"][k] for k in ("python", "torch", "transformers", "vllm", "math-verify", "gpus")})
ck("base weights hash is the pin", all(J(f"{V}/{d}/DONE.json")["lineage"]["parent"]["weights_sha256"]["model.safetensors"].startswith("f7aa8d665ab4") for d in runs[:3]))
ls, lg = J(f"{V}/sft/DONE.json")["lineage"], J(f"{V}/grpo_from_sft/DONE.json")["lineage"]
ck("V5 parent weights == V4 output weights", ls["weights_sha256"] == lg["parent"]["weights_sha256"] and lg["chain"] == ["base", "sft", "grpo"])
ck("checkpoints all differ", len({J(f"{V}/{d}/DONE.json")["lineage"]["weights_sha256"]["model.safetensors"] for d in runs}) == 4)
ck("RL pool file hash", J(f"{V}/grpo_ref/data.json")["file_sha256"] == sha(DATA + "/rl_train_pool.jsonl"))
sd = J(f"{V}/sft/data.json")
ck("SFT data file hash and subset ids", sd["file_sha256"] == sha(DATA + "/sft_all_labeled_4836.jsonl") and sd["ids"] == J(DATA + "/subsets/reverified_full.json")["ids"], f"n={sd['n']}")
tj, lj = J(f"{V}/sft/checkpoint/tokenizer.json"), J(os.environ["TOKDIR"] + "/tokenizer.json")
mg = lambda j: [m if isinstance(m, str) else " ".join(m) for m in j["model"]["merges"]]
ck("tokenizer: same vocabulary, merges and rules as the pinned copy (files were re-serialised by transformers)", tj["model"]["vocab"] == lj["model"]["vocab"] and mg(tj) == mg(lj) and all(tj[k] == lj[k] for k in lj if k != "model")
   and len({sha(f"{V}/{d}/checkpoint/tokenizer.json") for d in runs}) == 1)

# 2 timeline
steps = JL(f"{V}/logs/steps.jsonl")
ck("6 steps, all rc 0, in order", [s["step"] for s in steps] == "V1_sync V3_memory V2a_grpo_ref V2b_grpo_resume V4_sft_mb2 V5_from_sft".split() and all(s["rc"] == 0 for s in steps))
ck("tracebacks/errors in logs", not any(w in open(f, errors="replace").read() for f in glob.glob(V + "/logs/V*.log") for w in ("Traceback", "SystemExit", "check failed", "NaN")))

# 3 per-step gates and rollouts
pool = {r["id"]: r for r in JL(DATA + "/rl_train_pool.jsonl")}
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ["TOKDIR"])
sr = {r["id"]: r for r in JL(DATA + "/sft_all_labeled_4836.jsonl")}
order = list(range(len(sd["ids"]))); random.Random("0-0").shuffle(order)
mine = [sum(len(tok(sr[sd["ids"][j]]["reasoning_trace"], add_special_tokens=False)["input_ids"]) + 1 for j in order[k * 32:(k + 1) * 32]) for k in range(10)]
ck("sft: batch token counts rebuilt from data and seed", mine == [r["tokens"] for r in JL(f"{V}/sft/metrics.jsonl") if r["event"] == "train"])
allm = {}
for d in runs:
    m = JL(f"{V}/{d}/metrics.jsonl")
    t = [r for r in m if r["event"] == "train"]
    allm[d] = t
    ck(f"{d}: steps are exactly 1..n, each once", [r["step"] for r in t] == list(range(1, len(t) + 1)) and len(t) == J(f"{V}/{d}/DONE.json")["steps"], str(len(t)))
    ck(f"{d}: all numeric values finite", all(math.isfinite(v) for r in t for v in r.values() if isinstance(v, (int, float))))
    w = meta[d]["config"]["train"]["warmup"]; lr = float(meta[d]["config"]["train"]["lr"])
    ck(f"{d}: lr schedule", all(abs(r["lr"] - lr * min(1, r["step"] / w)) < 1e-12 for r in t), str([f"{r['lr']:.1e}" for r in t]))
    if not d.startswith("grpo"):
        continue
    ck(f"{d}: exact gate 198 checked / 0 mismatched every step", all(r["params_checked"] == 198 and r["params_mismatched"] == 0 for r in t))
    ck(f"{d}: rollout gap <=0.05 every step, every token covered", all(r["rollout_logprob_mad"] <= .05 for r in t), str([round(r["rollout_logprob_mad"], 4) for r in t]))
    print("      p99/max", [(round(r["rollout_logprob_p99"], 3), round(r["rollout_logprob_max"], 2)) for r in t])
    rng_seen = []
    diff = adv_bad = 0; long = 0; rescored = 0; mismatch = 0; rt = 0
    for r in t:
        ro = JL(f"{V}/{d}/rollouts/step{r['step']}.jsonl")
        assert len(ro) == 256
        rewards = []
        for i, x in enumerate(ro):
            assert x["id"] in pool and pool[x["id"]]["answer"] == x["answer"]
            s = score(x["response"], x["answer"])["reward"]
            rewards.append(x["reward"]); rescored += 1; mismatch += s != x["reward"]
            n = len(tok(x["response"], add_special_tokens=False)["input_ids"]); rt += abs(n - x["n_tokens"]) > 3
        for g in range(32):
            rr = rewards[g * 8:g * 8 + 8]; mu = sum(rr) / 8
            adv_bad += any(abs(ro[g * 8 + j]["adv"] - (rr[j] - mu)) > 1e-6 for j in range(8)) or len({ro[g * 8 + j]["id"] for j in range(8)}) != 1
        inf = sum(1 for g in range(32) if 0 < sum(rewards[g * 8:g * 8 + 8]) < 8) / 32
        ok = (sum(x["n_tokens"] for x in ro) == r["rollout_logprob_tokens"] == r["trained_tokens"] and abs(sum(rewards) / 256 - r["reward_mean"]) < 1e-9 and abs(inf - r["informative_rate"]) < 1e-9
              and max(x["n_tokens"] for x in ro) == r["len_max"] and abs(sum(x["n_tokens"] for x in ro) / 256 - r["len_mean"]) < 1e-9)
        diff += not ok
        rng_seen.append([ro[g * 8]["id"] for g in range(32)])
    ck(f"{d}: metrics recomputed from rollouts (tokens, reward, informative, lengths)", diff == 0)
    ck(f"{d}: rewards re-scored on a different machine", mismatch == 0, f"{rescored} rescored, {mismatch} differ")
    ck(f"{d}: advantages recomputed, 32 groups of 8", adv_bad == 0)
    ck(f"{d}: step_s is the sum of its parts", all(abs(r["step_s"] - sum(r[k] for k in ("sync_s", "rollout_s", "reward_s", "old_lp_s", "update_s"))) < 1e-6 for r in t))
# prompts sampled as the recorded seed says
for d in ("grpo_ref", "grpo_resume", "grpo_from_sft"):
    from posttrain.diagnostics import DifficultySampler
    s = DifficultySampler(pool, random.Random(f"{meta[d]['config']['seed']}-prompts"), False)
    ok = True
    for r in allm[d]:
        ids = s.sample(32)
        ro = JL(f"{V}/{d}/rollouts/step{r['step']}.jsonl")
        ok &= [ro[g * 8]["id"] for g in range(32)] == list(ids)
        s.update(ids, [sum(x["reward"] for x in ro[g * 8:g * 8 + 8]) / 8 for g in range(32)])
    ck(f"{d}: prompts equal the ones the seed generates from scratch", ok)

# 4 resume
A, B = allm["grpo_ref"], allm["grpo_resume"]
R = lambda d, k: [(x["id"], x["response"]) for x in JL(f"{V}/{d}/rollouts/step{k}.jsonl")]
print("   identical rollouts ref vs resume:", {k: sum(a == b for a, b in zip(R("grpo_ref", k), R("grpo_resume", k))) for k in (1, 2, 3, 4)}, "of 256")
for k in ("grad_norm", "param_mean_abs_delta", "param_changed_frac_bf16", "wall_s", "lr", "gpu1_peak_gib"):
    print("   ", k, [r[k] for r in A], [r[k] for r in B])
res = JL(f"{V}/grpo_resume/resumes.jsonl")
ck("resume: one resume, same code", len(res) == 1 and res[0]["code_sha256"] == ch)
ck("resume: Adam moments restored (update size at step 3 not that of a fresh optimizer)", abs(B[2]["param_mean_abs_delta"] / A[2]["param_mean_abs_delta"] - 1) < .25,
   f"ref {A[2]['param_mean_abs_delta']:.3e} resumed {B[2]['param_mean_abs_delta']:.3e}; fresh Adam at this lr would be about {A[0]['param_mean_abs_delta'] / A[0]['lr'] * A[2]['lr']:.3e}")

# 5 sync.json / policy.json gates recomputed
s = J(f"{V}/sync.json"); P = lambda k: s[k]["params_mismatched"]
ck("V1 recomputed", P("1_after_initial_sync") == P("6_after_resync") == P("9_after_final_sync") == 0 and all(s[k]["params_checked"] == 198 and s[k]["missing"] == s[k]["unexpected"] == 0 for k in s if isinstance(s[k], dict) and "params_checked" in s[k])
   and P("3_stale_by_one_step") > 0 and s["4_one_tensor_not_synced"]["mismatch_names"] == ["model.layers.1.self_attn.qkv_proj.weight"] and s["5_embeddings_not_synced"]["mismatch_names"] == ["model.embed_tokens.weight"]
   and max(s["2_rollouts_after_sync"]["rollout_logprob_mad"], s["7_rollouts_after_resync"]["rollout_logprob_mad"]) < .05 < s["8_rollouts_from_stale_engine"]["rollout_logprob_mad"])
p = J(f"{V}/policy.json")
f32 = all(v["cosine"] >= .9999 and abs(v["norm_ratio"] - 1) <= .001 and v["same_update"] for v in p["gradients_fp32"].values())
b16 = all(v["cosine"] >= .99 and abs(v["norm_ratio"] - 1) <= .02 and v["same_update"] for v in p["gradients_bf16"].values())
c = p["gradients_fp32_control_one_rollout_lost"]
ck("V3 recomputed: gradients, control", f32 and b16 and c["cosine"] < .9999 and not p["dry_run"])
must = [v for v in p["stress"].values() if v["must_fit"]]
ck("V3 recomputed: memory", all(v["fits"] and v["finite"] and v["peak_allocated_gib"] <= 45.4 for v in must), f"worst {max(v['peak_allocated_gib'] for v in must)} of limit {p['memory_limit_gib']:.2f}")
print("   prediction error GiB", {k: round(v["peak_allocated_gib"] - v["predicted_gib"], 2) for k, v in p["stress"].items() if v["fits"]})

# 6 gpu.csv
rows = list(csv.reader(open(f"{V}/logs/gpu.csv")))[1:]
for g in ("0", "1"):
    x = [(int(r[2]), int(r[4])) for r in rows if r[1].strip() == g]
    print("   GPU", g, "samples", len(x), "first", x[0], "last", x[-1], "max MiB", max(a for a, _ in x), "mean util", round(st.mean(b for _, b in x)))
print("   first", rows[0][0], "last", rows[-1][0])
# 7 timing
for d in runs:
    t = allm[d]; gaps = [round(t[i]["wall_s"] - t[i - 1]["wall_s"] - t[i]["step_s"], 1) for i in range(1, len(t))]
    print("   TIME", d, "startup", round(t[0]["wall_s"] - t[0]["step_s"], 1), "steps", [round(r["step_s"], 1) for r in t], "gaps", gaps, "total", round(J(f"{V}/{d}/DONE.json")["wall_s"], 1), "end_after_last", round(J(f"{V}/{d}/DONE.json")["wall_s"] - t[-1]["wall_s"], 1), "sumtok", [r.get("trained_tokens", r.get("tokens")) for r in t])
    if d != "sft":
        print("        parts", {k: round(st.mean(r[k] for r in t), 2) for k in ("sync_s", "sync_verify_s", "rollout_s", "reward_s", "old_lp_s", "update_s", "rollout_tok_per_s")})
mem = [(r["gpu1_peak_gib"], d, r["step"]) for d in runs for r in allm[d]]
ck("training peak below 46 GiB (policy GPU)", max(mem)[0] < 46, str(max(mem)))
files = sorted(f for f in glob.glob(V + "/**/*", recursive=True) if os.path.isfile(f))
json.dump(dict(package="bc0b6445155679ceb2fc825de03b3270134176f25343e1b5b64da325237ce3c7", checks={k: dict(ok=v[0], detail=v[1]) for k, v in rep.items()}, n_checks=len(rep), failed=[k for k, v in rep.items() if not v[0]],
               artifact_sha256={os.path.relpath(f, V): sha(f) for f in files}), open(os.environ["OUT"], "x"), indent=1)
print("FAILED:", [k for k, v in rep.items() if not v[0]], "of", len(rep))
