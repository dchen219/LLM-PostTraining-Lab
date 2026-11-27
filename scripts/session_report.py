"""Report for recovered Session A / B artifacts. Reads only. Run on the laptop.

python scripts/session_report.py a|b --runs ~/a5_sessions --out results/session_a
Every accuracy is recomputed from item files. Columns: strict (frozen verifier 2.0.0, the reported metric), second (strict, or math-verify accepts the
text inside the answer tags), relaxed (second, or the failure taxonomy found the right answer outside the required format). A gain in strict that is
absent from relaxed is answer formatting, not better mathematics.
"""
import argparse, csv, json, logging, os, statistics as st, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
logging.disable(logging.CRITICAL)
import stage0_verify_audit as A
from posttrain.common import jload
from posttrain.stats import holm, paired, wilson
from tier_ids import A as TIER_A

ap = argparse.ArgumentParser()
ap.add_argument("session", choices=("a", "b"))
ap.add_argument("--runs", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = os.path.expanduser(a.runs)
assert not os.path.exists(a.out), "refusing to overwrite"
J = lambda p: json.load(open(p, encoding="utf-8"))
path = lambda i: f"{R}/{'sft' if i.startswith('SFT') else 'grpo'}/{i.split('-', 1)[1]}"
cache = {}


def items(i, tag, name):
    """(strict, second, relaxed) as {id: bool}, and the stored summary must match."""
    k = (i, tag, name)
    if k not in cache:
        p = f"{path(i)}/eval/{tag}_{name}.jsonl"
        if not os.path.exists(p):
            return None
        s, b, r = {}, {}, {}
        for x in jload(p):
            s[x["id"]] = bool(x["correct"])
            t = x["response"]
            b[x["id"]] = s[x["id"]] or bool(x["format"] and A.second_verifier(t.split("<answer>")[-1].replace("</answer>", "").strip(), str(x["answer"])))
            r[x["id"]] = b[x["id"]] or x["category"] == "format_fail_answer_right"
        cache[k] = (s, b, r)
    return cache[k]


acc = lambda d: sum(d.values()) / len(d)
m = lambda rows, k: [r[k] for r in rows if r.get(k) is not None]
ids = [i for i in TIER_A if i.startswith("SFT" if a.session == "a" else "GRPO")]
rows, missing = [], []
for i in ids:
    d = path(i)
    if not os.path.exists(d + "/DONE.json"):
        missing.append(i)
        continue
    run, done, data = J(d + "/run.json"), J(d + "/DONE.json"), J(d + "/data.json")
    tr_all = [x for x in jload(d + "/metrics.jsonl") if x["event"] == "train"]
    tr = list({x["step"]: x for x in tr_all}.values())  # a resume before the rewind fix could log a step twice: keep the last row per step
    tr.sort(key=lambda x: x["step"])
    c, o = run["config"], run["config"]["train"]
    row = dict(id=i, lr=float(o["lr"]), seed=c["seed"], steps=done["steps"], lineage=">".join(done["lineage"]["chain"]), weights=done["lineage"]["weights_sha256"]["model.safetensors"][:12],
               parent=next(iter(done["lineage"]["parent"]["weights_sha256"].values()))[:12], data_sha256=data["file_sha256"][:12], n_data=data.get("n", data.get("n_pool")), ids_sha256=data.get("ids_sha256", "")[:12],
               code=run["code_sha256"][:12], verifier=(run.get("verifier") or {}).get("version"), micro_batch=o["micro_batch"], token_budget=o.get("token_budget"),
               wall_min=done["wall_s"] / 60, step_s=st.mean(m(tr, "step_s")), grad_norm_mean=st.mean(m(tr, "grad_norm")), grad_norm_max=max(m(tr, "grad_norm")),
               loss_first=tr[0]["loss"], loss_last5=st.mean(m(tr, "loss")[-5:]), entropy_first=tr[0].get("entropy"), entropy_last5=st.mean(m(tr, "entropy")[-5:]),
               policy_peak_gib=max(m(tr, "gpu1_peak_gib")), engine_peak_gib=max(m(tr, "gpu0_peak_gib")), resumes=len(jload(d + "/resumes.jsonl")) if os.path.exists(d + "/resumes.jsonl") else 0,
               duplicated_step_rows=len(tr_all) - len(tr))
    if a.session == "a":
        row.update(tok_per_s=st.mean(m(tr, "tok_per_s")))
    else:
        f, l = tr[:10], tr[-10:]
        row.update(reward_first10=st.mean(m(f, "reward_mean")), reward_last10=st.mean(m(l, "reward_mean")), informative_first10=st.mean(m(f, "informative_rate")), informative_last10=st.mean(m(l, "informative_rate")),
                   zero_adv_rate=sum(m(tr, "zero_adv")) / sum(m(tr, "groups")), len_first10=st.mean(m(f, "len_mean")), len_last10=st.mean(m(l, "len_mean")), truncated_last10=st.mean(m(l, "truncated_rate")),
                   clip_frac=st.mean(m(tr, "clip_frac")), ratio_min=min(m(tr, "ratio_min")), ratio_max=max(m(tr, "ratio_max")), rollout_gap_mean=st.mean(m(tr, "rollout_logprob_mad")), rollout_gap_max=max(m(tr, "rollout_logprob_mad")),
                   rollout_token_max=max(m(tr, "rollout_logprob_max")), weights_mismatched=sum(m(tr, "params_mismatched")), exact_checks=sum(x > 0 for x in m(tr, "params_checked")),
                   rollout_tok_per_s=st.mean(m(tr, "rollout_tok_per_s")), additive_rule_rewards=sum(m(tr, "verifier_additive_rule")), verifier_errors=sum(m(tr, "verifier_errors")))
    for name in ("gsm8k_dev", "math_dev"):
        s, b, r = items(i, "final_step", name)
        e = done["eval"][name]
        assert sum(s.values()) == e["correct"] and len(s) == e["n"], f"summary and items disagree: {i} {name}"
        lo, hi = wilson(e["correct"], e["n"])
        row.update({f"{name}_strict": acc(s), f"{name}_ci": f"{lo:.3f}-{hi:.3f}", f"{name}_second": acc(b), f"{name}_relaxed": acc(r), f"{name}_format": e["format_rate"], f"{name}_tokens": e["mean_tokens"]})
    rows.append(row)

S = "SFT-"
PAIRS = dict(a=[("data quality, full (pre-registered)", S + "exact_full_s0", "final_step", S + "reverified_full_s0", "final_step"),
                ("data quality, matched 512 (pre-registered)", S + "exact_n512_s0", "final_step", S + "reverified_n512_s0", "final_step"),
                ("composition at matched size", S + "exact_full_s0", "final_step", S + "reverified_sizecontrol_s0", "final_step"),
                ("volume", S + "reverified_sizecontrol_s0", "final_step", S + "reverified_full_s0", "final_step")],
             b=[("pathway B vs its SFT parent (pre-registered)", S + "reverified_n512_s0", "final_step", "GRPO-B_sft512_s2026", "final_step"),
                ("pathway A vs base, r1_zero prompt (pre-registered)", "GRPO-A_base_s2026", "step0", "GRPO-A_base_s2026", "final_step"),
                ("re-evaluation noise: SFT parent vs B step 0, same weights", S + "reverified_n512_s0", "final_step", "GRPO-B_sft512_s2026", "step0"),
                ("sequential vs direct", "GRPO-A_base_s2026", "final_step", "GRPO-B_sft512_s2026", "final_step")])[a.session]
cmp = []
for name in ("gsm8k_dev", "math_dev"):
    grp = []
    for label, x, tx, y, ty in PAIRS:
        ix, iy = items(x, tx, name), items(y, ty, name)
        if ix and iy:
            grp.append(dict(label=label, set=name, a=f"{x}:{tx}", b=f"{y}:{ty}", strict=paired(ix[0], iy[0]), second_diff=acc(iy[1]) - acc(ix[1]), relaxed_diff=acc(iy[2]) - acc(ix[2])))
    h = holm({g["label"]: g["strict"]["p"] for g in grp})
    for g in grp:
        g["p_holm"] = h[g["label"]]
    cmp += grp
os.makedirs(a.out)
json.dump(dict(session=a.session, missing=missing, runs=rows, comparisons=cmp), open(a.out + "/report.json", "w"), indent=1)
if rows:
    with open(a.out + "/runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
pc = lambda x: f"{100 * x:.1f}%"
L = [f"# Session {a.session.upper()} report", "", "Development sets only. One seed per run: no claim about seed variance. Strict = frozen verifier 2.0.0.", "",
     "| Run | lr | Data | GSM8K dev strict (95% CI) | second | relaxed | format | MATH dev strict | peak GiB | min |", "|---|---|---|---|---|---|---|---|---|---|"]
L += [f"| {r['id']} | {r['lr']:g} | {r['n_data']} | {pc(r['gsm8k_dev_strict'])} ({r['gsm8k_dev_ci']}) | {pc(r['gsm8k_dev_second'])} | {pc(r['gsm8k_dev_relaxed'])} | {pc(r['gsm8k_dev_format'])} | {pc(r['math_dev_strict'])} | {r['policy_peak_gib']} | {r['wall_min']:.0f} |" for r in rows]
if a.session == "b" and rows:
    L += ["", "| Run | reward first/last 10 | informative first/last 10 | zero-adv rate | entropy first/last 5 | length first/last 10 | clip fraction | grad norm mean/max | rollout gap max | token max | mismatches |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['id']} | {r['reward_first10']:.3f}/{r['reward_last10']:.3f} | {r['informative_first10']:.2f}/{r['informative_last10']:.2f} | {r['zero_adv_rate']:.2f} | {r['entropy_first']:.2f}/{r['entropy_last5']:.2f} | {r['len_first10']:.0f}/{r['len_last10']:.0f} | {r['clip_frac']:.4f} | {r['grad_norm_mean']:.3f}/{r['grad_norm_max']:.3f} | {r['rollout_gap_max']:.4f} | {r['rollout_token_max']:.2f} | {r['weights_mismatched']} |" for r in rows]
L += ["", "| Comparison (b minus a) | Set | Strict diff (95% CI) | p | Holm p | second diff | relaxed diff |", "|---|---|---|---|---|---|---|"]
L += [f"| {g['label']} | {g['set']} | {100 * g['strict']['diff']:+.1f} pp [{100 * g['strict']['ci95'][0]:+.1f}, {100 * g['strict']['ci95'][1]:+.1f}] | {g['strict']['p']:.3g} | {g['p_holm']:.3g} | {100 * g['second_diff']:+.1f} | {100 * g['relaxed_diff']:+.1f} |" for g in cmp]
L += ["", "Not finished: " + (", ".join(missing) or "none")]
open(a.out + "/REPORT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
