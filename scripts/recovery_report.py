"""Diagnosis of the failed GRPO-A_base_s2026 and comparison of its post-hoc recovery branches. Reads only. Run on the laptop.

python scripts/recovery_report.py --runs ~/a5_recovery --out results/grpo_a_incident
--runs is a copy of the server's runs/ folder (without *.safetensors and state.pt*). Works before the branches exist (diagnosis only).
Development sets only. One seed, one branch per learning rate: differences are descriptive, not evidence about learning rates in general.
"""
import argparse, glob, json, logging, os, statistics as st, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
logging.disable(logging.CRITICAL)
from posttrain.common import jload  # noqa: E402
from posttrain.stats import holm, paired, wilson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = os.path.expanduser(a.runs)
assert not os.path.exists(a.out), "refusing to overwrite"
ORIG = f"{R}/grpo/A_base_s2026"
BR = {"A1 replay lr 1e-5": f"{R}/recovery/A_replay80_lr1e-5_s2026", "A2 lr 5e-6": f"{R}/recovery/A_recovery80_lr5e-6_s2026"}
J = lambda p: json.load(open(p, encoding="utf-8"))
mean = lambda x: st.mean(x) if x else None
f = lambda v, d=3: "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def train_rows(d):
    rows = [r for r in jload(d + "/metrics.jsonl") if r.get("event") == "train"]
    steps = [r["step"] for r in rows]
    by = {r["step"]: r for r in rows}  # a step logged twice (an old-style resume) keeps its LAST row; the count is reported
    return [by[s] for s in sorted(by)], len(steps) - len(set(steps))


def col(rows, k):
    return [r[k] for r in rows if isinstance(r.get(k), (int, float))]


out, L = dict(), ["# GRPO-A_base_s2026: failure diagnosis and post-hoc recovery branches", "",
                  "Development sets only. The branches are diagnostic and do not replace the pre-registered run, which is reported as failed.", ""]
# ---------- 1. the original run ----------
if os.path.exists(ORIG + "/metrics.jsonl"):
    rows, dup = train_rows(ORIG)
    base = mean(col(rows[:10], "rollout_logprob_mad"))
    first2x = next((r["step"] for r in rows if (r.get("rollout_logprob_mad") or 0) > 2 * base), None) if base else None
    keys = ("reward_mean", "format_rate", "verifier_errors", "informative_rate", "entropy", "grad_norm", "param_mean_abs_delta", "len_mean", "truncated_rate",
            "rollout_logprob_mad", "rollout_logprob_p99", "rollout_logprob_max", "sync_mean_abs_logprob_diff")
    show = [r for r in rows if r["step"] % 10 == 0 or r["step"] > 70]
    L += ["## 1. Original run (read only)", "", f"Train rows: {len(rows)} steps, last step {rows[-1]['step'] if rows else None}; duplicated step rows: {dup}; "
          f"resumes: {len(jload(ORIG + '/resumes.jsonl')) if os.path.exists(ORIG + '/resumes.jsonl') else 0}; DONE.json: {os.path.exists(ORIG + '/DONE.json')}", "",
          f"Rollout log-prob gap, mean of steps 1-10: {f(base, 4)}; first step above twice that: {first2x}", "",
          "| step | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    L += [f"| {r['step']} | " + " | ".join(f(r.get(k), 4) for k in keys) + " |" for r in show]
    pairs = {}
    for k in ("len_mean", "entropy", "truncated_rate", "reward_mean", "grad_norm"):
        xs = [(r["rollout_logprob_mad"], r[k]) for r in rows if isinstance(r.get(k), (int, float)) and isinstance(r.get("rollout_logprob_mad"), (int, float))]
        if len(xs) > 3:
            try:
                pairs[k] = st.correlation(*zip(*xs))
            except st.StatisticsError:
                pairs[k] = None
    L += ["", "Correlation across steps of the mean rollout gap with: " + ", ".join(f"{k} {f(v, 2)}" for k, v in pairs.items()) + " (association only)"]
    ev = [r for r in jload(ORIG + "/metrics.jsonl") if r.get("event") == "eval"]
    L += ["", "Evaluations: " + "; ".join(f"step {r['step']} GSM8K {f(r.get('gsm8k_dev/accuracy'))} (format {f(r.get('gsm8k_dev/format_rate'))}), "
                                          f"MATH {f(r.get('math_dev/accuracy'))}" for r in ev)]
    for p in sorted(glob.glob(ORIG + "/rollouts/step*.jsonl"), key=lambda p: int(p.split("step")[-1].split(".")[0]))[-3:]:
        x = jload(p)
        L.append(f"Saved rollouts {os.path.basename(p)}: n {len(x)}, reward {f(mean([y['reward'] for y in x]))}, mean tokens {f(mean([y['n_tokens'] for y in x]), 0)}, "
                 f"at 1,024 tokens {sum(y['n_tokens'] >= 1024 for y in x)}")
    out["original"] = dict(steps=len(rows), duplicated_rows=dup, gap_baseline=base, first_step_gap_over_2x=first2x, correlations=pairs, rows=rows)
    L.append("")


# ---------- 2. branches ----------
def items(d, tag, name):
    p = f"{d}/eval/{tag}_{name}.jsonl"
    if not os.path.exists(p):
        return None
    import stage0_verify_audit as A
    s, b, r = {}, {}, {}
    for x in jload(p):
        s[x["id"]] = bool(x["correct"])
        b[x["id"]] = s[x["id"]] or bool(x["format"] and A.second_verifier(x["response"].split("<answer>")[-1].replace("</answer>", "").strip(), str(x["answer"])))
        r[x["id"]] = b[x["id"]] or x["category"] == "format_fail_answer_right"
    return s, b, r


acc = lambda d: sum(d.values()) / len(d)
B = {}
for name, d in BR.items():
    if not os.path.exists(d + "/run.json"):
        continue
    rows, dup = train_rows(d)
    fs = [r for r in jload(d + "/metrics.jsonl") if r.get("event") == "fork_start"]
    status = "finished" if os.path.exists(d + "/DONE.json") else ("stopped at a gate" if os.path.exists(d + "/FAILED.json") else "incomplete")
    end = J(d + "/DONE.json") if status == "finished" else (J(d + "/FAILED.json") if status == "stopped at a gate" else {})
    fork = J(d + "/run.json")["config"]["fork"]
    b = dict(status=status, last_step=rows[-1]["step"] if rows else fork["source_step"], duplicated_rows=dup, state_digest=fs[0]["state_digest"] if fs else None,
             source_unchanged=(end.get("source_check") or {}).get("source_unchanged"), lr=sorted({r["lr"] for r in rows}), differences=[x["key"] for x in fork["differences"]],
             wall_h=end.get("wall_s", max(col(jload(d + "/metrics.jsonl"), "wall_s") or [0])) / 3600)
    b["gpu_h"] = 2 * b["wall_h"]
    for k in ("reward_mean", "entropy", "grad_norm", "update_rms", "update_rel", "rollout_logprob_mad", "informative_rate", "len_mean", "truncated_rate", "format_rate"):
        v = col(rows, k)
        b[k] = dict(first5=mean(v[:5]), last5=mean(v[-5:]), max=max(v) if v else None)
    b["gap_p99_max"] = max(col(rows, "rollout_logprob_p99") or [0]) or None
    b["gap_token_max"] = max(col(rows, "rollout_logprob_max") or [0]) or None
    b["failed_batch"] = sorted(os.path.basename(p) for p in glob.glob(d + "/failed_batch_step*"))
    if b["failed_batch"]:
        b["failed_batch_meta"] = J(f"{d}/{b['failed_batch'][0]}/meta.json")["mismatch"]
    for tag in (f"fork_step{fork['source_step']}", "final_step"):
        for s in ("gsm8k_dev", "math_dev"):
            it = items(d, tag, s)
            if it:
                k = sum(it[0].values())
                b[f"{tag}/{s}"] = dict(strict=acc(it[0]), ci=wilson(k, len(it[0])), second=acc(it[1]), relaxed=acc(it[2]), n=len(it[0]))
    B[name] = (d, fork, rows, b)
if B:
    L += ["## 2. Recovery branches", "", "| branch | status | last step | lr | GPU-h | source unchanged | identical start digest |", "|---|---|---|---|---|---|---|"]
    digs = {v[3]["state_digest"] for v in B.values()}
    L += [f"| {n} | {b['status']} | {b['last_step']} | {b['lr']} | {f(b['gpu_h'], 2)} | {b['source_unchanged']} | {len(digs) == 1} |" for n, (_, _, _, b) in B.items()]
    L += ["", "| branch | eval | GSM8K strict (95% CI) | second | relaxed | MATH strict | second | relaxed |", "|---|---|---|---|---|---|---|---|"]
    for n, (_, fork, _, b) in B.items():
        for tag in (f"fork_step{fork['source_step']}", "final_step"):
            g, m = b.get(f"{tag}/gsm8k_dev"), b.get(f"{tag}/math_dev")
            if g:
                L.append(f"| {n} | {tag} | {g['strict']:.3f} ({g['ci'][0]:.3f}-{g['ci'][1]:.3f}) | {g['second']:.3f} | {g['relaxed']:.3f} | "
                         f"{f(m and m['strict'])} | {f(m and m['second'])} | {f(m and m['relaxed'])} |")
    ks = ("reward_mean", "entropy", "grad_norm", "update_rms", "update_rel", "rollout_logprob_mad", "informative_rate", "len_mean", "truncated_rate", "format_rate")
    L += ["", "First 5 / last 5 steps (max):", "", "| branch | " + " | ".join(ks) + " | gap p99 max | gap token max | failed batch |", "|" + "---|" * (len(ks) + 4)]
    L += [f"| {n} | " + " | ".join(f"{f(b[k]['first5'], 4)}/{f(b[k]['last5'], 4)} ({f(b[k]['max'], 4)})" for k in ks)
          + f" | {f(b['gap_p99_max'], 3)} | {f(b['gap_token_max'], 2)} | {', '.join(b['failed_batch']) or '-'} |" for n, (_, _, _, b) in B.items()]
    # replay fidelity: A1 against the original's own steps 81-97 (same weights and settings at step 80; GPU kernels are not bit-deterministic)
    if "original" in out and "A1 replay lr 1e-5" in B:
        o = {r["step"]: r for r in out["original"]["rows"]}
        rp = [(r["step"], o[r["step"]], r) for r in B["A1 replay lr 1e-5"][2] if r["step"] in o]
        L += ["", "Replay fidelity, original vs A1 at the same step:", "", "| step | reward orig / A1 | len orig / A1 | gap orig / A1 |", "|---|---|---|---|"]
        L += [f"| {s} | {f(x.get('reward_mean'))}/{f(y.get('reward_mean'))} | {f(x.get('len_mean'), 0)}/{f(y.get('len_mean'), 0)} | "
              f"{f(x.get('rollout_logprob_mad'), 4)}/{f(y.get('rollout_logprob_mad'), 4)} |" for s, x, y in rp]
    cmp = []
    n1, n2 = "A1 replay lr 1e-5", "A2 lr 5e-6"
    for s in ("gsm8k_dev", "math_dev"):
        grp = []
        for label, (da, ta), (db, tb) in ((f"A2 vs A1, final", (B.get(n1, [None])[0], "final_step"), (B.get(n2, [None])[0], "final_step")),
                                          ("A1 final vs step 80", (B.get(n1, [None])[0], "fork_step80"), (B.get(n1, [None])[0], "final_step")),
                                          ("A2 final vs step 80", (B.get(n2, [None])[0], "fork_step80"), (B.get(n2, [None])[0], "final_step"))):
            ia, ib = (items(da, ta, s) if da else None), (items(db, tb, s) if db else None)
            if ia and ib:
                grp.append(dict(label=label, set=s, strict=paired(ia[0], ib[0]), relaxed_diff=acc(ib[2]) - acc(ia[2])))
        h = holm({g["label"]: g["strict"]["p"] for g in grp})
        for g in grp:
            g["p_holm"] = h[g["label"]]
        cmp += grp
    if cmp:
        L += ["", "| comparison (b minus a) | set | strict diff (95% CI) | p | Holm p | relaxed diff |", "|---|---|---|---|---|---|"]
        L += [f"| {g['label']} | {g['set']} | {100 * g['strict']['diff']:+.1f} pp [{100 * g['strict']['ci95'][0]:+.1f}, {100 * g['strict']['ci95'][1]:+.1f}] | "
              f"{g['strict']['p']:.3g} | {g['p_holm']:.3g} | {100 * g['relaxed_diff']:+.1f} |" for g in cmp]
    # registered diagnostic criteria (recovery/diagnostic_criteria.yaml): diagnostic only
    import yaml
    from recovery_criteria import evaluate
    crit = yaml.safe_load(open(os.path.join(ROOT, "recovery", "diagnostic_criteria.yaml"), encoding="utf-8"))
    if "original" in out:
        L += ["", f"Registered diagnostic criteria (version {crit['version']}, frozen {crit['frozen']}; diagnostic only):", "",
              "| branch | early warning (first, all) | reward < 0.30 | two consecutive < 0.30 | numerical failure (gate) | completed step 100, all gates passed |", "|---|---|---|---|---|---|"]
        for n, (d, fork, rows, b) in B.items():
            hist = [r for r in out["original"]["rows"] if r["step"] <= fork["source_step"]]
            dn = J(d + "/DONE.json") if os.path.exists(d + "/DONE.json") else None
            fl = J(d + "/FAILED.json") if os.path.exists(d + "/FAILED.json") else None
            ev = evaluate(hist, rows, crit, dn, fl) if rows else None
            b["criteria"] = ev
            b["criteria_sha256_recorded"] = (fork.get("criteria") or {}).get("sha256")
            if ev:
                L.append(f"| {n} | {ev['early_warning']['first']}, {ev['early_warning']['steps']} | {ev['severe_reward_deterioration']['steps'] or 'no'} | "
                         f"{ev['sustained_reward_deterioration']['hit']} | {ev['numerical_failure']['hit']} (attempted {ev['numerical_failure']['attempted_step']}) | "
                         f"{ev['successful_completion']['hit']} |")
        L += ["", "Caveat: " + crit["caveat"]]
    out["branches"] = {n: v[3] for n, v in B.items()}
    out["comparisons"] = cmp
os.makedirs(a.out)
json.dump(out, open(a.out + "/report.json", "w"), indent=1, default=str)
open(a.out + "/REPORT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
