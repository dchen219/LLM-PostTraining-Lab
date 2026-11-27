"""Trajectory and plateau analysis of the post-hoc recovery branches A1/A2 against the failed original GRPO-A_base_s2026. Reads only.

python scripts/recovery_trajectories.py --runs ~/a5_recovery --out results/final_prep/recovery_trajectories
Development sets and training rollouts only; no held-out file is opened. Writes analysis.json, REPORT_DATA.md and fig1-fig4 (PNG).
Every step-level comparison between branches is descriptive: one run per branch, autocorrelated steps, no seed replication.
"""
import argparse, json, logging, math, os, statistics as st, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
logging.disable(logging.CRITICAL)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from posttrain.common import jload  # noqa: E402
from posttrain.stats import holm, paired, wilson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R, O = os.path.expanduser(a.runs), a.out
assert not os.path.exists(O), "refusing to overwrite"
ORIG = f"{R}/grpo/A_base_s2026"
BR = {"A1": f"{R}/recovery/A_replay80_lr1e-5_s2026", "A2": f"{R}/recovery/A_recovery80_lr5e-6_s2026"}
for p in [ORIG, *BR.values()]:
    assert os.path.isdir(p), f"missing {p}"
os.makedirs(O)
BLUE, ORANGE, AQUA, RED, INK, INK2, GRID = "#257de0", "#f47230", "#00b894", "#ec4647", "#191919", "#666666", "#e6e6e6"
COL = {"original": INK2, "A1": BLUE, "A2": ORANGE}
LAB = {"original": "original (lr 1e-5)", "A1": "A1 replay (lr 1e-5)", "A2": "A2 (lr 5e-6)"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9,
                     "axes.titleweight": "bold", "figure.dpi": 150})
out, L = {}, []


def train_rows(d):
    rows = [x for x in jload(d + "/metrics.jsonl") if x["event"] == "train"]
    keep = {x["step"]: x for x in rows}
    return [keep[s] for s in sorted(keep)], len(rows) - len(keep)


T, dup = {}, {}
T["original"], dup["original"] = train_rows(ORIG)
for b, d in BR.items():
    T[b], dup[b] = train_rows(d)
out["duplicated_step_rows"] = dup
out["steps"] = {k: [v[0]["step"], v[-1]["step"], len(v)] for k, v in T.items()}

# ---------- 1. per-step metrics, steps 81-100 ----------
K = ["reward_mean", "format_rate", "truncated_rate", "entropy", "grad_norm", "update_rms", "update_rel", "update_max_abs", "param_mean_abs_delta",
     "rollout_logprob_mad", "rollout_logprob_p99", "rollout_logprob_max", "rollout_logprob_over1", "rollout_logprob_over5", "informative_rate",
     "len_mean", "len_correct", "len_incorrect", "step_s", "update_s", "rollout_s"]
seg = lambda b, lo, hi: [x for x in T[b] if lo <= x["step"] <= hi]
summ = {}
for b in ("original", "A1", "A2"):
    for lo, hi in ((81, 95), (81, 100), (96, 100)):
        rows = seg(b, lo, hi)
        if not rows:
            continue
        summ[f"{b} {lo}-{hi}"] = {k: dict(mean=st.mean(v), sd=st.stdev(v) if len(v) > 1 else 0.0, max=max(v), min=min(v))
                                  for k in K for v in [[x[k] for x in rows if x.get(k) is not None]] if v}
out["segments"] = summ
ratio = {k: st.mean(x[k] for x in seg("A2", 81, 100)) / st.mean(x[k] for x in seg("A1", 81, 100)) for k in ("update_rms", "update_l2", "update_max_abs", "param_mean_abs_delta", "grad_norm", "adam_m_l2", "adam_sqrt_v_mean")}
out["A2_over_A1_mean_ratio_81_100"] = ratio

# replay fidelity: original vs A1 on the same steps (same start state, same lr, same data order; vLLM sampling is not bitwise reproducible)
fid = []
for s in range(81, 98):
    o, r = next(x for x in T["original"] if x["step"] == s), next(x for x in T["A1"] if x["step"] == s)
    fid.append(dict(step=s, **{f"{k}_orig": o.get(k) for k in ("reward_mean", "grad_norm", "rollout_logprob_mad", "len_mean", "entropy")},
                    **{f"{k}_A1": r.get(k) for k in ("reward_mean", "grad_norm", "rollout_logprob_mad", "len_mean", "entropy")}))
out["replay_fidelity"] = fid
exact81 = all(abs(fid[0][f"{k}_orig"] - fid[0][f"{k}_A1"]) < 1e-9 for k in ("reward_mean", "len_mean"))
out["replay_step81_reward_and_length_identical"] = exact81

# ---------- 2. response characteristics from every saved rollout ----------
ans = lambda t: t.split("<answer>")[-1].replace("</answer>", "").strip()


def rollstats(d, s):
    p = f"{d}/rollouts/step{s}.jsonl"
    if not os.path.exists(p):
        return None
    x = jload(p)
    from posttrain.verify import score
    for j, y in enumerate(x):  # records from the code before V4 carry only id, answer, response, n_tokens, reward, adv, rule: derive the rest
        if "format_reward" not in y:
            sc = score(y["response"], y["answer"])
            y.update(format_reward=sc["format_reward"], answer_reward=sc["answer_reward"], verifier_error=sc["error"], group=j // 8,
                     truncated=y["n_tokens"] >= 1024)
    cor, inc = [y["n_tokens"] for y in x if y["reward"] == 1], [y["n_tokens"] for y in x if y["reward"] != 1]
    return dict(step=s, n=len(x), reward=st.mean(y["reward"] for y in x), tokens_mean=st.mean(y["n_tokens"] for y in x),
                tokens_p90=sorted(y["n_tokens"] for y in x)[int(0.9 * len(x))], tokens_correct=st.mean(cor) if cor else None,
                tokens_incorrect=st.mean(inc) if inc else None, truncated=sum(bool(y["truncated"]) for y in x), format_fail=sum(y["format_reward"] != 1 for y in x),
                format_ok_wrong=sum(y["format_reward"] == 1 and y["answer_reward"] != 1 for y in x), double_think_close=sum(y["response"].count("</think>") >= 2 for y in x),
                boxed=sum("\\boxed" in y["response"] for y in x), entropy=st.mean(y["entropy"] for y in x if y.get("entropy") is not None) if any(y.get("entropy") is not None for y in x) else None,
                gap_mean=st.mean(y["gap_mean"] for y in x if y.get("gap_mean") is not None) if any(y.get("gap_mean") is not None for y in x) else None,
                gap_rollout_max=max((y["gap_max"] for y in x if y.get("gap_max") is not None), default=None),
                rollouts_gap_over_0p05=sum((y.get("gap_mean") or 0) > 0.05 for y in x),
                distinct_responses_per_group=st.mean(len({y["response"] for y in x if y["group"] == g}) for g in {y["group"] for y in x}),
                verifier_errors=sum(bool(y["verifier_error"]) for y in x), additive_rule=sum(bool(y["rule"]) for y in x))


RS = {"original": [r for s in range(10, 98, 10) for r in [rollstats(ORIG, s)] if r]}
for b, d in BR.items():
    RS[b] = [r for s in range(81, 101) for r in [rollstats(d, s)] if r]
out["rollouts"] = RS
out["rollout_steps_saved"] = {k: [r["step"] for r in v] for k, v in RS.items()}

# ---------- 3. dev evaluations, item level ----------
import stage0_verify_audit as A  # noqa: E402


def items(p):
    x = jload(p)
    s = {y["id"]: bool(y["correct"]) for y in x}
    sec = {y["id"]: s[y["id"]] or bool(y["format"] and A.second_verifier(ans(y["response"]), str(y["answer"]))) for y in x}
    rel = {y["id"]: sec[y["id"]] or y["category"] == "format_fail_answer_right" for y in x}
    meta = dict(n=len(x), clean=st.mean(bool(y["clean_format"]) for y in x), double_think=sum(y["response"].count("</think>") >= 2 for y in x), tokens=st.mean(y["n_tokens"] for y in x), truncated=sum(y["finish"] == "length" for y in x), format=st.mean(bool(y["format"]) for y in x),
                categories={c: sum(y["category"] == c for y in x) for c in {y["category"] for y in x}})
    return s, sec, rel, meta


EV = {}  # (label, set) -> items
POINTS = [("orig@0", ORIG, "step0", 0), ("orig@25", ORIG, "step25", 25), ("orig@50", ORIG, "step50", 50), ("orig@75", ORIG, "step75", 75),
          ("fork@80", BR["A1"], "fork_step80", 80), ("A1@100", BR["A1"], "final_step", 100), ("A2@100", BR["A2"], "final_step", 100), ("A2fork@80", BR["A2"], "fork_step80", 80)]
for lab, d, tag, _ in POINTS:
    for s in ("gsm8k_dev", "math_dev"):
        EV[(lab, s)] = items(f"{d}/eval/{tag}_{s}.jsonl")
# both branches evaluated the same restored state at step 80: identical answers are a determinism check
fork_same = {s: sum(EV[("fork@80", s)][0][i] != EV[("A2fork@80", s)][0][i] for i in EV[("fork@80", s)][0]) for s in ("gsm8k_dev", "math_dev")}
out["fork80_A1_vs_A2_discordant_items"] = fork_same

# cumulative GPU-hours (2 GPUs held for the whole run): training steps plus the evaluation passes; original evaluations are timed from wall_s gaps
def gpu_h(rows, upto):
    return 2 * sum(x["step_s"] for x in rows if x["step"] <= upto) / 3600


cum = {"orig@0": 0.0, "orig@25": gpu_h(T["original"], 25), "orig@50": gpu_h(T["original"], 50), "orig@75": gpu_h(T["original"], 75), "fork@80": gpu_h(T["original"], 80)}
cum["A1@100"] = cum["fork@80"] + gpu_h(T["A1"], 100) - 0
cum["A2@100"] = cum["fork@80"] + gpu_h(T["A2"], 100)
out["training_gpu_h_at_eval_point"] = cum
devtab = []
for lab, d, tag, step in POINTS:
    if lab == "A2fork@80":
        continue
    row = dict(point=lab, step=step, train_gpu_h=cum[lab])
    for s in ("gsm8k_dev", "math_dev"):
        sc, sec, rel, meta = EV[(lab, s)]
        k = sum(sc.values())
        row[s] = dict(strict=k / meta["n"], ci95=wilson(k, meta["n"]), second=sum(sec.values()) / meta["n"], relaxed=sum(rel.values()) / meta["n"],
                      format=meta["format"], clean_format=meta["clean"], double_think=meta["double_think"], tokens=meta["tokens"], truncated=meta["truncated"], categories=meta["categories"])
    devtab.append(row)
out["dev"] = devtab

# paired steps along the trajectory (exploratory; Holm within each set over this family)
STEPS = [("orig@0", "orig@25"), ("orig@25", "orig@50"), ("orig@50", "orig@75"), ("orig@75", "fork@80"), ("fork@80", "A1@100"), ("fork@80", "A2@100"), ("A1@100", "A2@100"), ("orig@25", "A1@100"), ("orig@50", "A1@100")]
pairs = []
for s in ("gsm8k_dev", "math_dev"):
    grp = []
    for x, y in STEPS:
        r = paired(EV[(x, s)][0], EV[(y, s)][0])
        gx, gy = cum[x], cum[y]
        grp.append(dict(set=s, a=x, b=y, **r, relaxed_diff=paired(EV[(x, s)][2], EV[(y, s)][2])["diff"], gpu_h=gy - gx,
                        pp_per_gpu_h=(100 * r["diff"] / (gy - gx)) if gy > gx else None))
    h = holm({f"{g['a']}->{g['b']}": g["p"] for g in grp})
    for g in grp:
        g["p_holm"] = h[f"{g['a']}->{g['b']}"]
    pairs += grp
out["paired_trajectory"] = pairs

# item stability after step 25: how many dev items change answer between checkpoints that are nearly identical
stab = {}
for s in ("gsm8k_dev", "math_dev"):
    labs = ["orig@25", "orig@50", "orig@75", "fork@80", "A1@100", "A2@100"]
    ids = list(EV[("orig@25", s)][0])
    v = {i: [EV[(l, s)][0][i] for l in labs] for i in ids}
    stab[s] = dict(checkpoints=labs, always_right=sum(all(x) for x in v.values()), always_wrong=sum(not any(x) for x in v.values()),
                   unstable=sum(0 < sum(x) < len(labs) for x in v.values()), n=len(ids),
                   wrong_at_A1_100_also_wrong_at_25=sum((not v[i][4]) and (not v[i][0]) for i in ids), wrong_at_A1_100=sum(not v[i][4] for i in ids),
                   discordant_75_vs_80=sum(v[i][2] != v[i][3] for i in ids))
out["item_stability"] = stab

# train reward trend: OLS slope per 10 steps over steps 50-95 (original) and 81-100 (branches); naive SE ignores autocorrelation
def ols(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    res = [y - my - b * (x - mx) for x, y in zip(xs, ys)]
    se = math.sqrt(sum(r * r for r in res) / (len(xs) - 2) / sxx)
    return b, se


trend = {}
for b, lo, hi in (("original", 1, 25), ("original", 26, 50), ("original", 51, 95), ("A1", 81, 100), ("A2", 81, 100)):
    rows = seg(b, lo, hi)
    for k in ("reward_mean", "informative_rate", "entropy", "len_mean"):
        sl, se = ols([x["step"] for x in rows], [x[k] for x in rows])
        trend[f"{b} {lo}-{hi} {k}"] = dict(per10=10 * sl, se10_naive=10 * se, mean=st.mean(x[k] for x in rows))
    ac = [x["all_correct"] / x["groups"] for x in rows]
    trend[f"{b} {lo}-{hi} all_correct_group_frac"] = dict(mean=st.mean(ac))
out["trends"] = trend

json.dump(out, open(O + "/analysis.json", "w"), indent=1, default=float)

# ---------- figures ----------
def lines(ax, k, lo=70, hi=100, scale=1.0):
    for b in ("original", "A1", "A2"):
        rows = [x for x in T[b] if lo <= x["step"] <= hi and x.get(k) is not None]
        if rows:
            ax.plot([x["step"] for x in rows], [scale * x[k] for x in rows], color=COL[b], lw=1.5 if b != "original" else 1.2,
                    ls="-" if b != "original" else (0, (4, 2)), label=LAB[b])
    ax.axvline(80, color=INK2, lw=0.8, ls=(0, (2, 3)))
    ax.set_xlim(lo, hi + 0.5)


fig, axs = plt.subplots(2, 3, figsize=(11.5, 6.2), constrained_layout=True)
for ax, (k, t, sc) in zip(axs.flat, [("reward_mean", "Training reward (mean of 256)", 1), ("entropy", "Mean token entropy (nats)", 1), ("grad_norm", "Gradient norm (before clipping)", 1),
                                     ("update_rms", "Update RMS over all parameters (x1e-6)", 1e6), ("format_rate", "Format rate (training rollouts)", 1), ("len_mean", "Mean response tokens", 1)]):
    lines(ax, k, scale=sc)
    ax.set_title(t, loc="left")
axs[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
axs[1, 0].text(81, 0.2, "original: not logged\n(code before V4)", fontsize=7, color=INK2, va="bottom")
fig.suptitle("Steps 70-100: failed original (dashed) and the two branches from its step-80 state (dotted line)", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
fig.savefig(O + "/fig1_training_trajectories.png")
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)
for ax, (k, t) in zip(axs, [("rollout_logprob_mad", "Rollout log-prob gap, mean (gate 0.05)"), ("rollout_logprob_p99", "Rollout log-prob gap, p99"), ("rollout_logprob_max", "Largest single-token gap (nats)")]):
    lines(ax, k)
    ax.set_title(t, loc="left")
axs[0].axhline(0.05, color=RED, lw=1, ls="--")
axs[0].set_yscale("log")
axs[2].set_yscale("log")
axs[0].legend(frameon=False, fontsize=7.5, loc="center left")
fig.suptitle("Numerical discrepancy between vLLM sampling and the policy, steps 70-100", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
fig.savefig(O + "/fig2_logprob_gap.png")
plt.close(fig)

fig, axs = plt.subplots(1, 4, figsize=(13.5, 3.4), constrained_layout=True)
for ax, (k, t) in zip(axs, [("tokens_correct", "Tokens, correct rollouts"), ("tokens_incorrect", "Tokens, incorrect rollouts"), ("truncated", "Rollouts at 1,024 tokens (of 256)"), ("double_think_close", "Doubled </think> (of 256)")]):
    for b in ("original", "A1", "A2"):
        rows = [r for r in RS[b] if r["step"] >= 70 and r.get(k) is not None]
        ax.plot([r["step"] for r in rows], [r[k] for r in rows], color=COL[b], lw=1.5, marker="o" if b == "original" else None, ms=4, label=LAB[b])
    ax.axvline(80, color=INK2, lw=0.8, ls=(0, (2, 3)))
    ax.set_title(t, loc="left")
axs[0].legend(frameon=False, fontsize=7.5)
fig.suptitle("Response characteristics from saved rollouts (original saved every 10 steps; branches every step)", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
fig.savefig(O + "/fig3_responses.png")
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
for ax, s, t in ((axs[0], "gsm8k_dev", "GSM8K dev (n = 1,000), strict, 95% CI"), (axs[1], "math_dev", "MATH dev (n = 500), strict, 95% CI")):
    orig = [r for r in devtab if r["point"].startswith("orig") or r["point"] == "fork@80"]
    ax.errorbar([r["train_gpu_h"] for r in orig], [r[s]["strict"] for r in orig], yerr=[[r[s]["strict"] - r[s]["ci95"][0] for r in orig], [r[s]["ci95"][1] - r[s]["strict"] for r in orig]],
                color=INK2, lw=1.3, marker="o", ms=4, capsize=3, label="original, steps 0/25/50/75; 80 (restored state)")
    for b in ("A1", "A2"):
        p0, p1 = next(r for r in devtab if r["point"] == "fork@80"), next(r for r in devtab if r["point"] == f"{b}@100")
        ax.errorbar([p0["train_gpu_h"], p1["train_gpu_h"] + (0.02 if b == "A2" else 0)], [p0[s]["strict"], p1[s]["strict"]], yerr=[[0, p1[s]["strict"] - p1[s]["ci95"][0]], [0, p1[s]["ci95"][1] - p1[s]["strict"]]],
                    color=COL[b], lw=1.5, marker="s", ms=4, capsize=3, label=LAB[b] + ", step 100")
    for r in orig:
        ax.annotate(f"{100 * r[s]['strict']:.1f}", (r["train_gpu_h"], r[s]["strict"]), textcoords="offset points", xytext=(5, -11), fontsize=7, color=INK)
    ax.set_xlabel("Training GPU-hours (2 GPUs; evaluations excluded)")
    ax.set_title(t, loc="left")
axs[0].set_ylim(0.85, 0.95)
axs[0].text(0.02, 0.04, "step 0: 21.8% (below the axis)", transform=axs[0].transAxes, fontsize=7, color=INK2)
axs[0].legend(frameon=False, fontsize=7.5, loc="lower right")
fig.suptitle("Development accuracy against training compute: the plateau after step 50", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
fig.savefig(O + "/fig4_dev_vs_compute.png")
plt.close(fig)

# ---------- tables ----------
f3 = lambda v: "-" if v is None else (f"{v:.2e}" if 0 < abs(v) < 1e-3 else f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}")
L += ["# Recovery trajectories: data tables (generated)", "", f"Steps logged: {out['steps']}; duplicated step rows: {dup}", ""]
L += ["## Segment means (sd) [max]", "", "| segment | " + " | ".join(K[:16]) + " |", "|---|" + "---|" * 16]
for name, v in summ.items():
    L.append(f"| {name} | " + " | ".join(f"{f3(v[k]['mean'])} ({f3(v[k]['sd'])}) [{f3(v[k]['max'])}]" if k in v else "-" for k in K[:16]) + " |")
L += ["", "A2/A1 mean ratio over steps 81-100: " + ", ".join(f"{k} {v:.3f}" for k, v in ratio.items()), ""]
L += ["## Saved rollouts", "", "| run | step | reward | tokens | p90 | correct | incorrect | at 1,024 | format fail | double </think> | boxed | distinct/group | gap max | rollouts gap>0.05 |", "|" + "---|" * 14]
for b in ("original", "A1", "A2"):
    for r in RS[b]:
        if b == "original" and r["step"] < 70:
            continue
        L.append(f"| {b} | {r['step']} | {r['reward']:.3f} | {r['tokens_mean']:.0f} | {r['tokens_p90']} | {f3(r['tokens_correct'])} | {f3(r['tokens_incorrect'])} | {r['truncated']} | {r['format_fail']} | {r['double_think_close']} | {r['boxed']} | {r['distinct_responses_per_group']:.2f} | {f3(r['gap_rollout_max'])} | {r['rollouts_gap_over_0p05']} |")
L += ["", "## Dev evaluations", "", "| point | train GPU-h | GSM8K strict (CI) | second | relaxed | format | clean format | double </think> | tokens | trunc | MATH strict (CI) | second | relaxed | format | clean format | double </think> | tokens | trunc |", "|" + "---|" * 18]
for r in devtab:
    g, m = r["gsm8k_dev"], r["math_dev"]
    L.append(f"| {r['point']} | {r['train_gpu_h']:.2f} | {g['strict']:.3f} ({g['ci95'][0]:.3f}-{g['ci95'][1]:.3f}) | {g['second']:.3f} | {g['relaxed']:.3f} | {g['format']:.3f} | {g['clean_format']:.3f} | {g['double_think']} | {g['tokens']:.0f} | {g['truncated']} | {m['strict']:.3f} ({m['ci95'][0]:.3f}-{m['ci95'][1]:.3f}) | {m['second']:.3f} | {m['relaxed']:.3f} | {m['format']:.3f} | {m['clean_format']:.3f} | {m['double_think']} | {m['tokens']:.0f} | {m['truncated']} |")
L += ["", f"Step-80 state evaluated by both branches: discordant items {fork_same}", "", "## Paired trajectory comparisons (exploratory; Holm within set over these 9)", "",
      "| set | a -> b | a only | b only | diff (95% CI) | p | Holm p | relaxed diff | GPU-h | pp per GPU-h |", "|" + "---|" * 10]
for g in pairs:
    L.append(f"| {g['set']} | {g['a']} -> {g['b']} | {g['a_only']} | {g['b_only']} | {100 * g['diff']:+.1f} [{100 * g['ci95'][0]:+.1f}, {100 * g['ci95'][1]:+.1f}] | {g['p']:.3g} | {g['p_holm']:.3g} | {100 * g['relaxed_diff']:+.1f} | {g['gpu_h']:.2f} | {f3(g['pp_per_gpu_h'])} |")
L += ["", "## Item stability", ""] + [f"- {s}: {json.dumps(v)}" for s, v in stab.items()]
L += ["", "## Trends (per 10 steps; naive SE)", ""] + [f"- {k}: {json.dumps({kk: round(vv, 5) for kk, vv in v.items()})}" for k, v in trend.items()]
open(O + "/REPORT_DATA.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
