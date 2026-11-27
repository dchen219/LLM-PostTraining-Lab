"""Pre-recovery analysis of the failed GRPO-A_base_s2026 from its recovered logs. Reads only.

python scripts/incident_analysis.py --run ~/a5_recovery/grpo/A_base_s2026 --out results/grpo_a_incident_analysis
Writes analysis.json, REPORT_DATA.md (tables) and fig1-fig4 (PNG). Development sets only.
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
from posttrain.stats import paired, wilson  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R, O = os.path.expanduser(a.run), a.out
assert not os.path.exists(O), "refusing to overwrite"
os.makedirs(O)
BLUE, ORANGE, AQUA, RED, INK, INK2, GRID = "#257de0", "#f47230", "#00b894", "#ec4647", "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9,
                     "axes.titleweight": "bold", "figure.dpi": 150})
out = {}

# ---------- integrity ----------
M = jload(R + "/metrics.jsonl")
T = [r for r in M if r["event"] == "train"]
E = [r for r in M if r["event"] == "eval"]
steps = [r["step"] for r in T]
out["integrity"] = dict(rows=len(M), train_rows=len(T), eval_rows=len(E), duplicated_train_steps=len(steps) - len(set(steps)),
                        contiguous_1_to_97=steps == list(range(1, 98)), eval_steps=[r["step"] for r in E],
                        resumes=len(jload(R + "/resumes.jsonl")) if os.path.exists(R + "/resumes.jsonl") else 0,
                        done_json=os.path.exists(R + "/DONE.json"), weight_mismatches=sum(r["params_mismatched"] for r in T),
                        exact_check_every_step=all(r["params_checked"] > 0 for r in T), verifier_errors=sum(r["verifier_errors"] for r in T),
                        clip_frac_max=max(r["clip_frac"] for r in T), ratio_range=[min(r["ratio_min"] for r in T), max(r["ratio_max"] for r in T)],
                        lr_after_warmup=sorted({r["lr"] for r in T if r["step"] > 5}))
for r in T:
    r["grad_per_informative_group"] = r["grad_norm"] / max(1, r["groups"] - r["zero_adv"])

# ---------- earliest warning: robust z against the previous 20 steps ----------
KEYS = ["reward_mean", "format_rate", "truncated_rate", "entropy", "grad_norm", "grad_per_informative_group", "param_mean_abs_delta",
        "param_changed_frac_bf16", "len_mean", "rollout_logprob_mad", "rollout_logprob_p99", "rollout_logprob_max"]


def rz(i, k, w=20):
    prev = [T[j][k] for j in range(max(0, i - w), i)]
    med = st.median(prev)
    mad = st.median([abs(x - med) for x in prev]) * 1.4826 or 1e-12
    return (T[i][k] - med) / mad


Z = {k: {T[i]["step"]: rz(i, k) for i in range(20, len(T))} for k in KEYS}
first = {k: next((s for s, z in Z[k].items() if s >= 30 and abs(z) >= 3.5), None) for k in KEYS}
out["robust_z"] = dict(window="previous 20 steps, median and 1.4826 x MAD", threshold=3.5, first_step_at_or_after_30=first,
                       z_steps_90_97={k: {s: round(Z[k][s], 2) for s in range(90, 98)} for k in KEYS},
                       run_max_before_96={k: max(r[k] for r in T[:95]) for k in KEYS}, run_min_before_96={k: min(r[k] for r in T[:95]) for k in KEYS})

# ---------- is the step-97 gap explained by its entropy? log(gap) ~ log(entropy), steps 20-95 ----------
fit = [(math.log(r["entropy"]), math.log(r["rollout_logprob_mad"])) for r in T if 20 <= r["step"] <= 95]
xs, ys = zip(*fit)
mx, my = st.mean(xs), st.mean(ys)
b = sum((x - mx) * (y - my) for x, y in fit) / sum((x - mx) ** 2 for x in xs)
c0 = my - b * mx
res = [y - (c0 + b * x) for x, y in fit]
sd = st.pstdev(res)
pred = lambda e: math.exp(c0 + b * math.log(e))
out["gap_vs_entropy"] = dict(steps="20-95", slope=b, intercept=c0, r=st.correlation(xs, ys), residual_sd_log=sd,
                             step96=dict(entropy=T[95]["entropy"], gap=T[95]["rollout_logprob_mad"], predicted=pred(T[95]["entropy"]),
                                         z=(math.log(T[95]["rollout_logprob_mad"]) - math.log(pred(T[95]["entropy"]))) / sd),
                             step97=dict(entropy=T[96]["entropy"], gap=T[96]["rollout_logprob_mad"], predicted=pred(T[96]["entropy"]),
                                         z=(math.log(T[96]["rollout_logprob_mad"]) - math.log(pred(T[96]["entropy"]))) / sd),
                             steps_1_10_gap_per_entropy=st.mean(r["rollout_logprob_mad"] / r["entropy"] for r in T[:10]),
                             steps_60_95_gap_per_entropy=st.mean(r["rollout_logprob_mad"] / r["entropy"] for r in T[59:95]))

# ---------- saved rollouts (every 10 steps) ----------
ro = []
for s in range(10, 98, 10):
    p = f"{R}/rollouts/step{s}.jsonl"
    if not os.path.exists(p):
        continue
    x = jload(p)
    grp = {}
    for y in x:
        grp.setdefault(y["id"], []).append(y["response"])

    def rep4(t):
        w = t.split()
        g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
        return 1 - len(set(g)) / len(g) if g else 0.0
    ro.append(dict(step=s, n=len(x), reward=st.mean(y["reward"] for y in x), distinct_per_group=st.mean(len(set(v)) for v in grp.values()),
                   at_1024_tokens=sum(y["n_tokens"] >= 1024 for y in x), repeated_4gram_frac=st.mean(rep4(y["response"]) for y in x),
                   double_think_close=sum(y["response"].count("</think>") >= 2 for y in x), additive_rule=sum(bool(y["rule"]) for y in x)))
out["rollouts"] = ro

# rewards re-scored with the frozen verifier, and checked with math-verify
from posttrain.verify import score  # noqa: E402
import stage0_verify_audit as A  # noqa: E402
ans = lambda t: t.split("<answer>")[-1].replace("</answer>", "").strip()
vr = dict(rollouts=0, rescore_differs=0, rejected_with_tag=0, rejected_second_accepts=0, accepted=0, accepted_second_rejects=0, disputed_examples=[])
for s in range(10, 98, 10):
    p = f"{R}/rollouts/step{s}.jsonl"
    if not os.path.exists(p):
        continue
    for y in jload(p):
        vr["rollouts"] += 1
        vr["rescore_differs"] += score(y["response"], y["answer"])["reward"] != y["reward"]
        if y["reward"] == 0 and "<answer>" in y["response"]:
            vr["rejected_with_tag"] += 1
            vr["rejected_second_accepts"] += bool(A.second_verifier(ans(y["response"]), str(y["answer"])))
        if y["reward"] == 1:
            vr["accepted"] += 1
            if not A.second_verifier(ans(y["response"]), str(y["answer"])):
                vr["accepted_second_rejects"] += 1
                vr["disputed_examples"].append(dict(step=s, answer=ans(y["response"])[:60], gold=str(y["answer"])[:60]))
out["verifier"] = vr

# ---------- intermediate dev evaluations ----------
ev, items = [], {}
for t in (0, 25, 50, 75):
    for s, n in (("gsm8k_dev", 1000), ("math_dev", 500)):
        x = jload(f"{R}/eval/step{t}_{s}.jsonl")
        stc = {y["id"]: bool(y["correct"]) for y in x}
        sec = {y["id"]: stc[y["id"]] or bool(y["format"] and A.second_verifier(ans(y["response"]), str(y["answer"]))) for y in x}
        rel = {y["id"]: sec[y["id"]] or y["category"] == "format_fail_answer_right" for y in x}
        items[(t, s)] = (stc, rel)
        k = sum(stc.values())
        ev.append(dict(step=t, set=s, n=len(x), strict=k / len(x), ci=wilson(k, len(x)), second=sum(sec.values()) / len(x), relaxed=sum(rel.values()) / len(x),
                       format=sum(y["format"] for y in x) / len(x), clean_format=sum(y["clean_format"] for y in x) / len(x),
                       truncated=sum(y["finish"] == "length" for y in x), verifier_errors=sum(bool(y["error"]) for y in x),
                       double_think_close=sum(y["response"].count("</think>") >= 2 for y in x), mean_tokens=st.mean(y["n_tokens"] for y in x)))
out["evals"] = ev
out["eval_paired"] = [dict(set=s, a=x, b=y, strict=paired(items[(x, s)][0], items[(y, s)][0]), relaxed=paired(items[(x, s)][1], items[(y, s)][1]))
                      for s in ("gsm8k_dev", "math_dev") for x, y in ((0, 25), (25, 50), (50, 75))]
json.dump(out, open(O + "/analysis.json", "w"), indent=1, default=str)

# ---------- figures ----------
S = [r["step"] for r in T]


def mark(ax):
    ax.axvline(80, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.axvspan(95.5, 97.5, color=RED, alpha=0.10, lw=0)


panels = [("reward_mean", "Training reward (mean of 256 rollouts)", False), ("entropy", "Entropy per token (log scale)", True),
          ("grad_norm", "Gradient norm before clipping (clip at 1.0)", False), ("param_mean_abs_delta", "Mean |parameter change| per update (sampled)", False),
          ("rollout_logprob_mad", "Mean |vLLM - PyTorch| log-prob (log)", True), ("rollout_logprob_p99", "p99 token gap (log)", True),
          ("rollout_logprob_max", "Largest token gap (log)", True), ("format_rate", "Format rate", False), ("truncated_rate", "Truncated at 1,024 tokens", False)]
fig, axs = plt.subplots(3, 3, figsize=(11, 7.2), sharex=True)
for ax, (k, title, lg) in zip(axs.flat, panels):
    ax.plot(S, [r[k] for r in T], color=BLUE, lw=1.4)
    ax.plot([97], [T[-1][k]], "o", color=RED, ms=5, mec="white", mew=1)
    if lg:
        ax.set_yscale("log")
    if k == "rollout_logprob_mad":
        ax.axhline(0.05, color=RED, lw=1, ls="--")
        ax.text(2, 0.053, "gate 0.05 (step 98: 0.079)", color=INK2, fontsize=7.5, va="bottom")
    ax.set_title(title, loc="left")
    mark(ax)
for ax in axs[-1]:
    ax.set_xlabel("training step")
fig.suptitle("GRPO-A_base_s2026: all 97 completed steps (dashed line: saved state at step 80; shaded: steps 96-97)", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(O + "/fig1_trajectory.png")
plt.close(fig)

zk = [("reward_mean", "reward"), ("entropy", "entropy"), ("grad_norm", "grad norm"), ("grad_per_informative_group", "grad norm / informative group"),
      ("param_mean_abs_delta", "|parameter change|"), ("rollout_logprob_mad", "mean log-prob gap"), ("rollout_logprob_p99", "p99 gap"),
      ("rollout_logprob_max", "max token gap"), ("truncated_rate", "truncation")]
fig, axs = plt.subplots(3, 3, figsize=(11, 6.6), sharex=True)
zs = list(range(70, 98))
for ax, (k, lab) in zip(axs.flat, zk):
    v = [Z[k][s] for s in zs]
    ax.bar(zs, v, color=[RED if abs(x) >= 3.5 else BLUE for x in v], width=0.75)
    ax.axhline(3.5, color=INK2, lw=0.8, ls="--")
    ax.axhline(-3.5, color=INK2, lw=0.8, ls="--")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title(lab, loc="left")
    top = max(abs(x) for x in v)
    ax.set_ylim(-max(6, min(top, 40)) * 1.05, max(6, min(top, 40)) * 1.05)
    for s, x in zip(zs, v):
        if abs(x) > 40:
            ax.text(s, 40, f"{x:.0f}", ha="center", va="bottom", fontsize=7, color=INK)
for ax in axs[-1]:
    ax.set_xlabel("training step")
fig.suptitle("Robust z-score against the previous 20 steps (red: |z| >= 3.5), steps 70-97", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(O + "/fig2_warning_scores.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
grp = [(range(1, 20), "steps 1-19", "#9c9c9c"), (range(20, 96), "steps 20-95 (fit)", BLUE), (range(96, 97), "step 96", ORANGE), (range(97, 98), "step 97", RED)]
for rg, lab, col in grp:
    pts = [(T[s - 1]["entropy"], T[s - 1]["rollout_logprob_mad"]) for s in rg]
    ax.scatter(*zip(*pts), s=22 if len(pts) > 1 else 46, color=col, edgecolor="white", linewidth=0.8, label=lab, zorder=3)
e = [0.02, 0.8]
ax.plot(e, [pred(x) for x in e], color=INK2, lw=1, ls="--", label=f"fit: gap ∝ entropy^{b:.2f}")
ax.axhline(0.05, color=RED, lw=1, ls=":")
ax.text(0.75, 0.052, "gate 0.05", color=INK2, fontsize=7.5, ha="right")
ax.set_xscale("log"); ax.set_yscale("log")
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
for axis, ticks in ((ax.xaxis, [0.02, 0.05, 0.1, 0.2, 0.5]), (ax.yaxis, [0.003, 0.005, 0.01, 0.02, 0.04])):
    axis.set_major_locator(FixedLocator(ticks)); axis.set_minor_locator(NullLocator()); axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("entropy per token"); ax.set_ylabel("mean |vLLM - PyTorch| log-prob")
ax.set_title("Rollout log-prob gap against entropy, one point per step", loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(O + "/fig3_gap_vs_entropy.png")
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, s, title in ((axs[0], "gsm8k_dev", "GSM8K dev (n = 1,000)"), (axs[1], "math_dev", "MATH dev (n = 500)")):
    rows = [x for x in ev if x["set"] == s]
    st_ = [x["step"] for x in rows]
    ax.errorbar(st_, [x["strict"] for x in rows], yerr=[[x["strict"] - x["ci"][0] for x in rows], [x["ci"][1] - x["strict"] for x in rows]],
                color=BLUE, lw=1.6, marker="o", ms=5, capsize=3, label="strict (95% CI)")
    ax.plot(st_, [x["relaxed"] for x in rows], color=ORANGE, lw=1.4, marker="s", ms=4, label="relaxed (answer right, any format)")
    for x in rows:
        ax.annotate(f"{100 * x['strict']:.1f}", (x["step"], x["strict"]), textcoords="offset points", xytext=(8, 6), ha="left", fontsize=7.5, color=INK)
    ax.axvspan(95.5, 97.5, color=RED, alpha=0.10, lw=0)
    ax.set_xlim(-5, 100); ax.set_ylim(0, 1); ax.set_xticks([0, 25, 50, 75, 97])
    ax.set_title(title, loc="left"); ax.set_xlabel("training step (no evaluation after 75; collapse at 96-97)")
axs[0].legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(O + "/fig4_dev_evals.png")
plt.close(fig)
print(json.dumps({k: out[k] for k in ("integrity", "gap_vs_entropy")}, indent=1, default=str))
print(json.dumps(out["robust_z"]["first_step_at_or_after_30"], indent=1))
print(json.dumps(out["robust_z"]["z_steps_90_97"], indent=0))
