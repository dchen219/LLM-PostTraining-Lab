"""Figures for the research analysis. Reads only the tables written by scripts/research_analysis.py.

PYTHONPATH=~/a5_plotlib python scripts/research_figures.py --data results/research_analysis
Writes figures/figNN_*.png and .pdf, and figures/data/figNN_*.csv with exactly the values each figure plots.
"""
import argparse, csv, json, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
a = ap.parse_args()
D = a.data
F = os.path.join(D, "figures")
os.makedirs(F + "/data", exist_ok=True)


def rcsv(p):
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if v not in ("", None) and k not in ("set", "run", "name", "label", "checkpoint", "family", "cls", "prompt", "window", "category",
                                                                    "kind", "transition", "stage", "data_source", "reading", "a", "b", "metric", "applies") else v
            except ValueError:
                pass
    return rows


R = rcsv(D + "/consolidated_results.csv")
C = rcsv(D + "/statistical_comparisons.csv")
AN = json.load(open(D + "/analysis.json", encoding="utf-8"))
BLUE, ORANGE, AQUA, RED, PURPLE, INK, INK2, GRID, LIGHT = "#257de0", "#f47230", "#00b894", "#ec4647", "#8e6cdb", "#191919", "#666666", "#e6e6e6", "#b3b3b3"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "figure.dpi": 150, "savefig.dpi": 200, "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42})
SETN = {"gsm8k_test_final": "GSM8K test (n = 1,319)", "math_val_final": "MATH val (n = 5,000)", "gsm8k_dev": "GSM8K dev (n = 1,000)", "math_dev": "MATH dev (n = 500)"}
ORDER = ["BASE:r1_zero", "BASE:question_only", "SFT-reverified_n128_s0:r1_zero", "SFT-reverified_n512_s0:r1_zero", "SFT-exact_n512_s0:r1_zero", "SFT-reverified_n2048_s0:r1_zero",
         "SFT-reverified_sizecontrol_s0:r1_zero", "SFT-exact_full_s0:r1_zero", "SFT-reverified_full_s0:r1_zero", "GRPO-B_sft512_s2026:r1_zero",
         "GRPO-A-replay80-lr1e-5-s2026:r1_zero", "GRPO-A-recovery80-lr5e-6-s2026:r1_zero"]
NICE = {"BASE:r1_zero": "Base, r1_zero prompt", "BASE:question_only": "Base, question-only prompt", "SFT-reverified_n128_s0:r1_zero": "SFT re-verified, 128",
        "SFT-reverified_n512_s0:r1_zero": "SFT re-verified, 512", "SFT-exact_n512_s0:r1_zero": "SFT exact-match, 512", "SFT-reverified_n2048_s0:r1_zero": "SFT re-verified, 2,048",
        "SFT-reverified_sizecontrol_s0:r1_zero": "SFT re-verified, 3,420 (size control)", "SFT-exact_full_s0:r1_zero": "SFT exact-match, 3,420 (full)",
        "SFT-reverified_full_s0:r1_zero": "SFT re-verified, 4,423 (full)", "GRPO-B_sft512_s2026:r1_zero": "GRPO-B: SFT re-verified 512 + GRPO",
        "GRPO-A-replay80-lr1e-5-s2026:r1_zero": "A1: GRPO-A step 80 + 20 (post-hoc)", "GRPO-A-recovery80-lr5e-6-s2026:r1_zero": "A2: GRPO-A step 80 + 20, lr 5e-6 (post-hoc)"}


def kind(k):
    return "base" if k.startswith("BASE") else "sft" if k.startswith("SFT") else "grpo" if k.startswith("GRPO-B") else "posthoc"


KC = {"base": INK2, "sft": BLUE, "grpo": ORANGE, "posthoc": AQUA}
get = {(r["checkpoint"] + ":" + r["prompt"], r["set"]): r for r in R}


def save(fig, name, rows):
    fig.savefig(f"{F}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{F}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    if rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(f"{F}/data/{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


def note(fig, txt):
    fig.text(0.0, -0.02, txt, fontsize=6.8, color=INK2, ha="left", va="top", wrap=True)


# ---------------------------------------------------------------- fig01 overall
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, constrained_layout=True)
rows = []
for ax, s in zip(axs, ("gsm8k_test_final", "math_val_final")):
    y = list(range(len(ORDER)))[::-1]
    for yi, k in zip(y, ORDER):
        r = get[(k, s)]
        c = KC[kind(k)]
        ax.plot([100 * r["strict_ci_lo"], 100 * r["strict_ci_hi"]], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(100 * r["strict"], yi, "o", ms=6, color=c, mfc="white" if kind(k) == "posthoc" else c, mew=1.6, zorder=3)
        ax.text(100 * r["strict_ci_hi"] + 1.2, yi, f"{100 * r['strict']:.1f}", va="center", fontsize=7, color=INK)
        rows.append(dict(set=s, checkpoint=k, label=NICE[k], strict_pct=100 * r["strict"], ci_lo=100 * r["strict_ci_lo"], ci_hi=100 * r["strict_ci_hi"], cls=r["cls"]))
    qo = 100 * get[("BASE:question_only", s)]["strict"]
    ax.axvline(qo, color=INK2, ls="--", lw=0.8)
    ax.text(qo, len(ORDER) - 0.35, " base, question-only", fontsize=6.8, color=INK2, va="bottom")
    ax.set_title(SETN[s] + ": strict accuracy, 95% Wilson CI")
    ax.set_xlim(0, 100)
    ax.set_xlabel("strict accuracy (%)")
    ax.set_ylim(-0.7, len(ORDER) + 0.2)
axs[0].set_yticks(list(range(len(ORDER)))[::-1], [NICE[k] for k in ORDER])
h = [Line2D([], [], marker="o", color=KC[k], mfc=KC[k] if k != "posthoc" else "white", lw=0, label=l) for k, l in
     (("base", "base model"), ("sft", "SFT (pre-registered)"), ("grpo", "GRPO-B (pre-registered)"), ("posthoc", "post-hoc branch of failed GRPO-A"))]
axs[1].legend(handles=h, loc="lower left", fontsize=7)
fig.suptitle("Held-out accuracy of every frozen checkpoint (one training seed each; pre-registered GRPO-A FAILED and has no checkpoint)", x=0.0, ha="left", fontsize=10, fontweight="bold")
note(fig, "Intervals cover sampling of test questions only, not training-seed variance. A1/A2 are post-hoc diagnostics and do not replace GRPO-A_base_s2026.")
save(fig, "fig01_overall_performance", rows)

# ---------------------------------------------------------------- fig02 data quality (paired differences, item CI and seed-inflated interval)
SS = {(r["name"], r["set"]): r for r in rcsv(D + "/data/seed_sensitivity.csv")}
DQ = [("C1 data quality, full", "C1: exact full -> re-verified full", "confirmatory"), ("C2 data quality, matched 512", "C2: exact 512 -> re-verified 512", "confirmatory"),
      ("composition at matched size", "exact full -> re-verified size control (3,420)", "descriptive"), ("volume", "re-verified 3,420 -> 4,423", "descriptive"),
      ("Exact-match scaling 512 -> 3,420", "exact 512 -> exact 3,420", "post-hoc (unregistered)")]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, constrained_layout=True)
rows = []
for ax, s in zip(axs, ("gsm8k_test_final", "math_val_final")):
    for i, (nm, lab, fam) in enumerate(DQ):
        g = next(x for x in C if x["name"] == nm and x["set"] == s and x["metric"] == "strict")
        ss = SS.get((nm, s))
        y = len(DQ) - 1 - i
        if ss and ss.get("seed_sd_contrast_pp") not in ("", None):
            ax.plot([ss["approx_ci_lo_pp"], ss["approx_ci_hi_pp"]], [y, y], color=LIGHT, lw=6, solid_capstyle="butt", zorder=1)
        c = BLUE if fam == "confirmatory" else INK2 if fam == "descriptive" else PURPLE
        ax.plot([g["ci_lo_pp"], g["ci_hi_pp"]], [y, y], color=c, lw=2, zorder=2)
        ax.plot(g["diff_pp"], y, "o", color=c, ms=6, zorder=3)
        ph = g["p_holm"] if g.get("p_holm") not in ("", None) else g["p"]
        ax.text(max(g["ci_hi_pp"], ss["approx_ci_hi_pp"] if ss else -99) + 0.6, y, f"{g['diff_pp']:+.1f} pp, {'Holm ' if g.get('p_holm') not in ('', None) else ''}p = {ph:.2g}",
                fontsize=6.8, va="center", color=INK)
        rows.append(dict(set=s, comparison=lab, family=fam, diff_pp=g["diff_pp"], item_ci_lo=g["ci_lo_pp"], item_ci_hi=g["ci_hi_pp"], p=g["p"], p_holm=g.get("p_holm"),
                         seed_inflated_lo=ss["approx_ci_lo_pp"] if ss else None, seed_inflated_hi=ss["approx_ci_hi_pp"] if ss else None))
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim(-20, 14)
    ax.set_title(SETN[s] + ": paired difference (b minus a)")
    ax.set_xlabel("accuracy difference (percentage points)")
axs[0].set_yticks(range(len(DQ))[::-1], [f"{lab}\n[{fam}]" for _, lab, fam in DQ], fontsize=7)
h = [Line2D([], [], color=BLUE, lw=2, marker="o", label="item-sampling 95% CI (McNemar family)"),
     Line2D([], [], color=LIGHT, lw=6, label="heuristic interval adding training-seed variance\n(estimated from ONE dev seed pair, 1 df)")]
fig.legend(handles=h, loc="outside lower right", fontsize=6.8, ncol=2)
fig.suptitle("Training-data quality: exact-match vs re-verified SFT data", x=0.0, ha="left", fontsize=10, fontweight="bold")
note(fig, "Seed band: SD of a two-arm contrast estimated from SFT-reverified_n512 seed 0 vs seed 100 on dev (GSM8K 4.0 pp, MATH 2.5 pp). It is a sensitivity "
          "illustration, not a test; with one degree of freedom the true seed SD could be several times smaller or larger.")
save(fig, "fig02_data_quality", rows)

# ---------------------------------------------------------------- fig03 SFT scaling
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), constrained_layout=True)
rows = []
curve = [("SFT-reverified_n128_s0", 128), ("SFT-reverified_n512_s0", 512), ("SFT-reverified_n2048_s0", 2048), ("SFT-reverified_full_s0", 4423)]
for ax, s in zip(axs, ("gsm8k_test_final", "math_val_final")):
    xs = [n for _, n in curve]
    v = [get[(c + ":r1_zero", s)] for c, _ in curve]
    ax.fill_between(xs, [100 * x["strict_ci_lo"] for x in v], [100 * x["strict_ci_hi"] for x in v], color=BLUE, alpha=0.15, lw=0)
    ax.plot(xs, [100 * x["strict"] for x in v], "-o", color=BLUE, ms=5, lw=2, label="re-verified (held-out)")
    ax.plot(xs, [100 * x["dev_strict"] for x in v], "--o", color=BLUE, ms=3.5, lw=1, alpha=0.6, mfc="white", label="re-verified (dev set)")
    for c, n in curve:
        rows.append(dict(set=s, run=c, n=n, source="reverified", strict=100 * get[(c + ":r1_zero", s)]["strict"], lo=100 * get[(c + ":r1_zero", s)]["strict_ci_lo"],
                         hi=100 * get[(c + ":r1_zero", s)]["strict_ci_hi"], dev=100 * get[(c + ":r1_zero", s)]["dev_strict"]))
    for c, n, mk, lab in (("SFT-exact_n512_s0", 512, "s", "exact-match"), ("SFT-exact_full_s0", 3420, "s", None), ("SFT-reverified_sizecontrol_s0", 3420, "D", "re-verified size control")):
        r = get[(c + ":r1_zero", s)]
        xx = n * (1.04 if mk == "D" else 0.96)
        ax.errorbar([xx], [100 * r["strict"]], yerr=[[100 * (r["strict"] - r["strict_ci_lo"])], [100 * (r["strict_ci_hi"] - r["strict"])]], fmt=mk, color=ORANGE if mk == "s" else PURPLE,
                    ms=6, mfc="white", mew=1.6, capsize=2.5, lw=1.2, label=lab)
        rows.append(dict(set=s, run=c, n=n, source="exact" if "exact" in c else "reverified size control", strict=100 * r["strict"], lo=100 * r["strict_ci_lo"], hi=100 * r["strict_ci_hi"],
                         dev=100 * r["dev_strict"]))
    ax.plot([512, 3420], [100 * get[("SFT-exact_n512_s0:r1_zero", s)]["strict"], 100 * get[("SFT-exact_full_s0:r1_zero", s)]["strict"]], color=ORANGE, lw=1, ls=":")
    for k, ls, lab in (("BASE:question_only", "--", "base, question-only"), ("BASE:r1_zero", ":", "base, r1_zero")):
        ax.axhline(100 * get[(k, s)]["strict"], color=INK2, ls=ls, lw=0.9)
        ax.text(118, 100 * get[(k, s)]["strict"] + (0.8 if k == "BASE:r1_zero" else -3.2), lab, fontsize=6.8, color=INK2, ha="left")
    ax.set_xscale("log")
    ax.set_xticks([128, 512, 2048, 4423], ["128", "512", "2,048", "4,423"])
    ax.minorticks_off()
    ax.set_xlabel("SFT training examples (log scale; lr 1e-5, 2 epochs, seed 0)")
    ax.set_ylabel("strict accuracy (%)")
    ax.set_title(SETN[s].split(" (")[0] + ": accuracy vs SFT data size")
    ax.set_ylim(0 if s.startswith("gsm8k") else 0, 95)
axs[0].legend(loc="center right", fontsize=7)
fig.suptitle("SFT data scaling: most of the gain arrives by 512 examples; no detectable gain beyond 2,048", x=0.0, ha="left", fontsize=10, fontweight="bold")
note(fig, "Shaded band and error bars: 95% Wilson CI (questions only). More data also means more optimizer steps (fixed 2 epochs). One seed per point.")
save(fig, "fig03_sft_scaling", rows)

# ---------------------------------------------------------------- fig04 GRPO learning curves (dev) + held-out at step 100
CV = rcsv(D + "/data/grpo_dev_curves.csv")
GC = {"GRPO-B (sequential)": (ORANGE, "-", "o"), "GRPO-A (direct, FAILED at 98)": (INK2, "-", "o"), "A1 (post-hoc, lr 1e-5)": (AQUA, "-", "s"), "A2 (post-hoc, lr 5e-6)": (PURPLE, "-", "D")}
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.1, 1.1, 0.9]))
rows = []
for ax, s in zip(axs[:2], ("gsm8k_dev", "math_dev")):
    for run, (c, ls, mk) in GC.items():
        pts = sorted([r for r in CV if r["run"] == run and r["set"] == s], key=lambda r: r["step"])
        if run.startswith("A1") or run.startswith("A2"):
            a80 = [r for r in CV if r["run"] == "GRPO-A (direct, FAILED at 98)" and r["set"] == s]
        ax.fill_between([p["step"] for p in pts], [100 * p["ci_lo"] for p in pts], [100 * p["ci_hi"] for p in pts], color=c, alpha=0.12, lw=0)
        ax.plot([p["step"] for p in pts], [100 * p["strict"] for p in pts], ls=ls, marker=mk, color=c, ms=4.5, lw=1.8, mfc="white" if run[0] == "A" and run[1] in "12" else c, label=run)
        rows += [dict(set=s, run=run, step=p["step"], strict=100 * p["strict"], lo=100 * p["ci_lo"], hi=100 * p["ci_hi"], cumulative_gpu_h=p["cumulative_gpu_h"]) for p in pts]
    for run, c in (("GRPO tune lr 1e-5", LIGHT),):
        pass
    ax.axvline(98, color=RED, lw=0.8, ls="--")
    ax.text(97, 2 if s == "gsm8k_dev" else 2, "GRPO-A rejected\nat step 98", color=RED, fontsize=6.5, ha="right", va="bottom")
    ax.set_xlabel("GRPO step (256 rollouts per step)")
    ax.set_ylabel("strict accuracy (%)")
    ax.set_title(SETN[s] + ": during training")
    ax.set_ylim(0, 100)
axs[0].legend(loc="center right", fontsize=7)
ax = axs[2]
pts = [("SFT-reverified_n512_s0:r1_zero", "SFT parent (step 0 of B)", BLUE), ("GRPO-B_sft512_s2026:r1_zero", "GRPO-B, step 100", ORANGE),
       ("GRPO-A-replay80-lr1e-5-s2026:r1_zero", "A1, step 100 (post-hoc)", AQUA), ("GRPO-A-recovery80-lr5e-6-s2026:r1_zero", "A2, step 100 (post-hoc)", PURPLE)]
for j, s in enumerate(("gsm8k_test_final", "math_val_final")):
    for i, (k, lab, c) in enumerate(pts):
        r = get[(k, s)]
        x = j * 5 + i
        ax.errorbar([x], [100 * r["strict"]], yerr=[[100 * (r["strict"] - r["strict_ci_lo"])], [100 * (r["strict_ci_hi"] - r["strict"])]], fmt="o", color=c, ms=6, capsize=2.5,
                    mfc="white" if "post-hoc" in lab else c, mew=1.5, label=lab if j == 0 else None)
        rows.append(dict(set=s, run=lab, step=100, strict=100 * r["strict"], lo=100 * r["strict_ci_lo"], hi=100 * r["strict_ci_hi"], cumulative_gpu_h=r["cumulative_gpu_h"]))
ax.set_xticks([1.5, 6.5], ["GSM8K test", "MATH val"])
ax.set_ylim(55, 92)
ax.set_title("Held-out accuracy at the final step")
ax.legend(loc="lower left", fontsize=6.8)
fig.suptitle("GRPO learning curves: dev gains arrive in the first 25-50 steps; direct (A) and sequential (B) converge on GSM8K, B is ahead on MATH",
             x=0.0, ha="left", fontsize=10, fontweight="bold")
note(fig, "Left panels: development sets (used during training; no checkpoint was chosen from them). GRPO-A has no step-100 evaluation: it failed. A1/A2 branch from its "
          "saved step-80 state and are post-hoc. Bands: 95% Wilson CI.")
save(fig, "fig04_grpo_learning_curves", rows)

# ---------------------------------------------------------------- fig05 collapse and recovery
DY = rcsv(D + "/data/grpo_dynamics.csv")
fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.2), constrained_layout=True)
rows = []
SER = [("GRPO-A (direct, FAILED at 98)", INK2, "original GRPO-A (failed)"), ("A1 (post-hoc, lr 1e-5)", AQUA, "A1: replay from 80, lr 1e-5"),
       ("A2 (post-hoc, lr 5e-6)", PURPLE, "A2: from 80, lr 5e-6"), ("GRPO-B (sequential)", ORANGE, "GRPO-B (reference)")]
for ax, (k, ttl, logy) in zip(axs.flat, [("rollout_logprob_mad", "Rollout log-prob gap (mean |vLLM - policy|), gate 0.05", True), ("reward_mean", "Training reward (mean over 256 rollouts)", False),
                                         ("entropy", "Mean token entropy (nats)", True), ("grad_norm", "Gradient norm", True)]):
    for run, c, lab in SER:
        pts = sorted([r for r in DY if r["run"] == run and r.get(k) not in ("", None) and r.get("rejected_by_gate") in ("", None)], key=lambda r: r["step"])
        ax.plot([p["step"] for p in pts], [p[k] for p in pts], color=c, lw=1.2 if run.startswith("GRPO-B") else 1.5, alpha=0.55 if run.startswith("GRPO-B") else 1, label=lab)
        rows += [dict(panel=k, run=run, step=p["step"], value=p[k]) for p in pts]
    if k == "rollout_logprob_mad":
        rej = next(r for r in DY if r.get("rejected_by_gate") not in ("", None))
        ax.plot(rej["step"], rej[k], "X", color=RED, ms=9, zorder=5, label="step 98 rejected (0.079)")
        ax.axhline(0.05, color=RED, ls="--", lw=0.9)
        rows.append(dict(panel=k, run="GRPO-A rejected step", step=rej["step"], value=rej[k]))
    ax.axvspan(80, 100, color=GRID, alpha=0.5, lw=0)
    if logy:
        ax.set_yscale("log")
    ax.set_title(ttl)
    ax.set_xlabel("GRPO step")
axs[0, 0].legend(fontsize=6.8, loc="upper left", bbox_to_anchor=(0.0, 0.88))
fig.suptitle("GRPO-A optimization collapse at steps 96-98 and the two post-hoc branches from step 80", x=0.0, ha="left", fontsize=10, fontweight="bold")
note(fig, "The original run's step 97 shows reward 0.05, entropy 0.41 and gradient norm 0.21 before the gate rejected step 98 (no update made). Both branches replay from the "
          "same saved step-80 state (grey band: steps 81-100); neither reproduces the collapse within 20 steps. One seed: the collapse's frequency is unknown.")
save(fig, "fig05_collapse_recovery", rows)

# ---------------------------------------------------------------- fig06 GRPO training dynamics
WN = rcsv(D + "/data/grpo_compute_windows.csv")


def roll(v, w=5):
    return [sum(v[max(0, i - w + 1):i + 1]) / len(v[max(0, i - w + 1):i + 1]) for i in range(len(v))]


fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.2), constrained_layout=True)
rows = []
ax = axs[0, 0]
for run, c in (("GRPO-B (sequential)", ORANGE), ("GRPO-A (direct, FAILED at 98)", INK2)):
    pts = sorted([r for r in DY if r["run"] == run and r.get("zero_adv_frac") not in ("", None)], key=lambda r: r["step"])
    ax.plot([p["step"] for p in pts], [100 * p["zero_adv_frac"] for p in pts], color=c, lw=0.6, alpha=0.35)
    ax.plot([p["step"] for p in pts], roll([100 * p["zero_adv_frac"] for p in pts]), color=c, lw=2, label=run + " (5-step mean)")
    rows += [dict(panel="zero_adv_pct", run=run, step=p["step"], value=100 * p["zero_adv_frac"]) for p in pts]
ax.axhline(40, color=RED, lw=0.8, ls="--")
ax.text(2, 3, "dashed: 40% zero-advantage = informative rate 0.6 (pre-registered trigger below it)", fontsize=6.5, color=RED)
ax.set_ylim(0, 100)
ax.set_title("Zero-advantage groups (% of 32 groups of 8)")
ax.set_xlabel("GRPO step")
ax.legend(fontsize=6.8, loc="lower right")
ax = axs[0, 1]
pts = sorted([r for r in DY if r["run"] == "GRPO-B (sequential)" and r.get("zero_adv_frac") not in ("", None)], key=lambda r: r["step"])
ax.plot([p["step"] for p in pts], roll([100 * p["all_correct_frac"] for p in pts]), color=AQUA, lw=2, label="all 8 correct")
ax.plot([p["step"] for p in pts], roll([100 * p["all_incorrect_frac"] for p in pts]), color=RED, lw=2, label="all 8 incorrect")
rows += [dict(panel="B_all_correct_pct", run="GRPO-B", step=p["step"], value=100 * p["all_correct_frac"]) for p in pts]
rows += [dict(panel="B_all_incorrect_pct", run="GRPO-B", step=p["step"], value=100 * p["all_incorrect_frac"]) for p in pts]
ax.set_ylim(0, 100)
ax.set_title("GRPO-B: why groups carry no signal (5-step mean)")
ax.set_xlabel("GRPO step")
ax.legend(fontsize=7)
ax = axs[1, 0]
for run, c in (("GRPO-B (sequential)", ORANGE), ("GRPO-A (direct, FAILED at 98)", INK2)):
    pts = sorted([r for r in DY if r["run"] == run and r.get("len_correct") not in ("", None)], key=lambda r: r["step"])
    ax.plot([p["step"] for p in pts], roll([p["len_correct"] for p in pts]), color=c, lw=2, label=f"{run.split(' (')[0]}: correct rollouts")
    pi = [p for p in pts if p.get("len_incorrect") not in ("", None)]
    ax.plot([p["step"] for p in pi], roll([p["len_incorrect"] for p in pi]), color=c, lw=1.3, ls="--", label=f"{run.split(' (')[0]}: incorrect rollouts")
    rows += [dict(panel="len_correct", run=run, step=p["step"], value=p["len_correct"]) for p in pts]
    rows += [dict(panel="len_incorrect", run=run, step=p["step"], value=p["len_incorrect"]) for p in pts if p.get("len_incorrect") not in ("", None)]
ax.set_title("Rollout length, tokens (5-step mean)")
ax.set_xlabel("GRPO step")
ax.legend(fontsize=6.8)
ax = axs[1, 1]
wb = [w for w in WN if w["run"].startswith("GRPO-B")]
x = range(len(wb))
ax.bar([i - 0.2 for i in x], [w["gsm8k_dev_gain_pp"] for w in wb], width=0.38, color=BLUE, label="GSM8K dev gain (pp)")
ax.bar([i + 0.2 for i in x], [w["math_dev_gain_pp"] for w in wb], width=0.38, color=ORANGE, label="MATH dev gain (pp)")
for i, w in zip(x, wb):
    ax.text(i, max(w["gsm8k_dev_gain_pp"], w["math_dev_gain_pp"]) + 0.6, f"zero-adv {100 * w['zero_adv_frac']:.0f}%\n{w['gpu_h']:.1f} GPU-h", ha="center", fontsize=6.5, color=INK2)
    rows.append(dict(panel="B_window_gain", run="GRPO-B", window=w["window"], gsm8k_dev_gain_pp=w["gsm8k_dev_gain_pp"], math_dev_gain_pp=w["math_dev_gain_pp"],
                     zero_adv_frac=w["zero_adv_frac"], gpu_h=w["gpu_h"]))
ax.axhline(0, color=INK, lw=0.8)
ax.set_xticks(list(x), [f"steps {w['window']}" for w in wb])
ax.set_ylim(-5, 21)
ax.set_title("GRPO-B: dev gain per 25-step window")
ax.legend(fontsize=7, loc="upper right")
fig.suptitle("GRPO training dynamics: signal-free groups dominate after step 25 while responses lengthen", x=0.0, ha="left", fontsize=10, fontweight="bold")
note(fig, "Window gains are differences between single dev evaluations (GSM8K dev n = 1,000, MATH dev n = 500); after steps 0-25 none is significant (see RESEARCH_FINDINGS.md).")
save(fig, "fig06_grpo_dynamics", rows)

# ---------------------------------------------------------------- fig07 strict vs relaxed accuracy and format compliance
CATS = [("correct", "correct (strict)", AQUA), ("format_ok_answer_wrong", "formatted, wrong answer", INK2), ("format_fail_answer_right", "misformatted, right answer", ORANGE),
        ("format_fail_answer_wrong", "misformatted, wrong answer", PURPLE), ("format_fail_truncated", "truncated at 1,024 tokens", RED)]
fig, axs = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.15]))
rows = []
for j, s in enumerate(("gsm8k_test_final", "math_val_final")):
    ax = axs[j, 0]
    for yi, k in zip(range(len(ORDER))[::-1], ORDER):
        r = get[(k, s)]
        ax.plot([100 * r["strict"], 100 * r["relaxed"]], [yi, yi], color=LIGHT, lw=2.5, zorder=1)
        ax.plot(100 * r["strict"], yi, "o", color=BLUE, ms=5, zorder=3)
        ax.plot(100 * r["second"], yi, "|", color=INK, ms=8, mew=1.5, zorder=3)
        ax.plot(100 * r["relaxed"], yi, "o", color=ORANGE, ms=5, mfc="white", mew=1.5, zorder=3)
        ax.text(101, yi, f"fmt {100 * r['format']:.0f}%", fontsize=6.5, va="center", color=INK2)
        rows.append(dict(set=s, checkpoint=k, strict=100 * r["strict"], second=100 * r["second"], relaxed=100 * r["relaxed"], format=100 * r["format"],
                         truncated=100 * r["truncated_rate"], **{f"pct_{c}": 100 * (r.get(f"cat_{c}") or 0) / r["n"] for c, _, _ in CATS}))
    ax.set_yticks(range(len(ORDER))[::-1], [NICE[k] for k in ORDER], fontsize=7)
    ax.set_xlim(0, 110)
    ax.set_xticks(range(0, 101, 20))
    ax.set_title(SETN[s].split(" (")[0] + ": strict, second-verifier, relaxed")
    ax.set_xlabel("accuracy (%); right column: format-compliance rate")
    if j == 0:
        ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, lw=0, label="strict (frozen metric)"), Line2D([], [], marker="|", color=INK, lw=0, mew=1.5, ms=8, label="+ second verifier"),
                           Line2D([], [], marker="o", color=ORANGE, mfc="white", lw=0, label="relaxed (+ misformatted right answers)")], fontsize=6.5, loc="lower left")
    ax = axs[j, 1]
    for yi, k in zip(range(len(ORDER))[::-1], ORDER):
        r = get[(k, s)]
        left = 0
        for c, lab, col in CATS:
            v = 100 * (r.get(f"cat_{c}") or 0) / r["n"]
            ax.barh(yi, v, left=left, color=col, height=0.72, edgecolor="white", linewidth=0.6, label=lab if yi == len(ORDER) - 1 else None)
            left += v
    ax.set_yticks(range(len(ORDER))[::-1], [])
    ax.set_xlim(0, 100)
    ax.set_title(SETN[s].split(" (")[0] + ": outcome of every response")
    ax.set_xlabel("share of items (%)")
    if j == 0:
        ax.legend(fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.suptitle("Formatting vs reasoning: the base model's r1_zero deficit is mostly protocol; trained models' remaining errors are mostly well-formatted wrong answers",
             x=0.0, ha="left", fontsize=10, fontweight="bold")
note(fig, "The question-only prompt requires no tags, so its format rate is defined as 100%; its outcome categories still record whether an answer could be extracted. Second verifier: math-verify, applied only to formatted r1_zero answers.")
save(fig, "fig07_strict_relaxed_format", rows)

# ---------------------------------------------------------------- fig08 efficiency
EF = rcsv(D + "/data/efficiency.csv")
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1, 0.85]))
rows = []
OFF = {"SFT-reverified_n128_s0": (4, -8), "SFT-reverified_n512_s0": (5, -9), "SFT-exact_n512_s0": (-6, 0), "SFT-reverified_n2048_s0": (5, -3), "SFT-reverified_sizecontrol_s0": (5, -8),
       "SFT-exact_full_s0": (5, 3), "SFT-reverified_full_s0": (5, -3), "GRPO-B_sft512_s2026": (-6, 6), "GRPO-A-replay80-lr1e-5-s2026": (-6, -11), "GRPO-A-recovery80-lr5e-6-s2026": (-6, 4)}
OFFM = {"SFT-reverified_n128_s0": (4, -8), "SFT-reverified_n512_s0": (5, -10), "SFT-exact_n512_s0": (-6, 0), "SFT-reverified_n2048_s0": (0, 9), "SFT-reverified_sizecontrol_s0": (5, -10),
        "SFT-exact_full_s0": (6, 4), "SFT-reverified_full_s0": (5, -3), "GRPO-B_sft512_s2026": (-6, 6), "GRPO-A-replay80-lr1e-5-s2026": (-6, -11), "GRPO-A-recovery80-lr5e-6-s2026": (-6, 4)}
for ax, s in zip(axs[:2], ("gsm8k_test_final", "math_val_final")):
    for e in [x for x in EF if x["set"] == s]:
        k = e["checkpoint"] + ":r1_zero"
        c = KC[kind(k)]
        ax.plot(e["cumulative_gpu_h"], e["gain_vs_base_qo_pp"], "o", color=c, ms=6, mfc="white" if kind(k) == "posthoc" else c, mew=1.5, zorder=3)
        dx, dy = (OFFM if s.startswith("math") else OFF).get(e["checkpoint"], (5, 0))
        ax.annotate(NICE[k].replace("SFT ", "").replace(" (post-hoc)", "*").split(":")[0] if kind(k) != "sft" else NICE[k].replace("SFT ", ""), (e["cumulative_gpu_h"], e["gain_vs_base_qo_pp"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=6.3, color=INK, ha="right" if dx < 0 else "left")
        rows.append(dict(set=s, checkpoint=e["checkpoint"], cumulative_gpu_h=e["cumulative_gpu_h"], gain_vs_base_qo_pp=e["gain_vs_base_qo_pp"], gain_vs_base_r1_pp=e["gain_vs_base_r1_pp"]))
    p = next(x for x in EF if x["set"] == s and x["checkpoint"] == "SFT-reverified_n512_s0")
    b = next(x for x in EF if x["set"] == s and x["checkpoint"] == "GRPO-B_sft512_s2026")
    ax.annotate("", (b["cumulative_gpu_h"], b["gain_vs_base_qo_pp"]), (p["cumulative_gpu_h"], p["gain_vs_base_qo_pp"]), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1, ls="--"))
    ax.axhline(0, color=INK2, ls="--", lw=0.8)
    ax.set_xlabel("cumulative training GPU-hours (A6000; 2 GPUs x wall-clock)")
    ax.set_xlim(-2.2, 8.4)
    ax.set_ylabel("gain over base question-only prompt (pp)")
    ax.set_title(SETN[s].split(" (")[0] + ": gain vs training compute")
ax = axs[2]
for run, c in (("GRPO-B (sequential)", ORANGE), ("GRPO-A (direct, FAILED at 98)", INK2)):
    ws = [w for w in WN if w["run"] == run]
    for s, ls in (("gsm8k_dev", "-"), ("math_dev", "--")):
        ax.plot(range(len(ws)), [w[f"{s}_pp_per_gpu_h"] for w in ws], marker="o", ls=ls, color=c, ms=4, lw=1.5, label=f"{run.split(' (')[0]}, {'GSM8K' if s.startswith('gsm8k') else 'MATH'} dev")
        rows += [dict(set=s, run=run, window=w["window"], pp_per_gpu_h=w[f"{s}_pp_per_gpu_h"], gpu_h=w["gpu_h"]) for w in ws]
ax.axhline(0, color=INK, lw=0.8)
ax.set_yscale("symlog", linthresh=2)
ax.set_xticks(range(4), ["0-25", "25-50", "50-75", "75-100"])
ax.set_xlabel("GRPO steps (window)")
ax.set_ylabel("dev accuracy gain per GPU-hour (pp, symlog)")
ax.set_title("Marginal returns to GRPO compute (dev)")
ax.legend(fontsize=6.3)
fig.suptitle("Computational efficiency: SFT on 512 examples buys most of the gain per GPU-hour; GRPO returns collapse after ~25 steps",
             x=0.0, ha="left", fontsize=10, fontweight="bold")
note(fig, "GRPO-B includes its SFT parent (0.44 GPU-h); A1/A2 (*, post-hoc) include GRPO-A steps 1-80. The failed GRPO-A itself consumed about "
          f"{AN['failed_A']['gpu_h_total']:.1f} GPU-h. Evaluation compute is excluded.")
save(fig, "fig08_efficiency", rows)

# ---------------------------------------------------------------- fig09 cross-benchmark and dev-to-final agreement
DF = rcsv(D + "/data/dev_vs_final_accuracy.csv")
fig, axs = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
rows = []
ax = axs[0]
tr = [k for k in ORDER if not k.startswith("BASE")]
for k in tr:
    g, m = get[(k, "gsm8k_test_final")], get[(k, "math_val_final")]
    c = KC[kind(k)]
    ax.plot(100 * g["strict"], 100 * m["strict"], "o", color=c, mfc="white" if kind(k) == "posthoc" else c, mew=1.5, ms=6)
    rows.append(dict(panel="gsm8k_vs_math", checkpoint=k, x=100 * g["strict"], y=100 * m["strict"]))
ax.set_xlabel("GSM8K test strict accuracy (%)")
ax.set_ylabel("MATH val strict accuracy (%)")
xb = AN["cross_benchmark"]
ax.set_title(f"GSM8K vs MATH, held-out (Spearman rho = {xb['spearman_gsm8k_math']:.2f})")
for ax, s, ds in ((axs[1], "gsm8k_test_final", "GSM8K"), (axs[2], "math_val_final", "MATH")):
    pts = [r for r in DF if r["set"] == s]
    for r in pts:
        k = r["checkpoint"] + ":r1_zero"
        c = KC[kind(k)]
        ax.plot(100 * r["dev"], 100 * r["final"], "o", color=c, mfc="white" if kind(k) == "posthoc" else c, mew=1.5, ms=6)
        rows.append(dict(panel=f"dev_vs_final_{s}", checkpoint=k, x=100 * r["dev"], y=100 * r["final"]))
    ax.plot([0, 100], [0, 100], color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel(f"{ds} dev strict accuracy (%)")
    ax.set_ylabel(f"{ds} held-out strict accuracy (%)")
    ax.set_title(f"{ds}: dev vs held-out (Spearman rho = {xb['spearman_dev_final_' + s]:.2f})")
    ax.text(3, 93, "dotted: y = x", fontsize=6.5, color=INK2)
fig.suptitle("Generalization: model rankings agree across benchmarks and between dev and held-out sets; absolute levels do not", x=0.0, ha="left", fontsize=10, fontweight="bold")
note(fig, "MATH dev (500 problems the teacher model left untraced) is much harder than MATH val, so every point sits far above y = x. GSM8K dev comes from GSM8K train and is "
          "easier than the test split. Points: base r1_zero (grey) and the 10 trained checkpoints.")
save(fig, "fig09_cross_benchmark", rows)

# ---------------------------------------------------------------- fig10 where the gains come from
TR = rcsv(D + "/data/gain_transitions.csv")
TL = [("Base r1_zero -> base question-only", "base: r1_zero -> question-only prompt"), ("Base r1_zero -> SFT re-verified n512", "base r1_zero -> SFT re-verified 512"),
      ("Base question-only -> SFT re-verified n512", "base question-only -> SFT re-verified 512"), ("C3: SFT parent -> GRPO-B", "SFT parent -> GRPO-B (C3)"),
      ("SFT exact n512 -> SFT re-verified n512 (C2)", "SFT exact 512 -> re-verified 512 (C2)")]
CC2 = [("format_ok_answer_wrong", "was formatted, wrong", INK2), ("format_fail_answer_right", "was misformatted, right", ORANGE), ("format_fail_answer_wrong", "was misformatted, wrong", PURPLE),
       ("format_fail_truncated", "was truncated", RED)]
fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.8), sharey=True, constrained_layout=True)
rows = []
for ax, s in zip(axs, ("gsm8k_test_final", "math_val_final")):
    for yi, (t, lab) in zip(range(len(TL))[::-1], TL):
        g = {r["category"]: r["count"] for r in TR if r["transition"] == t and r["set"] == s and r["kind"].startswith("gained")}
        l_ = sum(r["count"] for r in TR if r["transition"] == t and r["set"] == s and r["kind"].startswith("lost"))
        left = 0
        for c, cl, col in CC2:
            v = g.get(c, 0)
            ax.barh(yi, v, left=left, color=col, height=0.62, edgecolor="white", linewidth=0.6, label=cl if yi == len(TL) - 1 else None)
            left += v
        ax.barh(yi, -l_, color=LIGHT, height=0.62, edgecolor="white", linewidth=0.6, label="lost (correct -> wrong)" if yi == len(TL) - 1 else None)
        ax.text(left + 15, yi, f"+{int(left)} / -{int(l_)}", fontsize=6.8, va="center")
        rows.append(dict(set=s, transition=lab, lost=l_, **{f"gained_{c}": g.get(c, 0) for c, _, _ in CC2}))
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_title(SETN[s] + ": items that changed (count)")
    ax.set_xlabel("items lost (left) and gained (right), gained split by the earlier outcome")
axs[0].set_yticks(range(len(TL))[::-1], [lab for _, lab in TL], fontsize=7)
axs[1].legend(fontsize=6.5, loc="lower right")
xmax = max(ax.get_xlim()[1] for ax in axs)
fig.suptitle("Where the gains come from: GRPO converts well-formatted wrong answers; the prompt change and SFT mostly repair the output protocol",
             x=0.0, ha="left", fontsize=10, fontweight="bold")
note(fig, "Discordant items of each paired comparison. For C3 on GSM8K, 234 of the 237 newly-correct items were already well-formatted in the SFT parent.")
save(fig, "fig10_gain_decomposition", rows)
print("figures written:", sorted(f for f in os.listdir(F) if f.endswith(".png")))
