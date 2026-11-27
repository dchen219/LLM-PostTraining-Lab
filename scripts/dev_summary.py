"""Consolidated development-set results for every Tier A run, the failed original and the post-hoc branches. Reads only. Run on the laptop.

python scripts/dev_summary.py --runs ~/a5_sessions --runs ~/a5_recovery --out results/final_prep/dev_summary
Development sets only (GSM8K dev n = 1,000 from GSM8K train; MATH dev n = 500, excluded from every training pool). No held-out file is opened.
Every comparison is labelled: "registered (dev)" = a pre-registered final comparison computed on dev data (NOT the confirmatory test, which is
the held-out evaluation); "exploratory" = any other contrast; "post-hoc" = involves the recovery branches; "diagnostic" = the failed run.
One seed per arm: McNemar p-values cover item sampling only, not training-seed variance.
"""
import argparse, json, logging, os, statistics as st, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
logging.disable(logging.CRITICAL)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from posttrain.common import jload  # noqa: E402
from posttrain.stats import holm, paired, wilson  # noqa: E402
import stage0_verify_audit as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", action="append", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
TREES = [os.path.expanduser(t) for t in a.runs]
O = a.out
assert not os.path.exists(O), "refusing to overwrite"
J = lambda p: json.load(open(p, encoding="utf-8"))
BR = {"A1": "recovery/A_replay80_lr1e-5_s2026", "A2": "recovery/A_recovery80_lr5e-6_s2026"}


def find(rel):
    for t in TREES:
        if os.path.isdir(os.path.join(t, rel)):
            return os.path.join(t, rel)
    raise SystemExit(f"missing {rel}")


R = {}  # short name -> folder
for n in ("tune_lr1e-5", "tune_lr2e-5", "tune_lr5e-5", "exact_full_s0", "reverified_full_s0", "reverified_sizecontrol_s0", "exact_n512_s0", "reverified_n512_s0",
          "reverified_n128_s0", "reverified_n2048_s0"):
    R["SFT-" + n] = find("sft/" + n)
for n in ("tune_lr2e-6", "tune_lr5e-6", "tune_lr1e-5", "A_base_s2026", "B_sft512_s2026"):
    R["GRPO-" + n] = find("grpo/" + n)
for k, v in BR.items():
    R[k] = find(v)
os.makedirs(O)
BLUE, ORANGE, AQUA, RED, PURPLE, INK, INK2, GRID = "#257de0", "#f47230", "#00b894", "#ec4647", "#8e6cdb", "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9,
                     "axes.titleweight": "bold", "figure.dpi": 150})
ans = lambda t: t.split("<answer>")[-1].replace("</answer>", "").strip()
CACHE = {}


def items(run, tag, s):
    k = (run, tag, s)
    if k not in CACHE:
        p = f"{R[run]}/eval/{tag}_{s}.jsonl"
        if not os.path.exists(p):
            CACHE[k] = None
        else:
            x = jload(p)
            sc = {y["id"]: bool(y["correct"]) for y in x}
            sec = {y["id"]: sc[y["id"]] or bool(y["format"] and A.second_verifier(ans(y["response"]), str(y["answer"]))) for y in x}
            rel = {y["id"]: sec[y["id"]] or y["category"] == "format_fail_answer_right" for y in x}
            CACHE[k] = dict(strict=sc, second=sec, relaxed=rel, n=len(x), format=st.mean(bool(y["format"]) for y in x), tokens=st.mean(y["n_tokens"] for y in x),
                            truncated=sum(y["finish"] == "length" for y in x), double_think=sum(y["response"].count("</think>") >= 2 for y in x))
    return CACHE[k]


def acc(run, tag, s, kind="strict"):
    e = items(run, tag, s)
    return None if e is None else sum(e[kind].values()) / e["n"]


def train(run):
    rows = [x for x in jload(R[run] + "/metrics.jsonl") if x["event"] == "train"]
    keep = {x["step"]: x for x in rows}  # last row per step (see the audit for the one replayed block)
    return [keep[s] for s in sorted(keep)]


def gpu_h(run):
    return 2 * J(R[run] + "/DONE.json")["wall_s"] / 3600 if os.path.exists(R[run] + "/DONE.json") else 2 * sum(x["step_s"] for x in train(run)) / 3600


out = dict(runs={})
SETS = ("gsm8k_dev", "math_dev")
FINAL = {r: ("step75" if r == "GRPO-A_base_s2026" else "final_step") for r in R}
for r in R:
    row = dict(lr=float(J(R[r] + "/run.json")["config"]["train"]["lr"]), gpu_h=gpu_h(r), steps=len(train(r)) if r not in BR else 20, point=FINAL[r])
    for s in SETS:
        e = items(r, FINAL[r], s)
        k = sum(e["strict"].values())
        row[s] = dict(strict=k / e["n"], ci95=wilson(k, e["n"]), second=acc(r, FINAL[r], s, "second"), relaxed=acc(r, FINAL[r], s, "relaxed"), format=e["format"], tokens=e["tokens"],
                      truncated=e["truncated"], double_think=e["double_think"])
    out["runs"][r] = row
# base model, r1_zero prompt: the step-0 evaluation inside GRPO-A (same engine and settings as the GRPO dev evaluations)
out["base_r1_zero"] = {s: dict(strict=acc("GRPO-A_base_s2026", "step0", s), ci95=wilson(sum(items("GRPO-A_base_s2026", "step0", s)["strict"].values()), items("GRPO-A_base_s2026", "step0", s)["n"]),
                               relaxed=acc("GRPO-A_base_s2026", "step0", s, "relaxed"), format=items("GRPO-A_base_s2026", "step0", s)["format"]) for s in SETS}

# ---------- paired comparisons ----------
FAM = {
    "registered (dev)": [("C1 data quality, full: exact -> re-verified", "SFT-exact_full_s0:final_step", "SFT-reverified_full_s0:final_step"),
                         ("C2 data quality, n512: exact -> re-verified", "SFT-exact_n512_s0:final_step", "SFT-reverified_n512_s0:final_step"),
                         ("C3 pathway B vs its SFT parent", "SFT-reverified_n512_s0:final_step", "GRPO-B_sft512_s2026:final_step"),
                         ("C4 pathway A vs base", "GRPO-A_base_s2026:step0", None)],
    "exploratory": [("composition at matched size: exact_full -> sizecontrol", "SFT-exact_full_s0:final_step", "SFT-reverified_sizecontrol_s0:final_step"),
                    ("volume: sizecontrol -> reverified_full", "SFT-reverified_sizecontrol_s0:final_step", "SFT-reverified_full_s0:final_step"),
                    ("scaling 128 -> 512", "SFT-reverified_n128_s0:final_step", "SFT-reverified_n512_s0:final_step"),
                    ("scaling 512 -> 2048", "SFT-reverified_n512_s0:final_step", "SFT-reverified_n2048_s0:final_step"),
                    ("scaling 2048 -> full (4,423)", "SFT-reverified_n2048_s0:final_step", "SFT-reverified_full_s0:final_step"),
                    ("training seed only: reverified_n512, lr 1e-5, seed 0 -> seed 100 (tuning run)", "SFT-reverified_n512_s0:final_step", "SFT-tune_lr1e-5:final_step"),
                    ("SFT lr 1e-5 -> 2e-5 (tuning)", "SFT-tune_lr1e-5:final_step", "SFT-tune_lr2e-5:final_step"),
                    ("SFT lr 1e-5 -> 5e-5 (tuning)", "SFT-tune_lr1e-5:final_step", "SFT-tune_lr5e-5:final_step"),
                    ("GRPO lr 1e-5 -> 5e-6 (tuning, 30 steps)", "GRPO-tune_lr1e-5:final_step", "GRPO-tune_lr5e-6:final_step"),
                    ("GRPO lr 1e-5 -> 2e-6 (tuning, 30 steps)", "GRPO-tune_lr1e-5:final_step", "GRPO-tune_lr2e-6:final_step"),
                    ("SFT parent vs base (r1_zero)", "GRPO-A_base_s2026:step0", "SFT-reverified_n512_s0:final_step"),
                    ("pathway B vs base (r1_zero)", "GRPO-A_base_s2026:step0", "GRPO-B_sft512_s2026:final_step"),
                    ("direct vs sequential at step 75: A (failed run, diagnostic) -> B", "GRPO-A_base_s2026:step75", "GRPO-B_sft512_s2026:step75")],
    "post-hoc": [("direct (A1, step 100) -> sequential (B, step 100)", "A1:final_step", "GRPO-B_sft512_s2026:final_step"),
                 ("direct (A2, step 100) -> sequential (B, step 100)", "A2:final_step", "GRPO-B_sft512_s2026:final_step")],
    "GRPO compute (exploratory)": [("B 0 -> 25", "GRPO-B_sft512_s2026:step0", "GRPO-B_sft512_s2026:step25"), ("B 25 -> 50", "GRPO-B_sft512_s2026:step25", "GRPO-B_sft512_s2026:step50"),
                                   ("B 50 -> 75", "GRPO-B_sft512_s2026:step50", "GRPO-B_sft512_s2026:step75"), ("B 75 -> 100", "GRPO-B_sft512_s2026:step75", "GRPO-B_sft512_s2026:final_step"),
                                   ("A 0 -> 25", "GRPO-A_base_s2026:step0", "GRPO-A_base_s2026:step25"), ("A 25 -> 50", "GRPO-A_base_s2026:step25", "GRPO-A_base_s2026:step50"),
                                   ("A 50 -> 75", "GRPO-A_base_s2026:step50", "GRPO-A_base_s2026:step75"), ("B 25 -> 100", "GRPO-B_sft512_s2026:step25", "GRPO-B_sft512_s2026:final_step"),
                                   ("B 50 -> 100", "GRPO-B_sft512_s2026:step50", "GRPO-B_sft512_s2026:final_step")],
}
key = lambda x: tuple(x.split(":"))


def cum_gpu_h(run, step):
    return 2 * sum(x["step_s"] for x in train(run) if x["step"] <= step) / 3600


comps = []
for fam, lst in FAM.items():
    for s in SETS:
        grp = []
        for name, x, y in lst:
            row = dict(family=fam, name=name, set=s, a=x, b=y)
            if y is None:
                row.update(unavailable="GRPO-A_base_s2026 failed at step 98: no final step (its step 75 is diagnostic)", p=1.0)
            else:
                ix, iy = items(*key(x), s), items(*key(y), s)
                row.update(paired(ix["strict"], iy["strict"]), relaxed_diff=paired(ix["relaxed"], iy["relaxed"])["diff"], second_diff=paired(ix["second"], iy["second"])["diff"])
                if fam.startswith("GRPO compute"):
                    st_ = lambda t: 100 if t == "final_step" else int(t.replace("step", ""))
                    rr = key(x)[0]
                    row["gpu_h"] = cum_gpu_h(rr, st_(key(y)[1])) - cum_gpu_h(rr, st_(key(x)[1]))
                    row["pp_per_gpu_h"] = 100 * row["diff"] / row["gpu_h"] if row["gpu_h"] else None
            grp.append(row)
        h = holm({g["name"]: g["p"] for g in grp})
        for g in grp:
            g["p_holm"] = h[g["name"]]
            g["holm_m"] = len(grp)
        comps += grp
out["comparisons"] = comps
# re-evaluation of identical weights (SFT parent final vs GRPO-B step 0): engine noise floor
noise = {}
for s in SETS:
    a_, b_ = items("SFT-reverified_n512_s0", "final_step", s)["strict"], items("GRPO-B_sft512_s2026", "step0", s)["strict"]
    noise[s] = dict(discordant=sum(a_[i] != b_[i] for i in a_), n=len(a_), acc_sft=sum(a_.values()) / len(a_), acc_b0=sum(b_.values()) / len(b_))
out["same_weights_reevaluation"] = noise

# ---------- GRPO training dynamics ----------
GR = ["GRPO-tune_lr2e-6", "GRPO-tune_lr5e-6", "GRPO-tune_lr1e-5", "GRPO-A_base_s2026", "GRPO-B_sft512_s2026"]
dyn = {}
for r in GR + list(BR):
    t = train(r)
    seg = lambda lo, hi: [x for x in t if lo <= x["step"] <= hi]
    f = lambda rows, k: st.mean(x[k] for x in rows if x.get(k) is not None) if rows else None
    out_r = {}
    for lab, rows in (("first10", t[:10]), ("last10", t[-10:]), ("all", t)):
        out_r[lab] = dict(reward=f(rows, "reward_mean"), zero_adv_frac=sum(x["zero_adv"] for x in rows) / sum(x["groups"] for x in rows),
                          all_correct_frac=sum(x["all_correct"] for x in rows) / sum(x["groups"] for x in rows), all_incorrect_frac=sum(x["all_incorrect"] for x in rows) / sum(x["groups"] for x in rows),
                          entropy=f(rows, "entropy"), len_mean=f(rows, "len_mean"), len_correct=f(rows, "len_correct"), len_incorrect=f(rows, "len_incorrect"), truncated=f(rows, "truncated_rate"),
                          format=f(rows, "format_rate"), grad_norm=f(rows, "grad_norm"), grad_norm_max=max(x["grad_norm"] for x in rows), gap_mean=f(rows, "rollout_logprob_mad"),
                          gap_max_step=max(x["rollout_logprob_mad"] for x in rows), gap_p99_max=max(x["rollout_logprob_p99"] for x in rows), token_gap_max=max(x["rollout_logprob_max"] for x in rows),
                          mismatched_params=sum(x.get("params_mismatched", 0) for x in rows), verifier_errors=sum(x.get("verifier_errors", 0) for x in rows),
                          step_s=f(rows, "step_s"), rollout_s=f(rows, "rollout_s"), update_s=f(rows, "update_s"))
    dyn[r] = out_r
out["grpo_dynamics"] = dyn
# zero-advantage share vs dev accuracy along GRPO-B and GRPO-A (25-step windows ending at each evaluation)
win = {}
for r in ("GRPO-B_sft512_s2026", "GRPO-A_base_s2026"):
    t = train(r)
    win[r] = [dict(window=f"{lo}-{hi}", zero_adv_frac=sum(x["zero_adv"] for x in t if lo <= x["step"] <= hi) / max(1, sum(x["groups"] for x in t if lo <= x["step"] <= hi)),
                   entropy=st.mean(x["entropy"] for x in t if lo <= x["step"] <= hi), reward=st.mean(x["reward_mean"] for x in t if lo <= x["step"] <= hi),
                   len_mean=st.mean(x["len_mean"] for x in t if lo <= x["step"] <= hi), gap=st.mean(x["rollout_logprob_mad"] for x in t if lo <= x["step"] <= hi))
              for lo, hi in ((1, 25), (26, 50), (51, 75), (76, 100)) if any(lo <= x["step"] <= hi for x in t)]
out["grpo_windows"] = win
# SFT dynamics
sft = {}
for r in [k for k in R if k.startswith("SFT")]:
    t = train(r)
    sft[r] = dict(steps=len(t), loss_first=t[0]["loss"], loss_last5=st.mean(x["loss"] for x in t[-5:]), grad_norm_max=max(x["grad_norm"] for x in t), tok_per_s=st.mean(x["tok_per_s"] for x in t),
                  gpu_h=gpu_h(r), n=J(R[r] + "/data.json")["n"])
out["sft_dynamics"] = sft
json.dump(out, open(O + "/summary.json", "w"), indent=1, default=float)

# ---------- figures ----------
def ci_err(vals):
    return [[v["strict"] - v["ci95"][0] for v in vals], [v["ci95"][1] - v["strict"] for v in vals]]


fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
ax = axs[0]
lrs = ["SFT-tune_lr1e-5", "SFT-tune_lr2e-5", "SFT-tune_lr5e-5"]
for s, c, lab in (("gsm8k_dev", BLUE, "GSM8K dev"), ("math_dev", ORANGE, "MATH dev")):
    v = [out["runs"][r][s] for r in lrs]
    ax.errorbar([1e-5, 2e-5, 5e-5], [x["strict"] for x in v], yerr=ci_err(v), color=c, marker="o", ms=4, capsize=3, lw=1.5, label=lab)
ax.set_xscale("log")
ax.set_xticks([1e-5, 2e-5, 5e-5], ["1e-5", "2e-5", "5e-5"])
ax.minorticks_off()
ax.set_xlabel("SFT learning rate (tuning runs, reverified n512, seed 100)")
ax.set_title("SFT learning-rate sensitivity", loc="left")
ax.legend(frameon=False, fontsize=7.5)
ax = axs[1]
curve = [("SFT-reverified_n128_s0", 128), ("SFT-reverified_n512_s0", 512), ("SFT-reverified_n2048_s0", 2048), ("SFT-reverified_full_s0", 4423)]
for s, c, mk in (("gsm8k_dev", BLUE, "o"), ("math_dev", ORANGE, "o")):
    v = [out["runs"][r][s] for r, _ in curve]
    ax.errorbar([n for _, n in curve], [x["strict"] for x in v], yerr=ci_err(v), color=c, marker=mk, ms=4, capsize=3, lw=1.5, label=f"re-verified, {'GSM8K' if s == 'gsm8k_dev' else 'MATH'}")
    ex = [("SFT-exact_n512_s0", 512), ("SFT-exact_full_s0", 3420)]
    v = [out["runs"][r][s] for r, _ in ex]
    ax.errorbar([n for _, n in ex], [x["strict"] for x in v], yerr=ci_err(v), color=c, marker="s", mfc="white", ms=5, capsize=3, lw=0, elinewidth=1.2, label=f"exact-match, {'GSM8K' if s == 'gsm8k_dev' else 'MATH'}")
    v = out["runs"]["SFT-reverified_sizecontrol_s0"][s]
    ax.errorbar([3420], [v["strict"]], yerr=ci_err([v]), color=c, marker="D", mfc="white", ms=5, capsize=3, lw=0, elinewidth=1.2, label=f"re-verified size control (3,420)" if s == "gsm8k_dev" else None)
for s, c in (("gsm8k_dev", BLUE), ("math_dev", ORANGE)):
    ax.axhline(out["base_r1_zero"][s]["strict"], color=c, lw=0.8, ls=":")
ax.text(140, out["base_r1_zero"]["gsm8k_dev"]["strict"] + 0.012, "base model, r1_zero (dotted)", fontsize=7, color=INK2)
for s, c in (("gsm8k_dev", BLUE), ("math_dev", ORANGE)):
    v = out["runs"]["SFT-tune_lr1e-5"][s]
    ax.errorbar([560], [v["strict"]], yerr=ci_err([v]), color=c, marker="^", mfc="white", ms=5, capsize=3, lw=0, elinewidth=1.2, label="re-verified n512, training seed 100" if s == "gsm8k_dev" else None)
ax.set_xscale("log")
ax.set_xticks([128, 512, 2048, 4423], ["128", "512", "2,048", "4,423"])
ax.minorticks_off()
ax.set_xlabel("SFT examples (lr 1e-5, 2 epochs, seed 0)")
ax.set_title("SFT data size and source (strict, 95% CI)", loc="left")
ax.legend(frameon=False, fontsize=6.5, loc="center right")
ax = axs[2]
col = {"SFT-reverified_n128_s0": AQUA, "SFT-reverified_n512_s0": BLUE, "SFT-exact_n512_s0": PURPLE, "SFT-reverified_n2048_s0": ORANGE, "SFT-exact_full_s0": INK2, "SFT-reverified_full_s0": RED}
for r, c in col.items():
    t = train(r)
    ax.plot([x["step"] for x in t], [x["loss"] for x in t], color=c, lw=1.1, label=r.replace("SFT-", ""))
ax.set_yscale("log")
ax.set_xlabel("optimizer step")
ax.set_title("SFT training loss", loc="left")
ax.legend(frameon=False, fontsize=6.5)
fig.suptitle("Supervised fine-tuning: learning rate, data size and data source (development sets)", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
fig.savefig(O + "/fig1_sft.png")
plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
ax = axs[0]
g = ["GRPO-tune_lr2e-6", "GRPO-tune_lr5e-6", "GRPO-tune_lr1e-5"]
for s, c, lab in (("gsm8k_dev", BLUE, "GSM8K dev"), ("math_dev", ORANGE, "MATH dev")):
    v = [out["runs"][r][s] for r in g]
    ax.errorbar([2e-6, 5e-6, 1e-5], [x["strict"] for x in v], yerr=ci_err(v), color=c, marker="o", ms=4, capsize=3, lw=1.5, label=lab)
    ax.axhline(out["base_r1_zero"][s]["strict"], color=c, lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_xticks([2e-6, 5e-6, 1e-5], ["2e-6", "5e-6", "1e-5"])
ax.minorticks_off()
ax.set_xlabel("GRPO learning rate (30 steps from base)")
ax.set_title("GRPO learning-rate sensitivity", loc="left")
ax.legend(frameon=False, fontsize=7.5)
for ax, s, ttl in ((axs[1], "gsm8k_dev", "GSM8K dev vs training compute"), (axs[2], "math_dev", "MATH dev vs training compute")):
    for r, c, lab, tags in (("GRPO-B_sft512_s2026", BLUE, "B: base > SFT(n512) > GRPO", ["step0", "step25", "step50", "step75", "final_step"]),
                            ("GRPO-A_base_s2026", INK2, "A: base > GRPO (failed at 98)", ["step0", "step25", "step50", "step75"])):
        stp = [0 if t_ == "step0" else 100 if t_ == "final_step" else int(t_[4:]) for t_ in tags]
        off = gpu_h("SFT-reverified_n512_s0") if r.startswith("GRPO-B") else 0.0
        xs = [off + cum_gpu_h(r, k) for k in stp]
        vs = [dict(strict=acc(r, t_, s), ci95=wilson(sum(items(r, t_, s)["strict"].values()), items(r, t_, s)["n"])) for t_ in tags]
        ax.errorbar(xs, [v["strict"] for v in vs], yerr=ci_err(vs), color=c, marker="o", ms=4, capsize=3, lw=1.5, label=lab)
    x80 = cum_gpu_h("GRPO-A_base_s2026", 80)
    for b, c in (("A1", AQUA), ("A2", ORANGE)):
        v = out["runs"][b][s]
        ax.errorbar([x80 + gpu_h(b) * 0 + 2 * sum(x["step_s"] for x in train(b)) / 3600], [v["strict"]], yerr=ci_err([v]), color=c, marker="s", ms=5, capsize=3, lw=0, elinewidth=1.2,
                    label=f"{b} (post-hoc, step 100)")
    ax.set_xlabel("cumulative training GPU-hours (B includes its SFT parent)")
    ax.set_title(ttl + " (strict, 95% CI)", loc="left")
    if s == "gsm8k_dev":
        ax.set_ylim(0.6, 0.96)
        ax.text(0.02, 0.03, f"A step 0 (base): {100 * acc('GRPO-A_base_s2026', 'step0', s):.1f}% (below axis)", transform=ax.transAxes, fontsize=7, color=INK2)
        ax.legend(frameon=False, fontsize=6.8, loc="center right")
fig.suptitle("GRPO: learning rate, direct (A) vs sequential (B), and returns to compute (development sets)", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
fig.savefig(O + "/fig2_grpo.png")
plt.close(fig)

fig, axs = plt.subplots(2, 3, figsize=(13, 6.4), constrained_layout=True)
CG = {"GRPO-B_sft512_s2026": (BLUE, "-", "B (sequential)"), "GRPO-A_base_s2026": (INK2, "-", "A (direct, failed at 98)"), "GRPO-tune_lr1e-5": (AQUA, (0, (3, 2)), "tune lr 1e-5"),
      "GRPO-tune_lr5e-6": (ORANGE, (0, (3, 2)), "tune lr 5e-6"), "GRPO-tune_lr2e-6": (PURPLE, (0, (3, 2)), "tune lr 2e-6")}
for ax, (k, ttl, fn) in zip(axs.flat, [("reward_mean", "Training reward", None), ("zero", "Zero-advantage groups (share of 32)", lambda x: x["zero_adv"] / x["groups"]),
                                       ("entropy", "Mean token entropy (nats)", None), ("len_mean", "Mean response tokens", None),
                                       ("rollout_logprob_mad", "Rollout log-prob gap, mean (gate 0.05)", None), ("grad_norm", "Gradient norm", None)]):
    for r, (c, ls, lab) in CG.items():
        t = train(r)
        ax.plot([x["step"] for x in t], [fn(x) if fn else x[k] for x in t], color=c, ls=ls, lw=1.2, label=lab)
    ax.set_title(ttl, loc="left")
    ax.set_xlabel("step")
axs[1, 1].axhline(0.05, color=RED, lw=1, ls="--")
axs[1, 1].set_yscale("log")
axs[1, 2].set_yscale("log")
axs[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
fig.suptitle("GRPO training dynamics: all registered GRPO runs (32 prompts x 8 rollouts per step)", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
fig.savefig(O + "/fig3_grpo_dynamics.png")
plt.close(fig)

lab = {r: r.replace("SFT-", "SFT ").replace("GRPO-", "GRPO ").replace("_s0", "").replace("_s2026", "") for r in R}
lab.update({"GRPO-A_base_s2026": "GRPO A, step 75 (failed run)", "A1": "A1 post-hoc (lr 1e-5)", "A2": "A2 post-hoc (lr 5e-6)"})
tot = {}
for r, v in out["runs"].items():  # total GPU-hours to produce the checkpoint, including parents (B: its SFT parent; branches: original steps 1-80)
    tot[r] = cum_gpu_h(r, 75) if r == "GRPO-A_base_s2026" else v["gpu_h"] + (gpu_h("SFT-reverified_n512_s0") if r == "GRPO-B_sft512_s2026" else 0) + (cum_gpu_h("GRPO-A_base_s2026", 80) if r in BR else 0)
order = sorted(R, key=lambda r: tot[r])
fig, axs = plt.subplots(1, 2, figsize=(12, 6.0), constrained_layout=True, sharey=True)
for ax, s, ttl in ((axs[0], "gsm8k_dev", "GSM8K dev"), (axs[1], "math_dev", "MATH dev")):
    for y, r in enumerate(order):
        v = out["runs"][r][s]
        c = RED if r in BR or r == "GRPO-A_base_s2026" else (BLUE if r.startswith("GRPO") else ORANGE)
        ax.errorbar([v["strict"]], [y], xerr=[[v["strict"] - v["ci95"][0]], [v["ci95"][1] - v["strict"]]], color=c, marker="^" if "tune" in r else "o", ms=5, capsize=2.5, lw=0, elinewidth=1.2)
    ax.axvline(out["base_r1_zero"][s]["strict"], color=INK2, lw=0.8, ls=":")
    ax.set_title(f"{ttl}, strict (95% CI)", loc="left")
    ax.set_xlabel("accuracy")
axs[0].set_yticks(range(len(order)), [f"{lab[r]}  ({tot[r]:.1f} GPU-h)" for r in order], fontsize=7.5)
axs[0].invert_yaxis()
axs[0].text(out["base_r1_zero"]["gsm8k_dev"]["strict"] + 0.01, len(order) - 0.6, "base model", fontsize=7, color=INK2)
from matplotlib.lines import Line2D  # noqa: E402
fig.legend(handles=[Line2D([], [], color=ORANGE, marker="o", lw=0, label="SFT"), Line2D([], [], color=BLUE, marker="o", lw=0, label="GRPO"),
                       Line2D([], [], color=RED, marker="o", lw=0, label="failed run / post-hoc branch"), Line2D([], [], color=INK2, marker="^", lw=0, label="learning-rate tuning run")],
           frameon=False, fontsize=7.5, loc="outside lower center", ncol=4)
fig.suptitle("Every development checkpoint, ordered by the training GPU-hours needed to produce it (parents included)", x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
fig.savefig(O + "/fig4_efficiency.png")
plt.close(fig)

# ---------- tables ----------
pc = lambda v: "-" if v is None else f"{100 * v:.1f}"
L = ["# Development results, all runs (generated)", "", "Strict = frozen verifier 2.0.0. GSM8K dev n = 1,000; MATH dev n = 500. One seed per arm.", "",
     f"Base model, r1_zero prompt (GRPO-A step-0 evaluation): GSM8K {pc(out['base_r1_zero']['gsm8k_dev']['strict'])}% (relaxed {pc(out['base_r1_zero']['gsm8k_dev']['relaxed'])}), "
     f"MATH {pc(out['base_r1_zero']['math_dev']['strict'])}% (relaxed {pc(out['base_r1_zero']['math_dev']['relaxed'])}). The question-only base prompt was not evaluated on dev (EVAL-base-dev not run).", "",
     "| run | point | lr | GSM8K strict (95% CI) | second | relaxed | MATH strict (95% CI) | second | relaxed | MATH format | MATH truncated | GSM8K tokens | training GPU-h |", "|" + "---|" * 13]
for r, v in out["runs"].items():
    g_, m_ = v["gsm8k_dev"], v["math_dev"]
    L.append(f"| {r} | {v['point']} | {v['lr']:g} | {pc(g_['strict'])} ({pc(g_['ci95'][0])}-{pc(g_['ci95'][1])}) | {pc(g_['second'])} | {pc(g_['relaxed'])} | {pc(m_['strict'])} ({pc(m_['ci95'][0])}-{pc(m_['ci95'][1])}) | "
             f"{pc(m_['second'])} | {pc(m_['relaxed'])} | {pc(m_['format'])} | {m_['truncated']} | {g_['tokens']:.0f} | {v['gpu_h']:.2f} |")
L += ["", "## Paired comparisons (b minus a; exact McNemar; Holm within family and set)", "", "| family | comparison | set | a only | b only | diff pp (95% CI) | p | Holm p (m) | relaxed diff | GPU-h | pp/GPU-h |", "|" + "---|" * 11]
for c in comps:
    if c.get("unavailable"):
        L.append(f"| {c['family']} | {c['name']} | {c['set']} | - | - | unavailable | - | {c['p_holm']:.3g} ({c['holm_m']}) | - | - | - |")
        continue
    L.append(f"| {c['family']} | {c['name']} | {c['set']} | {c['a_only']} | {c['b_only']} | {100 * c['diff']:+.1f} [{100 * c['ci95'][0]:+.1f}, {100 * c['ci95'][1]:+.1f}] | {c['p']:.3g} | {c['p_holm']:.3g} ({c['holm_m']}) | "
             f"{100 * c['relaxed_diff']:+.1f} | {'-' if c.get('gpu_h') is None else f'{c['gpu_h']:.2f}'} | {'-' if c.get('pp_per_gpu_h') is None else f'{c['pp_per_gpu_h']:+.2f}'} |")
L += ["", f"Same weights evaluated twice (SFT parent final vs GRPO-B step 0): {json.dumps(noise)}", "", "## GRPO dynamics (first 10 / last 10 steps)", "",
      "| run | reward | zero-adv share | all-correct share | entropy | tokens | correct/incorrect tokens (last 10) | truncated | grad norm (max) | gap mean (max step) | gap p99 max | token gap max | mismatched params | step s (update s) |", "|" + "---|" * 14]
for r, v in dyn.items():
    f_, l_ = v["first10"], v["last10"]
    L.append(f"| {r} | {f_['reward']:.3f}/{l_['reward']:.3f} | {f_['zero_adv_frac']:.2f}/{l_['zero_adv_frac']:.2f} | {f_['all_correct_frac']:.2f}/{l_['all_correct_frac']:.2f} | {f_['entropy']:.3f}/{l_['entropy']:.3f} | "
             f"{f_['len_mean']:.0f}/{l_['len_mean']:.0f} | {l_['len_correct']:.0f}/{l_['len_incorrect']:.0f} | {pc(f_['truncated'])}/{pc(l_['truncated'])}% | {v['all']['grad_norm']:.3f} ({v['all']['grad_norm_max']:.3f}) | "
             f"{v['all']['gap_mean']:.4f} ({v['all']['gap_max_step']:.4f}) | {v['all']['gap_p99_max']:.3f} | {v['all']['token_gap_max']:.2f} | {v['all']['mismatched_params']} | {v['all']['step_s']:.0f} ({v['all']['update_s']:.0f}) |")
L += ["", "## 25-step windows", ""] + [f"- {r} {w['window']}: zero-adv {w['zero_adv_frac']:.2f}, entropy {w['entropy']:.3f}, reward {w['reward']:.3f}, tokens {w['len_mean']:.0f}, gap {w['gap']:.4f}" for r, ws in win.items() for w in ws]
L += ["", "## SFT", ""] + [f"- {r}: n {v['n']}, {v['steps']} steps, loss {v['loss_first']:.3f} -> {v['loss_last5']:.3f}, grad norm max {v['grad_norm_max']:.2f}, {v['tok_per_s']:.0f} tok/s, {v['gpu_h']:.2f} GPU-h" for r, v in sft.items()]
open(O + "/REPORT_DATA.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
