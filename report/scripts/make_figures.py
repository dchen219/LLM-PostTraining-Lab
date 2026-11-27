"""Report figures. Copies the frozen research figures (byte-identical) and draws two diagrams (architecture, experimental design).

PYTHONPATH=~/a5_plotlib python report/scripts/make_figures.py
"""
import os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)

# ---- 1. frozen figures, copied unchanged ------------------------------------------------------------
SRC = os.path.join(ROOT, "results", "research_analysis", "figures")
copied = []
for f in sorted(os.listdir(SRC)):
    if f.endswith((".png", ".pdf")):
        shutil.copyfile(os.path.join(SRC, f), os.path.join(OUT, f))
        copied.append(f)
INC = os.path.join(ROOT, "results", "grpo_a_incident_analysis")
for src, dst in (("fig1_trajectory.png", "fig11_grpo_a_incident_trajectory.png"), ("fig2_warning_scores.png", "fig12_grpo_a_warning_scores.png")):
    shutil.copyfile(os.path.join(INC, src), os.path.join(OUT, dst))
    copied.append(dst)

# ---- 2. diagrams ----------------------------------------------------------------------------------
INK, MUTED, LINE = "#1f2328", "#59636e", "#8c959f"
BLUE, ORANGE, AQUA, RED, PURPLE = "#257de0", "#f47230", "#00b894", "#ec4647", "#8e6cdb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, title, lines=(), color=BLUE, fill=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", lw=1.4, ec=color, fc=fill or "white"))
    ax.text(x + w / 2, y + h - 0.028, title, ha="center", va="top", fontsize=9.2, fontweight="bold", color=INK)
    for i, l in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.066 - i * 0.032, l, ha="center", va="top", fontsize=7.6, color=MUTED)


def arrow(ax, a, b, color=LINE, text=None, rad=0.0, dx=0, dy=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, text, ha="center", va="center", fontsize=7.2, color=MUTED,
                bbox=dict(fc="white", ec="none", pad=1))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    copied.append(name)


# 2a. system architecture: four columns, two rows
fig, ax = plt.subplots(figsize=(12, 6.0))
ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 0.985, "System architecture: data → SFT → GRPO → frozen evaluation", fontsize=12, fontweight="bold", color=INK, va="top")
W, X = 0.225, (0.0, 0.258, 0.516, 0.774)
TOP, BOT, H = 0.58, 0.20, 0.29
box(ax, X[0], TOP, W, H, "Stage 0 data (CPU)", ["teacher traces (gpt-oss-120b)", "exact-match vs strict re-verify", "de-dup + decontamination", "nested matched subsets", "MANIFEST.json (sha256)"], PURPLE)
box(ax, X[1], TOP, W, H, "SFT trainer", ["fp32 master weights", "bf16 autocast", "token-mean NLL, accumulation", "lineage.json + weights sha256"], BLUE)
box(ax, X[2], TOP, W, H, "GRPO trainer", ["GPU 0: vLLM rollout engine", "GPU 1: fp32 policy + AdamW", "32 prompts × 8 samples", "reward: verifier 2.0.0", "group-mean advantages"], ORANGE)
box(ax, X[3], TOP, W, H, "Frozen evaluation", ["freeze_v1.yaml (sha256)", "checkpoint lock before eval", "greedy vLLM on 2 GPUs", "11 checkpoints, 2 sets"], AQUA)
box(ax, X[0], BOT, W, H, "Verifier 2.0.0 (frozen)", ["strict grader (r1_zero format)", "+2 additive rules", "hash-locked at run start", "second: math-verify", "relaxed extraction"], PURPLE)
box(ax, X[1], BOT, W, H, "Run integrity", ["immutable run folders", "resume rewinds, no duplicates", "incident guard:", "a failed run is never resumed"], BLUE)
box(ax, X[2], BOT, W, H, "Correctness gates", ["exact weight sync (bf16 bits)", "rollout log-prob gap ≤ 0.05", "non-finite loss / grad stop", "failed batch + FAILED.json"], RED, "#fff7f7")
box(ax, X[3], BOT, W, H, "Analysis (read-only)", ["exact McNemar + Holm", "Wilson / Wald CIs", "217-check audit", "hash manifests + freeze"], AQUA)
ym = TOP + H / 2
for i, t in enumerate(("subsets", "SFT parent", "final step")):
    arrow(ax, (X[i] + W, ym), (X[i + 1], ym))
arrow(ax, (X[2] + W / 2, TOP), (X[2] + W / 2, BOT + H), color=RED)
ax.text(X[2] + W / 2 + 0.008, (TOP + BOT + H) / 2, "checked every step", fontsize=7.2, color=MUTED, va="center")
arrow(ax, (X[3] + W / 2, TOP), (X[3] + W / 2, BOT + H))
ax.text(0.0, 0.11, "Hardware: 2× NVIDIA RTX A6000 (48 GB). Software: PyTorch 2.5.1, Transformers 4.51.3, vLLM 0.7.2, math-verify 0.7.0. "
        "Base model: Qwen2.5-Math-1.5B (pinned revision).", fontsize=7.8, color=MUTED)
ax.text(0.0, 0.065, "Every checkpoint carries lineage (parent weights sha256, config sha256, code sha256); evaluation refuses weights that do not match the lock.",
        fontsize=7.8, color=MUTED)
save(fig, "fig00_architecture")

# 2b. experimental design
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 0.985, "Experimental design (Tier A, one training seed per arm)", fontsize=12, fontweight="bold", color=INK, va="top")
box(ax, 0.00, 0.47, 0.15, 0.2, "Base model", ["Qwen2.5-Math-1.5B", "r1_zero and", "question-only prompts"], INK)
sft = [("exact 3,420 (full)", BLUE), ("re-verified 4,423 (full)", ORANGE), ("re-verified 3,420 (size ctl)", ORANGE), ("exact 512", BLUE),
       ("re-verified 512", ORANGE), ("re-verified 128", ORANGE), ("re-verified 2,048", ORANGE)]
for i, (t, c) in enumerate(sft):
    y = 0.84 - i * 0.092
    ax.add_patch(FancyBboxPatch((0.26, y), 0.21, 0.066, boxstyle="round,pad=0.006,rounding_size=0.01", lw=1.2, ec=c, fc="white"))
    ax.text(0.365, y + 0.033, "SFT " + t, ha="center", va="center", fontsize=8, color=INK)
    arrow(ax, (0.15, 0.57), (0.26, y + 0.033), color="#d0d7de")
yb = 0.84 - 4 * 0.092 + 0.033
ax.add_patch(FancyBboxPatch((0.58, yb - 0.035), 0.2, 0.07, boxstyle="round,pad=0.006,rounding_size=0.01", lw=1.6, ec=AQUA, fc="white"))
ax.text(0.68, yb, "GRPO-B (100 steps)", ha="center", va="center", fontsize=8.5, color=INK, fontweight="bold")
arrow(ax, (0.47, yb), (0.58, yb), color=AQUA)
ax.text(0.525, yb + 0.025, "C3", ha="center", fontsize=7.6, color=MUTED)
ax.add_patch(FancyBboxPatch((0.56, 0.075), 0.24, 0.07, boxstyle="round,pad=0.006,rounding_size=0.01", lw=1.6, ec=RED, fc="#fff7f7"))
ax.text(0.68, 0.11, "GRPO-A direct: FAILED @ 98", ha="center", va="center", fontsize=8.3, color=RED, fontweight="bold")
ax.add_patch(FancyArrowPatch((0.075, 0.47), (0.56, 0.11), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=RED, connectionstyle="angle,angleA=-90,angleB=180,rad=0"))
ax.text(0.30, 0.13, "C4: pathway A vs base (unavailable)", fontsize=7.6, color=MUTED)
arrow(ax, (0.80, 0.11), (0.82, 0.11), color=LINE)
ax.text(0.825, 0.11, "post-hoc branches from step 80:\nA1 (+20 steps, lr 1e-5)\nA2 (+20 steps, lr 5e-6)", fontsize=7.8, color=MUTED, va="center", style="italic")
ax.text(0.53, 0.86, "C1: exact full vs re-verified full", fontsize=8, color=MUTED)
ax.text(0.53, 0.82, "C2: exact 512 vs re-verified 512", fontsize=8, color=MUTED)
ax.text(0.53, 0.78, "SFT arms vs base question-only (×7)", fontsize=8, color=MUTED)
ax.text(0.53, 0.74, "scaling 128 → 512 → 2,048 → 4,423 (descriptive)", fontsize=8, color=MUTED)
ax.text(0.84, 0.47, "Held-out, once each:\nGSM8K test (1,319)\nMATH val (5,000)", fontsize=8.5, color=INK, va="center")
ax.text(0.0, 0.015, "Blue = exact-match data; orange = re-verified data. All SFT arms: lr 1e-5, 2 epochs, seed 0. GRPO: lr 1e-5, seed 2026.", fontsize=7.8, color=MUTED)
save(fig, "fig13_experimental_design")
print("figures:", len(copied))

# 2c. teacher-student data pipeline
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 0.985, "Teacher–student reasoning data pipeline (Stage 0, CPU)", fontsize=12, fontweight="bold", color=INK, va="top")
W2, H2, Y2 = 0.17, 0.25, 0.50
cols = [(0.00, "Teacher traces", ["public dataset", "gpt-oss-120b on math12k", "4,836 traces", "(not generated here)"], INK, None),
        (0.205, "Two label filters", ["exact-match: 3,496", "(dataset's filtered file)", "strict re-verify: 4,545", "(+1,049 recovered)"], PURPLE, None),
        (0.41, "Cleaning", ["normalised de-duplication", "13-gram + fuzzy decontam.", "vs GSM8K / MATH eval sets", "→ 3,420 / 4,423"], BLUE, None),
        (0.615, "Matched subsets", ["nested, shared ordering", "128 / 512 / 2,048 / full", "size control 3,420", "512 overlap: 387 shared"], ORANGE, None),
        (0.82, "Student SFT", ["Qwen2.5-Math-1.5B", "r1_zero format", "7 pre-registered arms", "+ RL pool 6,832 prompts"], AQUA, None)]
for x, t, ls, c, fl in cols:
    box(ax, x, Y2, W2, H2, t, ls, c, fl)
for i in range(4):
    arrow(ax, (cols[i][0] + W2, Y2 + H2 / 2), (cols[i + 1][0], Y2 + H2 / 2))
ax.text(0.0, 0.2, "Verification of recovered labels: 781 by an independent canonicaliser, 148 as exact rationals, 80 by math-verify, 40 by reading; 0 numeric conflicts.",
        fontsize=7.8, color=MUTED)
ax.text(0.0, 0.13, "Held-out sets (GSM8K test 1,319; MATH val 5,000) are never used for training or selection; their ids are disjoint from all SFT and dev ids.",
        fontsize=7.8, color=MUTED)
ax.text(0.0, 0.06, "Sources: data/stage0/MANIFEST.json, scripts/stage0_build_data.py, EXPERIMENT_PLAN.md §1.", fontsize=7.8, color=MUTED)
save(fig, "fig14_data_pipeline")
print("figures (with pipeline):", len(copied))
