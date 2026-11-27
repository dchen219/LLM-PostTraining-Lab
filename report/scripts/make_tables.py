"""Build every report table from the frozen Phase I artifacts (read-only). Writes report/tables/<name>.csv and <name>.md.

Sources (never modified):
  results/research_analysis/consolidated_results.csv, statistical_comparisons.csv, data/*.csv   (frozen, PHASE1_FREEZE.sha256)
  data/stage0/MANIFEST.json                                                                    (Stage 0 data manifest)
  ~/a5_sessions, ~/a5_recovery: run.json / DONE.json of every training run                     (PHASE1_INPUTS.sha256 manifests)

python report/scripts/make_tables.py --runs ~/a5_sessions --runs ~/a5_recovery
"""
import argparse, csv, json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RA = os.path.join(ROOT, "results", "research_analysis")
OUT = os.path.join(ROOT, "report", "tables")
ap = argparse.ArgumentParser()
ap.add_argument("--runs", action="append", required=True)
a = ap.parse_args()
RUNS = [os.path.expanduser(r) for r in a.runs]
os.makedirs(OUT, exist_ok=True)


def rcsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def f(x, d=1, sign=False):
    if x in ("", None):
        return "–"
    v = float(x)
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "−")


def pv(x):
    if x in ("", None):
        return "–"
    v = float(x)
    if v == 0:
        return "<1e-300"
    if v >= 0.001:
        return f"{v:.3f}"
    return f"{v:.1e}".replace("e-0", "e-")


def write(name, header, rows, note=""):
    with open(os.path.join(OUT, name + ".csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    md = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    md += "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)
    if note:
        md += "\n" + note + "\n"
    open(os.path.join(OUT, name + ".md"), "w", encoding="utf-8", newline="\n").write(md)


def run_dir(rel):
    for t in RUNS:
        p = os.path.join(t, rel)
        if os.path.isdir(p):
            return p
    raise SystemExit(f"missing run {rel}")


cons = rcsv(os.path.join(RA, "consolidated_results.csv"))
comp = rcsv(os.path.join(RA, "statistical_comparisons.csv"))
ORDER = ["BASE:r1_zero", "BASE:question_only", "SFT-reverified_n128_s0", "SFT-reverified_n512_s0", "SFT-exact_n512_s0", "SFT-reverified_n2048_s0",
         "SFT-reverified_sizecontrol_s0", "SFT-exact_full_s0", "SFT-reverified_full_s0", "GRPO-B_sft512_s2026",
         "GRPO-A-replay80-lr1e-5-s2026", "GRPO-A-recovery80-lr5e-6-s2026"]


def key(r):
    return r["checkpoint"] + (":" + r["prompt"] if r["checkpoint"] == "BASE" else "")


by = {(key(r), r["set"]): r for r in cons}
LABEL = {"BASE:r1_zero": "Base, r1_zero prompt", "BASE:question_only": "Base, question-only prompt"}
for r in cons:
    LABEL.setdefault(key(r), r["label"])

# ---- T1 main held-out results -------------------------------------------------------------------
rows = []
for k in ORDER:
    g, m = by[(k, "gsm8k_test_final")], by[(k, "math_val_final")]
    cls = {"preregistered": "pre-registered", "posthoc": "post-hoc"}[g["cls"]]
    ci = lambda r: f"{100 * float(r['strict']):.1f} ({100 * float(r['strict_ci_lo']):.1f}–{100 * float(r['strict_ci_hi']):.1f})"
    rows.append([LABEL[k], cls, ci(g), ci(m), f(100 * float(g["relaxed"])), f(100 * float(m["relaxed"])),
                 f(100 * float(g["format"])), f(100 * float(m["format"])), f(g["cumulative_gpu_h"], 2)])
rows.append(["GRPO-A, direct (pre-registered)", "pre-registered", "FAILED at step 98", "FAILED", "–", "–", "–", "–", "6.61 (sunk)"])
write("t01_main_results", ["Checkpoint", "Class", "GSM8K strict % (95% CI)", "MATH strict % (95% CI)", "GSM8K relaxed %", "MATH relaxed %",
                           "GSM8K format %", "MATH format %", "Cumulative GPU-h"], rows,
      "Source: `results/research_analysis/consolidated_results.csv`. Wilson 95% CI. GPU-h = 2 × training wall-clock on 2× RTX A6000, cumulative over the lineage (GRPO-B includes its SFT parent; A1/A2 include GRPO-A steps 1–80).")

# ---- T2 output-length and truncation ------------------------------------------------------------
rows = []
for k in ORDER:
    for s, nm in (("gsm8k_test_final", "GSM8K"), ("math_val_final", "MATH")):
        r = by[(k, s)]
        rows.append([LABEL[k], nm, f(r["tokens_mean"], 0), f(r["tokens_correct"], 0), f(r["tokens_incorrect"], 0), f(100 * float(r["truncated_rate"])),
                     f(100 * float(r["truncated_share_of_errors"])), f(100 * float(r["second"])), r["double_think"]])
write("t02_lengths_truncation", ["Checkpoint", "Set", "Mean tokens", "Tokens (correct)", "Tokens (wrong)", "Truncated %", "Truncated share of errors %",
                                 "Second-verifier acc. %", "Doubled `</think>`"], rows, "Source: `consolidated_results.csv`.")


# ---- comparison tables --------------------------------------------------------------------------
def comp_rows(fams, names=None):
    out = []
    for r in comp:
        if r["family"] not in fams or r["metric"] != "strict" or (names and r["name"] not in names):
            continue
        s = "GSM8K" if r["set"].startswith("gsm8k") else "MATH"
        if r["unavailable"]:
            out.append([r["name"], s, "–", "–", "–", "–", "–", "1 (counted)", "unavailable: pre-registered run failed"])
            continue
        out.append([r["name"], s, r["a_only"], r["b_only"], f"{f(r['diff_pp'], 1, True)} [{f(r['ci_lo_pp'], 1, True)}, {f(r['ci_hi_pp'], 1, True)}]",
                    f(r["relaxed_diff_pp"], 1, True), pv(r["p"]), pv(r["p_holm"]) if r["p_holm"] else "–", r["reading"]])
    return out


H = ["Comparison (a → b)", "Set", "a only", "b only", "b − a, pp [95% CI]", "relaxed b − a", "p (exact McNemar)", "Holm p", "Reading"]
write("t03_confirmatory", H, comp_rows({"confirmatory"}), "Pre-registered confirmatory family (Holm m = 4 per set; C4 counted p = 1). Source: `statistical_comparisons.csv`.")
write("t04_sft_vs_base", H, comp_rows({"confirmatory-secondary"}), "Pre-registered confirmatory-secondary family (Holm m = 7 per set).")
write("t05_descriptive", H, comp_rows({"descriptive"}), "Pre-registered descriptive comparisons: unadjusted p, no claim rests on them.")
write("t06_posthoc", H, comp_rows({"post-hoc (frozen)", "post-hoc (unregistered)"}),
      "Post-hoc. 'frozen' = the A1/A2 family listed in freeze_v1.yaml (Holm m = 4); 'unregistered' = chosen after the held-out results (Holm m = 5). Exploratory only.")
rows = []
for r in comp:
    if r["family"] == "sensitivity":
        rows.append([r["name"], "GSM8K" if r["set"].startswith("gsm8k") else "MATH", f(r["diff_pp"], 1, True), f(r["ci_lo_pp"], 1, True) + " to " + f(r["ci_hi_pp"], 1, True), pv(r["p"])])
write("t07_relaxed_sensitivity", ["Confirmatory contrast under relaxed scoring", "Set", "b − a, pp", "95% CI", "p"], rows, "Sensitivity analysis (not a test of the registered hypotheses).")

# ---- T8 seed sensitivity -------------------------------------------------------------------------
rows = []
for r in rcsv(os.path.join(RA, "data", "seed_sensitivity.csv")):
    if r["family"] not in ("confirmatory", "descriptive"):
        continue
    rows.append([r["name"], "GSM8K" if r["set"].startswith("gsm8k") else "MATH", f(r["diff_pp"], 1, True), f(r["item_se_pp"], 2),
                 f(r["seed_sd_contrast_pp"], 2) if r["seed_sd_contrast_pp"] else "unknown", f(r["total_se_pp"], 2),
                 f(r["ratio_diff_to_total_se"], 1, True)])
write("t08_seed_sensitivity", ["Contrast", "Set", "Diff pp", "Item SE pp", "Seed SD pp (heuristic)", "Combined SE pp", "Diff / combined SE"], rows,
      "Heuristic: seed SD from one dev seed pair (SFT re-verified 512, seed 0 vs 100; 1 df). Not a test. Source: `data/seed_sensitivity.csv`.")

# ---- T9 GRPO-B windows ---------------------------------------------------------------------------
rows = []
for r in rcsv(os.path.join(RA, "data", "grpo_compute_windows.csv")):
    rows.append([r["run"], r["window"], f(100 * float(r["zero_adv_frac"]), 0), f(r["entropy"], 2), f(r["reward"], 2), f(r["len_mean"], 0),
                 f(r["gsm8k_dev_gain_pp"], 1, True), f(r["math_dev_gain_pp"], 1, True), f(r["gpu_h"], 2)])
write("t09_grpo_windows", ["Run", "Steps", "Zero-advantage groups %", "Entropy (nats)", "Reward", "Mean length", "GSM8K dev gain pp", "MATH dev gain pp", "GPU-h"],
      rows, "Development sets only (exploratory). Source: `data/grpo_compute_windows.csv`.")

# ---- T10 gain transitions ------------------------------------------------------------------------
rows = []
for r in rcsv(os.path.join(RA, "data", "gain_transitions.csv")):
    rows.append([r["transition"], "GSM8K" if r["set"].startswith("gsm8k") else "MATH", r["kind"], r["category"].replace("_", " "), r["count"]])
write("t10_gain_transitions", ["Transition", "Set", "Kind", "Category", "Items"], rows,
      "Item-level outcome categories of the items gained or lost in each transition. Source: `data/gain_transitions.csv`.")

# ---- T11 efficiency --------------------------------------------------------------------------------
rows = []
for r in rcsv(os.path.join(RA, "data", "efficiency.csv")):
    rows.append([r["label"], "GSM8K" if r["set"].startswith("gsm8k") else "MATH", f(100 * float(r["strict"])), f(r["own_gpu_h"], 2), f(r["cumulative_gpu_h"], 2),
                 f(r["gain_vs_base_qo_pp"], 1, True), f(r["pp_per_gpu_h_vs_qo"], 1, True)])
write("t11_efficiency", ["Checkpoint", "Set", "Strict %", "Own GPU-h", "Cumulative GPU-h", "Gain vs base question-only pp", "pp per cumulative GPU-h"], rows,
      "Source: `data/efficiency.csv`. Training wall-clock includes in-training dev evaluations.")

# ---- T12 dev-set tuning and per-run dev results (from DONE.json) --------------------------------
rows = []
DEV = [("sft/tune_lr1e-5", "SFT lr tuning", "1e-5"), ("sft/tune_lr2e-5", "SFT lr tuning", "2e-5"), ("sft/tune_lr5e-5", "SFT lr tuning", "5e-5"),
       ("grpo/tune_lr2e-6", "GRPO lr tuning (30 steps, from base)", "2e-6"), ("grpo/tune_lr5e-6", "GRPO lr tuning (30 steps, from base)", "5e-6"),
       ("grpo/tune_lr1e-5", "GRPO lr tuning (30 steps, from base)", "1e-5")]
for rel, what, lr in DEV:
    d = json.load(open(os.path.join(run_dir(rel), "DONE.json")))
    e = d["eval"]
    rows.append([what, lr, d["steps"], f(100 * e["gsm8k_dev"]["accuracy"]), f(100 * e["math_dev"]["accuracy"]), f(d["wall_s"] / 1800, 2)])
write("t12_lr_tuning", ["Sweep", "Learning rate", "Steps", "GSM8K dev %", "MATH dev %", "GPU-h"], rows,
      "Rule fixed before tuning (EXPERIMENT_PLAN.md §10): highest final-step GSM8K dev accuracy; if another rate is within 1.0 pp, the smallest such rate. "
      "Selected: SFT 1e-5, GRPO 1e-5. Seed 100 (tuning only). Source: DONE.json of each run.")

# ---- T13 hyperparameters as run (run.json) --------------------------------------------------------
sft = json.load(open(os.path.join(run_dir("sft/reverified_n512_s0"), "run.json")))
grp = json.load(open(os.path.join(run_dir("grpo/B_sft512_s2026"), "run.json")))
sc, gc = sft.get("config", sft), grp.get("config", grp)
st, gt, gg = sc["train"], gc["train"], gc["grpo"]
rows = [["SFT", "optimizer", f"AdamW β=({st['betas'][0]}, {st['betas'][1]}), weight decay {st['weight_decay']}"],
        ["SFT", "learning rate / warm-up", f"{st['lr']:g} constant after {st['warmup']} warm-up steps (selected on dev)"],
        ["SFT", "batch / micro-batch", f"{st['batch_size']} sequences / {st['micro_batch']} (gradient accumulation)"],
        ["SFT", "epochs / max length", f"{st['epochs']} / {st['max_len']} tokens"],
        ["SFT", "loss", "token-mean NLL over response tokens of the whole optimizer batch; EOS appended"],
        ["SFT", "precision", f"{sc.get('dtype')} master weights, bf16 autocast"],
        ["GRPO", "rollouts per step", f"{gg['prompts_per_step']} prompts × {gg['group_size']} samples = {gg['prompts_per_step'] * gg['group_size']}, temperature {gg['temperature']}, max {gg['max_tokens']} new tokens"],
        ["GRPO", "steps / learning rate", f"{gg['steps']} / {gt['lr']:g} constant after {gt['warmup']} warm-up steps (selected on dev)"],
        ["GRPO", "loss", f"{gg['loss']} (clip {gg['clip']}), length normalisation '{gg['length_norm']}' ({gg['norm_const']}), std normalisation {gg['std_norm']}, no KL term"],
        ["GRPO", "updates per batch", f"{gg['epochs']} epoch, update batch {gg['update_batch']} (one optimizer step per rollout batch: ratio = 1)"],
        ["GRPO", "gradient clipping", f"max norm {gt['max_grad_norm']}"],
        ["GRPO", "micro-batching", f"micro-batch {gt['micro_batch']}, token budget {gt['token_budget']} padded tokens (memory only; gradient unchanged)"],
        ["GRPO", "gates", f"exact weight-sync check every step; rollout log-prob gate: mean absolute sampling-vs-policy gap ≤ {gg['rollout_tol']}"],
        ["GRPO", "precision / devices", f"{gc.get('dtype')} master weights on {gc['device']}; vLLM engine on {gc['engine']['device']} (gpu_mem {gc['engine']['gpu_mem']}, prefix caching {gc['engine']['prefix_caching']})"],
        ["Evaluation", "decoding", "greedy, 1,024 new tokens, vLLM 0.7.2, max_model_len 2,048, prefix caching off, seed 2026, RTX A6000"]]
write("t13_hyperparameters", ["Stage", "Setting", "Value (as run)"], rows,
      "Source: run.json of SFT-reverified_n512_s0 and GRPO-B_sft512_s2026 (identical across arms except data, size, seed and init); evaluation: final/freeze_v1.yaml.")

# ---- T14 data ------------------------------------------------------------------------------------
man = json.load(open(os.path.join(ROOT, "data", "stage0", "MANIFEST.json")))
c = man["counts"]
sub = man["subsets"]
rows = [["Teacher traces (gpt-oss-120b on math12k)", f"{c['raw']:,}", "all SFT candidates"],
        ["Exact-match (dataset's published string-match filter)", c["exact"], "SFT exact-match arms"],
        ["Re-verified (strict verifier on all traces; this project)", c["reverified"], "exact-match plus 1,049 recovered traces"],
        ["Exact-match after de-duplication / decontamination", sub["exact_full"]["n"], "SFT exact full"],
        ["Re-verified after de-duplication / decontamination", sub["reverified_full"]["n"], "SFT re-verified full"],
        ["Size control (re-verified pool, exact-match size)", sub["reverified_sizecontrol_s0"]["n"], "volume vs composition"],
        ["Matched subsets", 512, "exact_n512_s0, reverified_n512_s0 (nested, shared ordering)"],
        ["RL training pool (math12k problems)", c["rl_pool"], "GRPO prompts"],
        ["GSM8K dev (from GSM8K train)", c["gsm8k_dev"], "all tuning and monitoring"],
        ["MATH dev (untraced math12k problems)", c["math_dev"], "secondary dev"],
        ["GSM8K test (held-out, primary)", c["gsm8k_test"], "evaluated once per frozen checkpoint"],
        ["MATH val (held-out, secondary)", c["math_final"], "evaluated once per frozen checkpoint"]]
rows = [[r[0], f"{r[1]:,}" if isinstance(r[1], int) else r[1], r[2]] for r in rows]
write("t14_data", ["Dataset", "n", "Role"], rows, "Source: `data/stage0/MANIFEST.json`.")
print("tables:", sorted(x for x in os.listdir(OUT) if x.endswith(".md")))

# ---- T15 run inventory: all 17 documented training runs --------------------------------------------
AN = json.load(open(os.path.join(RA, "analysis.json")))
curves = {(r["run"], int(r["step"]), r["set"]): float(r["strict"]) for r in rcsv(os.path.join(RA, "data", "grpo_dev_curves.csv"))}
INV = [("sft/tune_lr1e-5", "SFT", "lr tuning (dev only)"), ("sft/tune_lr2e-5", "SFT", "lr tuning (dev only)"), ("sft/tune_lr5e-5", "SFT", "lr tuning (dev only)"),
       ("sft/reverified_n128_s0", "SFT", "pre-registered"), ("sft/reverified_n512_s0", "SFT", "pre-registered"), ("sft/exact_n512_s0", "SFT", "pre-registered"),
       ("sft/reverified_n2048_s0", "SFT", "pre-registered"), ("sft/reverified_sizecontrol_s0", "SFT", "pre-registered"), ("sft/exact_full_s0", "SFT", "pre-registered"),
       ("sft/reverified_full_s0", "SFT", "pre-registered"), ("grpo/tune_lr2e-6", "GRPO", "lr tuning (dev only)"), ("grpo/tune_lr5e-6", "GRPO", "lr tuning (dev only)"),
       ("grpo/tune_lr1e-5", "GRPO", "lr tuning (dev only)"), ("grpo/B_sft512_s2026", "GRPO", "pre-registered"), ("grpo/A_base_s2026", "GRPO", "pre-registered"),
       ("recovery/A_replay80_lr1e-5_s2026", "GRPO", "post-hoc diagnostic"), ("recovery/A_recovery80_lr5e-6_s2026", "GRPO", "post-hoc diagnostic")]
rows = []
for rel, kind, cls in INV:
    d = run_dir(rel)
    rj = json.load(open(os.path.join(d, "run.json")))
    c = rj.get("config", rj)
    ip = c["init"]["path"]
    if kind == "SFT":
        what = os.path.basename(c["data"]["subset"]).replace(".json", "")
    else:
        what = "RL pool, 32×8/step, " + ("fork of GRPO-A step-80 state" if rel.startswith("recovery/") else "init SFT re-verified 512" if "/sft/reverified_n512_s0" in ip else "init base model")
    if os.path.exists(os.path.join(d, "DONE.json")):
        dn = json.load(open(os.path.join(d, "DONE.json")))
        ev = dn["eval"]
        status, steps, gh = "completed", dn["steps"], dn["wall_s"] / 1800
        g, m = f(100 * ev["gsm8k_dev"]["accuracy"]), f(100 * ev["math_dev"]["accuracy"])
        held = "yes" if cls in ("pre-registered", "post-hoc diagnostic") else "no (tuning)"
    else:
        status, steps, gh = "FAILED at step 98 (rollout gate)", 97, AN["failed_A"]["gpu_h_total"]
        lab = "GRPO-A (direct, FAILED at 98)"
        g, m = f(100 * curves[(lab, 75, "gsm8k_dev")]) + " (step 75)", f(100 * curves[(lab, 75, "math_dev")]) + " (step 75)"
        held = "no (failed; never evaluated)"
    rows.append([("SFT-" if kind == "SFT" else "GRPO-") + rel.split("/")[1], cls, what, f"{float(c['train']['lr']):.0e}".replace("e-0", "e-"), c["seed"], steps, status, g, m, f"{gh:.2f}", held])
write("t15_run_inventory", ["Run", "Class", "Data / init", "lr", "Seed", "Optimizer steps", "Status", "GSM8K dev % (final)", "MATH dev % (final)", "GPU-h", "Held-out evaluated"],
      rows, "All 17 documented training runs (15 Tier A + 2 post-hoc branches; branch steps are 81–100 from GRPO-A's step-80 state). "
      "Sources: run.json / DONE.json of each run; GRPO-A from its metrics (`analysis.json`, `grpo_dev_curves.csv`).")
print("t15 rows:", len(rows))
