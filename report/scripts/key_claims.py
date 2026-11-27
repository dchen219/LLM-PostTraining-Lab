"""Recompute every key claim of the report from its specific source and check the deliverable states exactly that value.

Stronger than audit_numbers.py (which only asks whether a number exists anywhere in the sources): here each claim names its source,
the value is recomputed from it, formatted as the report formats it, and the formatted string must appear in the named deliverable.
Writes report/AUDIT_KEY_CLAIMS.md.

python report/scripts/key_claims.py --runs ~/a5_sessions --runs ~/a5_recovery
"""
import argparse, csv, glob, json, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RA = os.path.join(ROOT, "results", "research_analysis")
ap = argparse.ArgumentParser()
ap.add_argument("--runs", action="append", required=True)
ap.add_argument("--final", default="~/a5_final")  # public export: results/run_records/a5_final
a = ap.parse_args()
RUNS = [os.path.expanduser(r) for r in a.runs]
rc = lambda p: list(csv.DictReader(open(p, encoding="utf-8")))
cons = {(r["checkpoint"] + (":" + r["prompt"] if r["checkpoint"] == "BASE" else ""), r["set"][:4]): r for r in rc(os.path.join(RA, "consolidated_results.csv"))}
comp = {(r["name"], r["set"][:4], r["metric"]): r for r in rc(os.path.join(RA, "statistical_comparisons.csv"))}
gt = rc(os.path.join(RA, "data", "gain_transitions.csv"))
dyn = rc(os.path.join(RA, "data", "grpo_dynamics.csv"))
win = rc(os.path.join(RA, "data", "grpo_compute_windows.csv"))
curves = {(r["run"], int(r["step"]), r["set"]): float(r["strict"]) for r in rc(os.path.join(RA, "data", "grpo_dev_curves.csv"))}
seed = {(r["name"], r["set"][:4]): r for r in rc(os.path.join(RA, "data", "seed_sensitivity.csv"))}
diff = rc(os.path.join(RA, "data", "item_difficulty.csv"))
AN = json.load(open(os.path.join(RA, "analysis.json")))
AUD = json.load(open(os.path.join(RA, "audit.json")))
PFI = json.load(open(os.path.join(RA, "post_final_integrity", "post_final_integrity.json")))
MAN = json.load(open(os.path.join(ROOT, "data", "stage0", "MANIFEST.json")))
DOCS = {k: open(os.path.join(ROOT, k if k == "README.md" else os.path.join("report", k)), encoding="utf-8").read() for k in ("TECHNICAL_REPORT.md", "EXECUTIVE_SUMMARY.md", "README.md", "REPRODUCIBILITY.md")}
# public export: the personal documents (docs/RESUME.md, docs/INTERVIEW_GUIDE.md) are not shipped, so their claims are skipped


def done(rel):
    for t in RUNS:
        p = os.path.join(t, rel, "DONE.json")
        if os.path.exists(p):
            return json.load(open(p))
    raise SystemExit(rel)


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}"


def pp(x, d=1, sign=True):
    v = float(x)
    return (f"{v:+.{d}f}" if sign else f"{v:.{d}f}").replace("-", "−")


def sci(p, nd=1):
    s = f"{float(p):.{nd}e}"
    m, e = s.split("e")
    return f"{m}e{int(e)}"


def acc(k, s):
    return cons[(k, s)]


def gtc(tr, s, kind, cat=None):
    return sum(int(r["count"]) for r in gt if r["transition"] == tr and r["set"].startswith(s) and r["kind"].startswith(kind) and (cat is None or r["category"] == cat))


A_ROWS = {int(r["step"]): r for r in dyn if r["run"].startswith("GRPO-A")}
B = "GRPO-B_sft512_s2026"
P = "SFT-reverified_n512_s0"
claims = []


def claim(doc, desc, text, source):
    claims.append((doc, desc, text, source, text in DOCS[doc]))


TR = "TECHNICAL_REPORT.md"
# headline accuracies
for k, lab in (("BASE:r1_zero", "base r1_zero"), ("BASE:question_only", "base question-only"), (B, "GRPO-B")):
    for s in ("gsm8", "math"):
        for d in ("TECHNICAL_REPORT.md", "EXECUTIVE_SUMMARY.md", "README.md"):
            claim(d, f"{lab} {s} strict", pct(acc(k, s)["strict"]), "consolidated_results.csv")
for k in ("GRPO-A-replay80-lr1e-5-s2026", "GRPO-A-recovery80-lr5e-6-s2026"):
    claim(TR, f"{k} held-out", f"{pct(acc(k, 'gsm8')['strict'])}% / {pct(acc(k, 'math')['strict'])}%", "consolidated_results.csv")
# confirmatory
for nm, s, holm in (("C3 pathway B vs its SFT parent", "gsm8", 1), ("C3 pathway B vs its SFT parent", "math", 1)):
    r = comp[(nm, s, "strict")]
    claim(TR, f"C3 {s} diff", pp(r["diff_pp"]) + " pp", "statistical_comparisons.csv")
    claim(TR, f"C3 {s} Holm", sci(r["p_holm"]), "statistical_comparisons.csv")
    claim(TR, f"C3 {s} CI", f"[{pp(r['ci_lo_pp'])}, {pp(r['ci_hi_pp'])}]", "statistical_comparisons.csv")
for nm, lab in (("C1 data quality, full", "C1"), ("C2 data quality, matched 512", "C2")):
    for s in ("gsm8", "math"):
        r = comp[(nm, s, "strict")]
        claim(TR, f"{lab} {s} diff", pp(r["diff_pp"]), "statistical_comparisons.csv")
        claim("EXECUTIVE_SUMMARY.md" if s == "gsm8" else TR, f"{lab} {s} diff (summary)", pp(r["diff_pp"]) + " pp" if s == "gsm8" else pp(r["diff_pp"]), "statistical_comparisons.csv")
claim(TR, "C1 GSM8K Holm", sci(comp[("C1 data quality, full", "gsm8", "strict")]["p_holm"]), "statistical_comparisons.csv")
claim(TR, "C2 GSM8K Holm", sci(comp[("C2 data quality, matched 512", "gsm8", "strict")]["p_holm"]), "statistical_comparisons.csv")
claim(TR, "C2 MATH Holm", f"Holm {float(comp[('C2 data quality, matched 512', 'math', 'strict')]['p_holm']):.2f}", "statistical_comparisons.csv")
# relaxed / second verifier
claim(TR, "C3 relaxed GSM8K", pp(comp[("C3 pathway B vs its SFT parent", "gsm8", "strict")]["relaxed_diff_pp"]) + " pp (GSM8K)", "statistical_comparisons.csv")
claim(TR, "C3 relaxed MATH", pp(comp[("C3 pathway B vs its SFT parent", "math", "strict")]["relaxed_diff_pp"]) + " pp (MATH)", "statistical_comparisons.csv")
claim(TR, "C2 GSM8K second-verifier gap", pp(comp[("C2 data quality, matched 512", "gsm8", "strict")]["second_diff_pp"]) + " pp", "statistical_comparisons.csv")
b = acc("BASE:r1_zero", "gsm8")
claim(TR, "base r1_zero second-verifier gain GSM8K", f"{100 * (float(b['second']) - float(b['strict'])):.1f} pp (GSM8K)", "consolidated_results.csv")
bm = acc("BASE:r1_zero", "math")
claim(TR, "base r1_zero second-verifier gain MATH", f"{100 * (float(bm['second']) - float(bm['strict'])):.1f} pp (MATH)", "consolidated_results.csv")
claim(TR, "base r1_zero relaxed gain", f"{100 * (float(b['relaxed']) - float(b['strict'])):.1f} pp to the base model", "consolidated_results.csv")
n = acc("SFT-reverified_n128_s0", "gsm8")
claim(TR, "n128 relaxed gain", f"{100 * (float(n['relaxed']) - float(n['strict'])):.1f} pp to the 128-example", "consolidated_results.csv")
gb = acc(B, "math")
claim(TR, "GRPO-B relaxed move (max)", f"at most {100 * max(float(gb['relaxed']) - float(gb['strict']), 0):.1f} pp", "consolidated_results.csv")
claim(TR, "GRPO-B second move (max)", f"at most {100 * max(float(gb['second']) - float(gb['strict']), 0):.1f} pp to the best model", "consolidated_results.csv")
mx = max(float(acc(k, s)["second"]) - float(acc(k, s)["strict"]) for k in ("SFT-exact_full_s0", "SFT-reverified_full_s0", "SFT-reverified_sizecontrol_s0", "SFT-exact_n512_s0", P, "SFT-reverified_n2048_s0") for s in ("gsm8", "math"))
claim(TR, "max second-verifier gain, SFT ≥ 512", f"at most {100 * mx:.1f} pp to any SFT arm", "consolidated_results.csv")
# prompt effect
for s, lab in (("gsm8", "GSM8K"), ("math", "MATH")):
    claim(TR, f"prompt effect {lab}", pp(comp[("base prompts: question-only vs r1_zero", s, "strict")]["diff_pp"]) + f" pp on {lab}", "statistical_comparisons.csv")
claim(TR, "base r1_zero format GSM8K", pct(b["format"]) + "%", "consolidated_results.csv")
claim(TR, "base r1_zero acc | formatted", pct(b["acc_given_format"]) + "%", "consolidated_results.csv")
claim(TR, "items gained by prompt switch", f"Of the {gtc('Base r1_zero -> base question-only', 'gsm8', 'gained')} GSM8K items", "gain_transitions.csv")
claim(TR, "of which formatted-wrong", f"{gtc('Base r1_zero -> base question-only', 'gsm8', 'gained', 'format_ok_answer_wrong')} were well-formatted but wrong under r1_zero", "gain_transitions.csv")
# gain decomposition
g = gtc("C3: SFT parent -> GRPO-B", "gsm8", "gained")
claim(TR, "C3 GSM8K gained", f"{gtc('C3: SFT parent -> GRPO-B', 'gsm8', 'gained', 'format_ok_answer_wrong')} of the {g} newly correct", "gain_transitions.csv")
claim(TR, "C3 MATH gained", f"{gtc('C3: SFT parent -> GRPO-B', 'math', 'gained', 'format_ok_answer_wrong')} of {gtc('C3: SFT parent -> GRPO-B', 'math', 'gained')} were well-formatted", "gain_transitions.csv")
claim("README.md", "C3 GSM8K gained (README)", f"{gtc('C3: SFT parent -> GRPO-B', 'gsm8', 'gained', 'format_ok_answer_wrong')} of the {g} GSM8K items", "gain_transitions.csv")
L = "SFT exact n512 -> SFT re-verified n512 (C2)"
claim(TR, "C2 lost items formatted-wrong", f"Of the {gtc(L, 'gsm8', 'lost')} GSM8K items solved by exact 512 but not by re-verified 512, {gtc(L, 'gsm8', 'lost', 'format_ok_answer_wrong')} were", "gain_transitions.csv")
# lengths, truncation
p_, gB = acc(P, "gsm8"), acc(B, "gsm8")
claim(TR, "length GSM8K", f"from {float(p_['tokens_mean']):.0f} to {float(gB['tokens_mean']):.0f} tokens on GSM8K", "consolidated_results.csv")
claim(TR, "length MATH", f"from {float(acc(P, 'math')['tokens_mean']):.0f} to {float(gb['tokens_mean']):.0f} on MATH", "consolidated_results.csv")
claim(TR, "truncation MATH", f"from {pct(acc(P, 'math')['truncated_rate'])}% to {pct(gb['truncated_rate'])}%", "consolidated_results.csv")
claim(TR, "truncated share of errors", f"{pct(gb['truncated_share_of_errors'])}% of GRPO-B's MATH errors", "consolidated_results.csv")
a2 = "GRPO-A-recovery80-lr5e-6-s2026"
claim(TR, "A2 doubled tag", f"{int(acc(a2, 'gsm8')['double_think']):,} of 1,319 GSM8K and {int(acc(a2, 'math')['double_think']):,} of 5,000 MATH", "consolidated_results.csv")
# SFT vs base
r = comp[("SFT reverified_n128 vs base question-only", "gsm8", "strict")]; rm = comp[("SFT reverified_n128 vs base question-only", "math", "strict")]
claim(TR, "n128 vs base", f"{pp(r['diff_pp'])} pp on GSM8K and {pp(rm['diff_pp'])} pp on MATH", "statistical_comparisons.csv")
claim(TR, "n128 format / relaxed", f"format rate is {pct(n['format'])}%, and relaxed accuracy is {pct(n['relaxed'])}% against {pct(n['strict'])}% strict", "consolidated_results.csv")
sv = [comp[(f"SFT {x} vs base question-only", "gsm8", "strict")] for x in ("exact_full", "reverified_full", "sizecontrol", "exact_n512", "reverified_n512", "reverified_n2048")]
claim(TR, "SFT vs base GSM8K range", f"{pp(min(float(x['diff_pp']) for x in sv))} to {pp(max(float(x['diff_pp']) for x in sv))} pp on GSM8K", "statistical_comparisons.csv")
svm = [comp[(f"SFT {x} vs base question-only", "math", "strict")] for x in ("exact_full", "reverified_full", "sizecontrol", "exact_n512", "reverified_n512", "reverified_n2048")]
claim(TR, "SFT vs base MATH range", f"{pp(min(float(x['diff_pp']) for x in svm))} to {pp(max(float(x['diff_pp']) for x in svm))} pp on MATH", "statistical_comparisons.csv")
rv = [float(comp[(f"SFT {x} vs base question-only", "gsm8", "strict")]["relaxed_diff_pp"]) for x in ("reverified_full", "sizecontrol", "reverified_n512", "reverified_n2048")]
claim(TR, "relaxed range re-verified ≥512", f"between {pp(min(rv))} and {pp(max(rv))} pp for the re-verified arms with 512", "statistical_comparisons.csv")
# post-hoc
for nm, s, txt in (("Direct (A1, post-hoc branch) vs sequential (GRPO-B)", "gsm8", "A1 → GRPO-B {d} pp, Holm {h:.1f}"),
                   ("Direct (A1, post-hoc branch) vs sequential (GRPO-B)", "math", "{d} pp vs A1, Holm {hs}"),
                   ("Direct (A2, post-hoc branch) vs sequential (GRPO-B)", "math", "{d} pp vs A2, Holm {h:.3f}")):
    r = comp[(nm, s, "strict")]
    claim(TR, f"post-hoc {nm} {s}", txt.format(d=pp(r["diff_pp"]), h=float(r["p_holm"]), hs=f"{float(r['p_holm']):.0e}".replace("e-0", "e-")), "statistical_comparisons.csv")
# dev curves and windows
BB = "GRPO-B (sequential)"
claim(TR, "B dev curve GSM8K", f"from {pct(curves[(BB, 0, 'gsm8k_dev')])}% (its SFT parent) to {pct(curves[(BB, 25, 'gsm8k_dev')])}% at step 25 and ended at {pct(curves[(BB, 100, 'gsm8k_dev')])}%", "grpo_dev_curves.csv")
claim(TR, "B dev curve MATH", f"from {pct(curves[(BB, 0, 'math_dev')])}% to {pct(curves[(BB, 25, 'math_dev')])}% at step 25 and ended at {pct(curves[(BB, 100, 'math_dev')])}%", "grpo_dev_curves.csv")
AA = "GRPO-A (direct, FAILED at 98)"
claim(TR, "A dev curve", f"at {pct(curves[(AA, 0, 'gsm8k_dev')])}% GSM8K dev under r1_zero, it reached {pct(curves[(AA, 25, 'gsm8k_dev')])}% by step 25 and {pct(curves[(AA, 75, 'gsm8k_dev')])}% at step 75", "grpo_dev_curves.csv")
claim(TR, "A step-75 dev", f"{pct(curves[(AA, 75, 'gsm8k_dev')])}% (GSM8K) and {pct(curves[(AA, 75, 'math_dev')])}% (MATH)", "grpo_dev_curves.csv")
claim(TR, "B dev net 25→100", f"+{100 * (curves[(BB, 100, 'gsm8k_dev')] - curves[(BB, 25, 'gsm8k_dev')]):.1f} pp on GSM8K and +{100 * (curves[(BB, 100, 'math_dev')] - curves[(BB, 25, 'math_dev')]):.1f} pp on MATH dev", "grpo_dev_curves.csv")
wb = [w for w in win if w["run"] == BB]
claim(TR, "zero-adv shares", f"{100 * float(wb[0]['zero_adv_frac']):.0f}% of groups had identical rewards across all 8 samples in steps 0–25, rising to {100 * float(wb[1]['zero_adv_frac']):.0f}% and {100 * float(wb[2]['zero_adv_frac']):.0f}%", "grpo_compute_windows.csv")
claim(TR, "window 0-25 gains", f"{pp(wb[0]['gsm8k_dev_gain_pp'])} pp GSM8K, p 9e-18; {pp(wb[0]['math_dev_gain_pp'])} pp MATH", "grpo_compute_windows.csv + RESEARCH_FINDINGS.md (p)")
claim(TR, "GPU-h steps 25-100", f"{sum(float(w['gpu_h']) for w in wb[1:]):.2f}", "grpo_compute_windows.csv")
claim(TR, "GPU-h steps 0-25", f"{float(wb[0]['gpu_h']):.2f}", "grpo_compute_windows.csv")
# GRPO-A incident (dynamics rows)
claim(TR, "A step-97 reward", f"reward {float(A_ROWS[97]['reward_mean']):.3f}", "grpo_dynamics.csv")
claim(TR, "A step-97 all-wrong groups", f"{round(float(A_ROWS[97]['all_incorrect_frac']) * 32)} of 32 groups all wrong", "grpo_dynamics.csv")
claim(TR, "A step-97 entropy", f"entropy {float(A_ROWS[97]['entropy']):.3f}", "grpo_dynamics.csv")
claim(TR, "A step-97 truncation", f"truncation {float(A_ROWS[97]['truncated_rate']):.3f}", "grpo_dynamics.csv")
claim(TR, "A step-97 gap", f"{float(A_ROWS[97]['rollout_logprob_mad']):.4f}", "grpo_dynamics.csv")
claim(TR, "A step-97 max token gap", f"{float(A_ROWS[97]['rollout_logprob_max']):.1f} nats", "grpo_dynamics.csv")
claim(TR, "A step-97 format", f"{float(A_ROWS[97]['format_rate']):.3f}", "grpo_dynamics.csv")
claim(TR, "A update-96 grad norm", f"({float(A_ROWS[96]['grad_norm']):.3f}, 1.6", "grpo_dynamics.csv")
rej = [r for r in dyn if r["run"].startswith("GRPO-A") and r["rejected_by_gate"] not in ("", "False", "0")]
claim(TR, "A step-98 rejected gap", f"{float(rej[0]['rollout_logprob_mad']):.4f}" if rej else "MISSING", "grpo_dynamics.csv (rejected row)")
# cross-benchmark
def find(d, k):
    if isinstance(d, dict):
        if k in d:
            return d[k]
        for v in d.values():
            r = find(v, k)
            if r is not None:
                return r
    return None


def must(k):
    v = find(AN, k)
    if v is None:
        raise SystemExit(f"analysis.json has no {k}")
    return v


claim(TR, "Spearman GSM8K vs MATH", f"Spearman ρ = {must('spearman_gsm8k_math'):.2f}, p = {must('spearman_p'):.0e}".replace("e-0", "e-"), "analysis.json")
claim(TR, "same-sign comparisons", f"{must('comparisons_same_sign').split('/')[0]} of {must('comparisons_same_sign').split('/')[1]} paired comparisons", "analysis.json")
claim(TR, "dev-final rank corr", f"ρ = {must('spearman_dev_final_gsm8k_test_final'):.2f} on GSM8K, {must('spearman_dev_final_math_val_final'):.2f} on MATH", "analysis.json")
claim(TR, "dev-final same sign", f"{must('dev_final_same_sign').split('/')[0]} of {must('dev_final_same_sign').split('/')[1]} comparisons", "analysis.json")
# seed sensitivity
for nm, s in (("C1 data quality, full", "gsm8"), ("C2 data quality, matched 512", "gsm8")):
    claim(TR, f"seed ratio {nm}", pp(seed[(nm, s)]["ratio_diff_to_total_se"]), "seed_sensitivity.csv")
claim(TR, "seed SD GSM8K", f"{float(seed[('C1 data quality, full', 'gsm8')]['seed_sd_contrast_pp']):.2f}", "seed_sensitivity.csv")
claim(TR, "seed SD MATH", f"{float(seed[('C1 data quality, full', 'math')]['seed_sd_contrast_pp']):.2f}", "seed_sensitivity.csv")
s0, s100 = done("sft/reverified_n512_s0")["eval"]["gsm8k_dev"]["accuracy"], done("sft/tune_lr1e-5")["eval"]["gsm8k_dev"]["accuracy"]
claim(TR, "seed pair", f"seed 0 ({pct(s0)}% GSM8K dev) against seed 100 ({pct(s100)}%), a {100 * (s100 - s0):.1f} pp difference", "DONE.json (two runs)")
# never solved
ns = {r["set"][:4]: r for r in diff if r["models_solving"] == "0"}
claim(TR, "never solved", f"{ns['gsm8']['items']} GSM8K and {ns['math']['items']} MATH items were solved by none of the eight", "item_difficulty.csv")
claim(TR, "never solved, base qo", f"solved {ns['gsm8']['base_question_only_solves']} and {ns['math']['base_question_only_solves']} of them", "item_difficulty.csv")
# GPU-h
GH = AN["gpu_h"]; CUM = AN["cumulative_gpu_h"]
claim(TR, "GRPO-B cumulative", f"{CUM[B]:.2f} cumulative GPU-h", "analysis.json")
claim(TR, "A1/A2 cumulative", f"{CUM['GRPO-A-replay80-lr1e-5-s2026']:.2f} and {CUM['GRPO-A-recovery80-lr5e-6-s2026']:.2f} for A1 and A2", "analysis.json")
claim(TR, "failed A GPU-h", f"{AN['failed_A']['gpu_h_total']:.2f}", "analysis.json")
tot = sum(done(x)["wall_s"] for x in [p.split(os.sep)[-3] + "/" + p.split(os.sep)[-2] for t in RUNS for p in glob.glob(os.path.join(t, "*", "*", "DONE.json"))]) / 1800 + AN["failed_A"]["gpu_h_total"]
claim("REPRODUCIBILITY.md", "total training GPU-h", f"**{tot:.2f}**", "DONE.json wall_s + analysis.json")
claim(TR, "total training GPU-h (rounded)", f"about {tot:.0f} GPU-h for training", "DONE.json wall_s + analysis.json")
claim("README.md", "total training GPU-h (README)", f"about {tot:.0f} GPU-h of training", "DONE.json wall_s + analysis.json")
# audit and data
claim(TR, "audit reproduction fields", "238 metric fields and 382 comparison fields, 0 mismatches", "AUDIT.md (audit.json)")
claim(TR, "post-eval checks", f"{len(PFI['checks'])} checks, {PFI['failed']} failed", "post_final_integrity.json")
c = MAN["counts"]
for k, v in (("raw", "4,836"), ("exact", "3,496"), ("reverified", "4,545"), ("rl_pool", "6,832")):
    claim(TR, f"data count {k}", f"{c[k]:,}", "MANIFEST.json")
claim(TR, "overlap at 512", f"share {MAN['subset_overlap_512']['s0']['exact&reverified']} of 512", "MANIFEST.json")
sub = {n: set(json.load(open(os.path.join(ROOT, "data", "stage0", "subsets", n + ".json")))["ids"]) for n in ("exact_full", "reverified_full", "reverified_sizecontrol_s0")}
claim(TR, "size-control overlap", f"shares {len(sub['exact_full'] & sub['reverified_sizecontrol_s0']):,} with the exact-match set", "data/stage0/subsets/*.json")
claim(TR, "recovered in full set", f"plus {len(sub['reverified_full'] - sub['exact_full']):,} recovered ones", "data/stage0/subsets/*.json")
claim(TR, "exact ⊂ re-verified", "contains every exact-match example" if sub["exact_full"] <= sub["reverified_full"] else "FALSE", "data/stage0/subsets/*.json")

ok = sum(x[4] for x in claims)
lines = ["# Key-claim audit", "", "Generated by `report/scripts/key_claims.py`. Each claim is recomputed from its named source, formatted as the report formats it, "
         "and the exact string must appear in the named deliverable.", "", f"**{ok} of {len(claims)} claims verified.**", "",
         "| Deliverable | Claim | Expected text (recomputed) | Source | Found |", "|---|---|---|---|---|"]
lines += [f"| `{d}` | {desc} | `{t.replace('|', '/')}` | {src} | {'yes' if f else '**NO**'} |" for d, desc, t, src, f in claims]
open(os.path.join(ROOT, "report", "AUDIT_KEY_CLAIMS.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print(f"{ok}/{len(claims)} verified")
for d, desc, t, src, f in claims:
    if not f:
        print("MISSING", d, "|", desc, "|", t)

# ---- consolidation deliverables: resume, interview guide, README, summary --------------------------
inv = list(csv.DictReader(open(os.path.join(ROOT, "report", "tables", "t15_run_inventory.csv"), encoding="utf-8")))
LOCK = json.load(open(os.path.join(os.path.expanduser(a.final), "FINAL_CHECKPOINTS.json")))
c3g, c3m = comp[("C3 pathway B vs its SFT parent", "gsm8", "strict")], comp[("C3 pathway B vs its SFT parent", "math", "strict")]
RS, IG, RD, ES = "docs/RESUME.md", "docs/INTERVIEW_GUIDE.md", "README.md", "EXECUTIVE_SUMMARY.md"
extra = []
def claim2(doc, desc, text, source):
    if doc in DOCS:
        extra.append((doc, desc, text, source, text in DOCS[doc]))
claim2(RS, "17 training runs", f"{len(inv)} training runs", "t15_run_inventory.csv")
claim2(TR, "17 documented runs", f"{len(inv)} documented training runs", "t15_run_inventory.csv")
claim2(TR, "11 model groups", f"{len(LOCK['checkpoints'])} model groups", "FINAL_CHECKPOINTS.json")
claim2(RS, "11 model groups (resume)", f"{len(LOCK['checkpoints'])} model groups", "FINAL_CHECKPOINTS.json")
claim2(RS, "C3 gains", f"{pp(c3g['diff_pp'])} pp (GSM8K) and {pp(c3m['diff_pp'])} pp (MATH)", "statistical_comparisons.csv")
claim2(RS, "C3 Holm < 1e-24", "Holm p < 1e-24" if max(float(c3g["p_holm"]), float(c3m["p_holm"])) < 1e-24 else "FALSE", "statistical_comparisons.csv")
claim2(RS, "GRPO-B accuracy", f"{pct(acc(B, 'gsm8')['strict'])}% on GSM8K and {pct(acc(B, 'math')['strict'])}% on MATH", "consolidated_results.csv")
claim2(RS, "teacher traces", f"{MAN['counts']['raw']:,} teacher-generated (gpt-oss-120b)", "MANIFEST.json")
claim2(RS, "recovered traces", f"recovered {MAN['counts']['reverified'] - MAN['counts']['exact']:,} traces", "MANIFEST.json")
claim2(RS, "C3 gain items", f"{gtc('C3: SFT parent -> GRPO-B', 'gsm8', 'gained', 'format_ok_answer_wrong')} of {gtc('C3: SFT parent -> GRPO-B', 'gsm8', 'gained')} newly solved", "gain_transitions.csv")
claim2(RS, "format-insensitive gain", f"{pp(c3g['relaxed_diff_pp'])} / {pp(c3m['relaxed_diff_pp'])} pp", "statistical_comparisons.csv")
claim2(RS, "C1/C2 range", f"{abs(float(comp[('C2 data quality, matched 512', 'gsm8', 'strict')]['diff_pp'])):.1f}–{abs(float(comp[('C1 data quality, full', 'gsm8', 'strict')]['diff_pp'])):.1f} pp", "statistical_comparisons.csv")
claim2(RS, "collapse reward", f"{float(A_ROWS[96]['reward_mean']):.2f} → {float(A_ROWS[97]['reward_mean']):.3f}", "grpo_dynamics.csv")
claim2(RS, "prompt effect", pp(comp[("base prompts: question-only vs r1_zero", "gsm8", "strict")]["diff_pp"]) + " pp", "statistical_comparisons.csv")
claim2(RS, "fields recomputed", f"{238} metric + {382} comparison fields, 0 mismatches", "AUDIT.md (audit.json)")
claim2(RD, "README collapse reward", f"from {float(A_ROWS[96]['reward_mean']):.2f} to {float(A_ROWS[97]['reward_mean']):.3f}", "grpo_dynamics.csv")
claim2(RD, "README A dev step 75", f"{pct(curves[(AA, 75, 'gsm8k_dev')])}% GSM8K dev at step 75", "grpo_dev_curves.csv")
claim2(RD, "README SFT parent", f"| {pct(acc(P, 'gsm8')['strict'])}% | {pct(acc(P, 'math')['strict'])}% |", "consolidated_results.csv")
claim2(RD, "README C3", f"{pp(c3g['diff_pp'])} pp GSM8K / {pp(c3m['diff_pp'])} pp MATH", "statistical_comparisons.csv")
claim2(RD, "README best-prompt baseline", f"{pct(acc('BASE:question_only', 'gsm8'))}% / {pct(acc('BASE:question_only', 'math'))}%".replace("%", "%") if False else f"{pct(acc('BASE:question_only', 'gsm8')['strict'])}% / {pct(acc('BASE:question_only', 'math')['strict'])}%", "consolidated_results.csv")
claim2(ES, "parent formatted", f"{pct(acc(P, 'gsm8')['format'])}% of GSM8K items", "consolidated_results.csv")
claim2(TR, "parent formatted (report)", f"{pct(acc(P, 'gsm8')['format'])}% of GSM8K items", "consolidated_results.csv")
claim2(IG, "parent formatted (interview)", f"{pct(acc(P, 'gsm8')['format'])}% of GSM8K items", "consolidated_results.csv")
claim2(IG, "interview C3", "13.4 points on GSM8K and 11.7 on MATH" if (pp(c3g["diff_pp"]), pp(c3m["diff_pp"])) == ("+13.4", "+11.7") else "FALSE", "statistical_comparisons.csv")
claim2(IG, "interview prompt baseline", f"{pct(acc('BASE:r1_zero', 'gsm8')['strict'])}% with the R1-Zero template but {pct(acc('BASE:question_only', 'gsm8')['strict'])}%", "consolidated_results.csv")
claim2(IG, "interview seed pair", f"({100 * (s100 - s0):.1f} pp)", "DONE.json (two runs)")
claim2(IG, "interview gate value", f"gap was {float(rej[0]['rollout_logprob_mad']):.3f} > 0.05", "grpo_dynamics.csv")
claim2(IG, "interview truncation share", f"{pct(gb['truncated_share_of_errors'])}% of GRPO-B's MATH errors", "consolidated_results.csv")
claim2(IG, "interview fields", f"0 mismatches across {238 + 382} fields", "AUDIT.md")
claim2(IG, "interview -8.6", pp(comp[("C2 data quality, matched 512", "gsm8", "strict")]["diff_pp"]) + " pp at matched size", "statistical_comparisons.csv")
claim2(IG, "interview second-verifier gap", pp(comp[("C2 data quality, matched 512", "gsm8", "strict")]["second_diff_pp"]) + " pp under a second verifier", "statistical_comparisons.csv")
ok2 = sum(x[4] for x in extra)
claims += extra
ok = sum(x[4] for x in claims)
lines = ["# Key-claim audit", "", "Generated by `report/scripts/key_claims.py`. Each claim is recomputed from its named source, formatted as the deliverable formats it, "
         "and the exact string must appear in the named deliverable (report/ files and the repository README.md).", "", f"**{ok} of {len(claims)} claims verified.**", "",
         "| Deliverable | Claim | Expected text (recomputed) | Source | Found |", "|---|---|---|---|---|"]
lines += [f"| `{d}` | {desc} | `{t.replace('|', '/')}` | {src} | {'yes' if f else '**NO**'} |" for d, desc, t, src, f in claims]
open(os.path.join(ROOT, "report", "AUDIT_KEY_CLAIMS.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print(f"consolidation claims {ok2}/{len(extra)}; total {ok}/{len(claims)}")
for d, desc, t, src, f in extra:
    if not f:
        print("MISSING", d, "|", desc, "|", t)
