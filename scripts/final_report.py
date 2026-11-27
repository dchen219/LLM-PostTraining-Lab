"""Held-out (final) report for the checkpoints frozen in final/freeze_v1.yaml. Reads only. Run on the laptop after recovery.

python scripts/final_report.py --final ~/a5_final --runs ~/a5_sessions --runs ~/a5_recovery --out results/final_eval
Refuses unless the recovered FINAL_CHECKPOINTS.json was written for this exact freeze file and every evaluation's weights equal the lock.
Per checkpoint, prompt and set: strict accuracy with Wilson 95% CI, second-verifier and relaxed accuracy, format and clean-format rates, doubled
</think>, response length (all / correct / incorrect), truncation, failure categories. Comparisons: paired difference (b minus a) with 95% CI,
exact McNemar, Holm within each frozen family and set. Efficiency: training GPU-hours and evaluation throughput.
Labels: pre-registered confirmatory, pre-registered secondary, descriptive, post-hoc exploratory. "Supported" means Holm p < alpha in a
pre-registered family; GSM8K differences below the frozen detectable size are reported as not detected whatever their p.
"""
import argparse, json, logging, os, statistics as st, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
logging.disable(logging.CRITICAL)
from posttrain.common import jload, sha_file  # noqa: E402
from posttrain.final_eval import FREEZE, load_freeze, validate  # noqa: E402
from posttrain.stats import holm, paired, wilson  # noqa: E402

LABEL = dict(confirmatory="pre-registered confirmatory", sft_vs_base="pre-registered secondary", descriptive="descriptive", posthoc="post-hoc exploratory")
SETS = ("gsm8k_test_final", "math_val_final")


def build(final_dir, trees, freeze_path=FREEZE, second=True):
    fz, fsha = load_freeze(freeze_path)
    bad = validate(fz)
    if bad:
        raise SystemExit("freeze invalid: " + "; ".join(bad))
    lk = json.load(open(os.path.join(final_dir, "FINAL_CHECKPOINTS.json"), encoding="utf-8"))
    if lk["freeze_sha256"] != fsha:
        raise SystemExit("REFUSED: the lock was written for a different freeze file")
    A = None
    if second:
        import stage0_verify_audit as A  # noqa: F811
    ans = lambda t: t.split("<answer>")[-1].replace("</answer>", "").strip()
    items, metrics, missing, problems = {}, [], [], []
    for c in fz["checkpoints"]:
        d = os.path.join(final_dir, c["id"])
        if not os.path.exists(os.path.join(d, "DONE.json")):
            missing.append(c["id"])
            continue
        summ = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
        if summ["parent"]["weights_sha256"] != lk["checkpoints"][c["id"]]["weights_sha256"]:
            problems.append(f"{c['id']}: evaluated weights differ from the lock")
            continue
        gpu_h = None
        if c["dir"]:
            for t in trees:
                p = os.path.join(t, c["dir"], "DONE.json")
                if os.path.exists(p):
                    gpu_h = 2 * json.load(open(p, encoding="utf-8"))["wall_s"] / 3600
                    break
        for pr in c["prompts"]:
            for s in SETS:
                x = jload(os.path.join(d, "eval", f"{pr}_{s}.jsonl"))
                n, k = len(x), sum(bool(y["correct"]) for y in x)
                if k != summ["results"][pr][s]["correct"] or n != summ["results"][pr][s]["n"]:
                    problems.append(f"{c['id']} {pr} {s}: items disagree with summary.json")
                fmt_ok = lambda y: bool(y["format"]) if pr == "r1_zero" else True
                strict = {y["id"]: bool(y["correct"]) for y in x}
                sec = {y["id"]: strict[y["id"]] or bool(A is not None and pr == "r1_zero" and y["format"] and A.second_verifier(ans(y["response"]), str(y["answer"]))) for y in x}
                relx = {y["id"]: sec[y["id"]] or y["category"] == "format_fail_answer_right" for y in x}
                items[(c["id"], pr, s)] = (strict, sec, relx)
                tok = [y["n_tokens"] for y in x]
                cor, inc = [y["n_tokens"] for y in x if y["correct"]], [y["n_tokens"] for y in x if not y["correct"]]
                g = summ["results"][pr][s]
                metrics.append(dict(id=c["id"], cls=c["class"], prompt=pr, set=s, n=n, strict=k / n, ci95=wilson(k, n), second=sum(sec.values()) / n if A else None,
                                    relaxed=sum(relx.values()) / n, format=st.mean(fmt_ok(y) for y in x), clean_format=st.mean(bool(y.get("clean_format")) for y in x) if pr == "r1_zero" else None,
                                    double_think=sum(y["response"].count("</think>") >= 2 for y in x), tokens_mean=st.mean(tok), tokens_median=st.median(tok),
                                    tokens_p90=sorted(tok)[int(0.9 * n)], tokens_correct=st.mean(cor) if cor else None, tokens_incorrect=st.mean(inc) if inc else None,
                                    truncated=sum(y["finish"] == "length" for y in x), truncated_rate=sum(y["finish"] == "length" for y in x) / n,
                                    verifier_errors=sum(bool(y["error"]) for y in x), categories=g.get("categories"), gen_s=g.get("gen_s"),
                                    tok_per_s=sum(tok) / g["gen_s"] if g.get("gen_s") else None, train_gpu_h=gpu_h))
    comps = []
    for fam, f in fz["families"].items():
        for s in SETS:
            grp = []
            for cmp in f["comparisons"]:
                ka, kb = [tuple(cmp[x].split(":")) + (s,) for x in ("a", "b")]
                row = dict(family=fam, label=LABEL[fam], name=cmp["name"], set=s, a=cmp["a"], b=cmp["b"])
                if cmp.get("unavailable"):
                    row.update(unavailable=cmp["unavailable"], p=1.0)
                elif ka in items and kb in items:
                    r = paired(items[ka][0], items[kb][0])
                    row.update(r, relaxed_diff=paired(items[ka][2], items[kb][2])["diff"], second_diff=paired(items[ka][1], items[kb][1])["diff"])
                else:
                    row.update(unavailable="not evaluated (missing)", p=None)
                grp.append(row)
            if f["holm_m"]:
                ps = {g["name"]: (g["p"] if g["p"] is not None else 1.0) for g in grp}
                h = holm(ps)
                for g in grp:
                    g["p_holm"] = h[g["name"]]
            for g in grp:
                adj = g.get("p_holm", g.get("p"))
                big = g.get("diff") is not None and (s != "gsm8k_test_final" or abs(100 * g["diff"]) >= fz["thresholds"]["detectable_pp"])
                if g.get("diff") is None:
                    g["verdict"] = "unavailable"
                elif fam in ("confirmatory", "sft_vs_base"):
                    g["verdict"] = "supported" if adj < fz["thresholds"]["alpha"] and big else "not detected"
                else:
                    g["verdict"] = ("descriptive" if fam == "descriptive" else "exploratory") + (" (p < alpha)" if adj is not None and adj < fz["thresholds"]["alpha"] else "")
            comps += grp
    return dict(freeze_sha256=fsha, lock=lk, missing=missing, problems=problems, metrics=metrics, comparisons=comps)


def render(R):
    pc = lambda v: "-" if v is None else f"{100 * v:.1f}"
    L = ["# Final held-out evaluation (generated)", "", f"Freeze sha256 {R['freeze_sha256']}. Not evaluated (missing): {', '.join(R['missing']) or 'none'}.",
         f"Problems: {'; '.join(R['problems']) or 'none'}.", "", "GRPO-A_base_s2026 (pre-registered) FAILED at step 98 and has no final checkpoint; comparison C4 is unavailable and counted with p = 1. "
         "Rows marked post-hoc are diagnostic branches started from its step-80 state; they do not replace it.", "",
         "| checkpoint | class | prompt | set | strict % (95% CI) | second | relaxed | format | clean | doubled </think> | tokens mean/median/p90 | correct/incorrect tokens | truncated | train GPU-h | tok/s |", "|" + "---|" * 15]
    for m in R["metrics"]:
        L.append(f"| {m['id']} | {m['cls']} | {m['prompt']} | {m['set']} | {pc(m['strict'])} ({pc(m['ci95'][0])}-{pc(m['ci95'][1])}) | {pc(m['second'])} | {pc(m['relaxed'])} | {pc(m['format'])} | {pc(m['clean_format'])} | {m['double_think']} | "
                 f"{m['tokens_mean']:.0f}/{m['tokens_median']:.0f}/{m['tokens_p90']} | {'-' if m['tokens_correct'] is None else f'{m['tokens_correct']:.0f}'}/{'-' if m['tokens_incorrect'] is None else f'{m['tokens_incorrect']:.0f}'} | {m['truncated']} ({pc(m['truncated_rate'])}%) | "
                 f"{'-' if m['train_gpu_h'] is None else f'{m['train_gpu_h']:.2f}'} | {'-' if m['tok_per_s'] is None else f'{m['tok_per_s']:.0f}'} |")
    L += ["", "| family | comparison | set | a only | b only | diff pp (95% CI) | p | Holm p | second diff | relaxed diff | verdict |", "|" + "---|" * 11]
    for g in R["comparisons"]:
        if g.get("diff") is None:
            L.append(f"| {g['label']} | {g['name']} | {g['set']} | - | - | - | - | {'-' if g.get('p_holm') is None else f'{g['p_holm']:.3g}'} | - | - | {g['verdict']}: {g.get('unavailable', '')} |")
            continue
        L.append(f"| {g['label']} | {g['name']} | {g['set']} | {g['a_only']} | {g['b_only']} | {100 * g['diff']:+.1f} [{100 * g['ci95'][0]:+.1f}, {100 * g['ci95'][1]:+.1f}] | {g['p']:.3g} | "
                 f"{'-' if g.get('p_holm') is None else f'{g['p_holm']:.3g}'} | {100 * g['second_diff']:+.1f} | {100 * g['relaxed_diff']:+.1f} | {g['verdict']} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--runs", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert not os.path.exists(a.out), "refusing to overwrite"
    R = build(os.path.expanduser(a.final), [os.path.expanduser(t) for t in a.runs])
    os.makedirs(a.out)
    json.dump(R, open(a.out + "/report.json", "w"), indent=1, default=str)
    txt = render(R)
    open(a.out + "/REPORT.md", "w", encoding="utf-8").write(txt)
    print(txt)
