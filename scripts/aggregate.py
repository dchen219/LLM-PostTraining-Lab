"""Aggregate finished runs into results/summary/*.{csv,json,md}. Only runs with DONE.json are reported; every number carries its file.

python scripts/aggregate.py --runs $PT_RUNS --out results/summary [--compare RUN_A RUN_B ...]
"""
import argparse, csv, glob, json, os, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from posttrain.common import jload, sha_file
from posttrain.stats import holm, paired, wilson


def eval_items(run, tag, name):
    p = os.path.join(run, "eval", f"{tag}_{name}.jsonl")
    return ({r["id"]: bool(r["correct"]) for r in jload(p)}, p) if os.path.exists(p) else (None, p)


def collect(root):
    out, skipped = [], []
    for start in sorted(glob.glob(os.path.join(root, "*", "*", "run.json"))):
        run = os.path.dirname(start)
        if not os.path.exists(os.path.join(run, "DONE.json")):
            skipped.append(run)
            continue
        meta = json.load(open(start, encoding="utf-8"))
        done = json.load(open(os.path.join(run, "DONE.json"), encoding="utf-8"))
        ms = jload(os.path.join(run, "metrics.jsonl")) if os.path.exists(os.path.join(run, "metrics.jsonl")) else []
        chain = (done.get("lineage") or {}).get("chain", ["base"])
        if meta["kind"] == "eval":
            ev = {f"{p}:{k}": v for p, r in json.load(open(os.path.join(run, "summary.json"), encoding="utf-8"))["results"].items() for k, v in r.items()}
        else:
            ev = done.get("eval") or {}
        for name, v in ev.items():
            f = os.path.join(run, v["file"])
            items = jload(f)  # recompute from item-level records instead of trusting the summary
            k = sum(x["correct"] for x in items)
            assert k == v["correct"] and len(items) == v["n"], f"summary does not match items: {f}"
            lo, hi = wilson(k, len(items))
            out.append(dict(run=os.path.relpath(run, root), kind=meta["kind"], seed=meta["config"]["seed"], chain=">".join(chain), eval_set=name, n=len(items),
                            correct=k, accuracy=k / len(items), ci_lo=lo, ci_hi=hi, format_rate=v["format_rate"], mean_tokens=v["mean_tokens"],
                            truncated=v["truncated"], verifier_errors=v["verifier_errors"], wall_h=done["wall_s"] / 3600,
                            usd=max([m.get("usd") or 0 for m in ms] or [0]), pilot="pilot" in run,
                            peak_gib=max([v2 for m in ms for k2, v2 in m.items() if k2.endswith("_peak_gib")] or [0]),
                            config_sha=meta["config_sha256"][:12], code_sha=meta["code_sha256"][:12], item_file=os.path.relpath(f, root), item_sha=sha_file(f)[:12]))
    return out, skipped


def group_by_arm(rows):
    import re
    g = {}
    for r in rows:
        if not r["pilot"]:
            g.setdefault((re.sub(r"_s\d+$", "", r["run"]), r["eval_set"]), []).append(r)
    return [dict(arm=a, eval_set=e, seeds=len(v), mean_acc=st.mean(x["accuracy"] for x in v), sd_acc=st.stdev([x["accuracy"] for x in v]) if len(v) > 1 else None,
                 min_acc=min(x["accuracy"] for x in v), max_acc=max(x["accuracy"] for x in v), runs=";".join(x["run"] for x in v)) for (a, e), v in sorted(g.items())]


def wcsv(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", nargs="*", default=[], help="pairs run:tag run:tag on the same eval set, e.g. sft/x:final_step grpo/y:final_step")
    ap.add_argument("--set", default="gsm8k_dev")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, skipped = collect(a.runs)
    arms = group_by_arm(rows)
    wcsv(f"{a.out}/runs.csv", rows)
    wcsv(f"{a.out}/arms.csv", arms)
    cmp = {}
    for x, y in zip(a.compare[::2], a.compare[1::2]):
        (rx, tx), (ry, ty) = x.rsplit(":", 1), y.rsplit(":", 1)
        (ix, px), (iy, py) = eval_items(os.path.join(a.runs, rx), tx, a.set), eval_items(os.path.join(a.runs, ry), ty, a.set)
        if ix is None or iy is None:
            sys.exit(f"missing item file: {px if ix is None else py}")
        cmp[f"{x} vs {y} [{a.set}]"] = dict(paired(ix, iy), files=[px, py])
    adj = holm({k: v["p"] for k, v in cmp.items()})
    for k in cmp:
        cmp[k]["p_holm"] = adj[k]
    json.dump(cmp, open(f"{a.out}/comparisons.json", "w"), indent=1)
    with open(f"{a.out}/REPORT.md", "w", encoding="utf-8") as f:
        f.write("# Aggregated results\n\nOnly completed runs. Pilot runs are excluded from arm means. CIs are Wilson 95% over items; they do not include seed variance.\n\n")
        f.write("| Arm | Eval set | Seeds | Mean acc | SD | Range |\n|---|---|---|---|---|---|\n")
        for r in arms:
            sd = f"{100 * r['sd_acc']:.2f}" if r["sd_acc"] is not None else "single seed"
            f.write(f"| {r['arm']} | {r['eval_set']} | {r['seeds']} | {100 * r['mean_acc']:.2f}% | {sd} | {100 * r['min_acc']:.2f}-{100 * r['max_acc']:.2f}% |\n")
        f.write("\n## Paired comparisons (exact McNemar, Holm-adjusted)\n\n")
        for k, v in cmp.items():
            f.write(f"- {k}: diff {100 * v['diff']:+.2f} pp, 95% CI [{100 * v['ci95'][0]:+.2f}, {100 * v['ci95'][1]:+.2f}], p={v['p']:.4g}, Holm p={v['p_holm']:.4g}\n")
        f.write("\n## Incomplete runs (not reported)\n\n" + ("\n".join(f"- {s}" for s in skipped) or "none") + "\n")
    print(f"{len(rows)} eval rows, {len(arms)} arms, {len(cmp)} comparisons, {len(skipped)} incomplete -> {a.out}")


if __name__ == "__main__":
    main()
