"""Trace every number in the report deliverables to the original artifacts (read-only). Writes report/AUDIT_REPORT_NUMBERS.md.

A number in a report passes if it equals some value in a source file after the report's own rounding (same number of decimals),
allowing proportions shown as percentages, thousands separators, signs and scientific notation. Sources are split into
  data   machine-written artifacts: report.json, frozen CSV/JSON files, MANIFEST.json, run.json/DONE.json, incident analysis.json
  docs   numbers stated in original project documents (frozen analysis prose, plan, incident report), used only as a fallback
Numbers that match nothing are listed with their context and must be explained in EXPLAINED below (derived or definitional).

python report/scripts/audit_numbers.py --runs ~/a5_sessions --runs ~/a5_recovery --final ~/a5_final
"""
import argparse, csv, glob, json, math, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--runs", action="append", required=True)
ap.add_argument("--final", required=True)
a = ap.parse_args()
RUNS = [os.path.expanduser(r) for r in a.runs]
FIN = os.path.expanduser(a.final)
NUM = re.compile(r"(?<![\w.\-/#])([+\-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+\-−]?\d+(?:\.\d+)?(?:e[+\-−]?\d+)?)(?![\w.]*[A-Za-z_/])")


def numbers_in_text(t):
    out = []
    for m in NUM.finditer(t):
        s = m.group(1).replace(",", "").replace("−", "-").lstrip("+")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def walk(x, acc):
    if isinstance(x, dict):
        for v in x.values():
            walk(v, acc)
    elif isinstance(x, list):
        for v in x:
            walk(v, acc)
    elif isinstance(x, bool):
        return
    elif isinstance(x, (int, float)):
        if math.isfinite(x):
            acc.add(float(x))
    elif isinstance(x, str):
        try:
            acc.add(float(x))
        except ValueError:
            for v in numbers_in_text(x):
                acc.add(v)


data, docs, data_files, doc_files = set(), set(), [], []
J = [os.path.join(ROOT, p) for p in ("results/final_eval/report.json", "results/research_analysis/analysis.json", "results/research_analysis/audit.json",
                                    "results/research_analysis/post_final_integrity/post_final_integrity.json", "data/stage0/MANIFEST.json",
                                    "results/grpo_a_incident_analysis/analysis.json", "results/verifier_v2/crosscheck.json", "results/stage0/bf16_update_check.json")]
J += [os.path.join(FIN, "FINAL_CHECKPOINTS.json")]
for t in RUNS:
    J += glob.glob(os.path.join(t, "*", "*", "DONE.json")) + glob.glob(os.path.join(t, "*", "*", "run.json"))
for p in J:
    if os.path.exists(p):
        walk(json.load(open(p, encoding="utf-8")), data)
        data_files.append(p)
C = glob.glob(os.path.join(ROOT, "results", "research_analysis", "*.csv")) + glob.glob(os.path.join(ROOT, "results", "research_analysis", "data", "*.csv"))
C += glob.glob(os.path.join(ROOT, "results", "research_analysis", "figures", "data", "*.csv")) + glob.glob(os.path.join(ROOT, "report", "tables", "*.csv"))
for p in C:
    for row in csv.reader(open(p, encoding="utf-8")):
        for cell in row:
            walk(cell, data)
    data_files.append(p)
for p in [os.path.join(ROOT, "final", "freeze_v1.yaml"), os.path.join(ROOT, "requirements-cloud.lock.txt")]:
    walk(open(p, encoding="utf-8").read(), data)
    data_files.append(p)
D = ["results/research_analysis/RESEARCH_FINDINGS.md", "results/research_analysis/LIMITATIONS.md", "results/research_analysis/AUDIT.md",
     "results/research_analysis/EXECUTIVE_SUMMARY.md", "results/final_eval/REPORT.md", "results/grpo_a_incident_analysis/REPORT.md",
     "EXPERIMENT_PLAN.md", "GRPO_A_INCIDENT.md", "FINAL_EVAL_PLAN.md"]
for p in D:
    walk(open(os.path.join(ROOT, p), encoding="utf-8").read(), docs)
    doc_files.append(os.path.join(ROOT, p))


def decimals(tok):
    t = tok.replace(",", "")
    if "e" in t.lower():
        return None
    return len(t.split(".")[1]) if "." in t else 0


def matches(tok, pool):
    t = tok.replace(",", "").replace("−", "-").lstrip("+")
    x = float(t)
    if "e" in t.lower():  # p-values: compare at the shown precision
        mant, ex = t.lower().split("e")
        nd = len(mant.split(".")[1]) if "." in mant else 0
        for v in pool:
            if v > 0 and abs(x) > 0:
                if f"{v:.{nd}e}".replace("e-0", "e-").replace("e+0", "e+") == f"{float(mant):.{nd}f}e{int(ex)}".replace("e-0", "e-") or \
                   abs(math.log10(v) - math.log10(abs(x))) < 0.5 * 10 ** (-nd) / 2.3:
                    return True
        return False
    d = decimals(tok)
    for v in pool:
        for s in (1.0, 100.0, -1.0, -100.0):
            if abs(round(v * s, d) - x) < 10 ** (-d) * 0.5001:
                return True
    return False


STRIP = [(re.compile(r"`[^`]*`"), " "), (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "), (re.compile(r"\]\([^)]*\)"), "] "),
         (re.compile(r"\$\$.*?\$\$", re.S), " "), (re.compile(r"https?://\S+"), " "), (re.compile(r"\b[0-9a-f]{8}…[0-9a-f]{4}\b"), " ")]
TRIVIAL = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0}   # counts and list/section numbers are checked by eye

# Numbers that are definitional or arithmetic on sourced values, with the explanation shown in the audit file.
EXPLAINED = {
    "2026": "year / seed (configs)", "2025": "year (publication and release dates)",
    "34": "sum of own GPU-h over every training and tuning run (DONE.json wall_s × 2): 1.43 + 8.13 + 5.71 + 7.35 + 6.61 + 2.28 + 2.51 = 34.0",
    "30": "GRPO tuning steps (run.json grpo.steps)", "100": "GRPO steps / percent scale", "1024": "max new tokens / length normaliser (run.json)",
    "2064": "GRPO max_len (run.json)", "48": "GB per RTX A6000 (hardware spec)", "1.5": "model size 1.5B / about 1.5 h evaluation (AUDIT.md: 1.49 h)",
    "10100": "lines of Python in posttrain/, scripts/, tests/ (wc -l: 10,099)", "174": "pytest result: 174 passed (2026-11-27)",
    "25": "GRPO step / window boundary", "50": "window boundary", "75": "window boundary / step", "80": "saved step of GRPO-A", "81": "first branch step",
    "96": "collapse update (incident analysis)", "97": "completed updates of GRPO-A", "98": "attempted step rejected by the gate", "94": "early-warning step",
    "95": "plateau end step", "20": "steps / phase boundary", "32": "prompts per step / optimizer steps at n=512",
    "256": "rollouts per step (32 × 8)", "0.2": "clip ε (run.json)", "0.05": "rollout gate tolerance (run.json)", "0.6": "Stage 6 trigger: informative rate < 0.6",
    "1.96": "normal quantile for 95% CI", "1.4": "pre-registered detectable difference (freeze_v1.yaml)", "1.0": "lr selection margin 1.0 pp (EXPERIMENT_PLAN.md §10)",
    "128": "SFT subset size", "512": "SFT subset size", "2048": "SFT subset size",
    "8.13": "sum of DONE.json wall_s × 2 over the 7 SFT arms (key_claims.py recomputes the total)", "5.71": "sum over the 3 GRPO tuning runs",
    "4.79": "sum over A1 and A2 (2.28 + 2.51)", "34.01": "training and tuning total, recomputed in AUDIT_KEY_CLAIMS.md", "2.98": "2 × 1.49 h evaluation span (AUDIT.md)",
    "1.43": "sum over the 3 SFT tuning runs (0.37 + 0.44 + 0.62)",
}


def audit(path):
    text = open(path, encoding="utf-8").read()
    body = text
    for rx, rep in STRIP:
        body = rx.sub(rep, body)
    res = dict(data=0, docs=0, trivial=0, explained=0, unexplained=[], doc_only=[])
    for ln, line in enumerate(body.splitlines(), 1):
        if line.lstrip().startswith("#"):
            line = re.sub(r"^#+\s*[\d.]+\s", "", line.lstrip())
        for m in NUM.finditer(line):
            tok = m.group(1)
            x = float(tok.replace(",", "").replace("−", "-").lstrip("+"))
            key = tok.replace(",", "").replace("−", "-").lstrip("+-")
            if abs(x) in TRIVIAL and "." not in tok:
                res["trivial"] += 1
            elif matches(tok, data):
                res["data"] += 1
            elif matches(tok, docs):
                res["docs"] += 1
                res["doc_only"].append((ln, tok, line.strip()[:160]))
            elif key in EXPLAINED:
                res["explained"] += 1
            else:
                res["unexplained"].append((ln, tok, line.strip()[:160]))
    return res


TARGETS = ["report/TECHNICAL_REPORT.md", "report/EXECUTIVE_SUMMARY.md", "README.md", "report/REPRODUCIBILITY.md"]
out = ["# Number audit of the report deliverables", "",
       "Generated by `report/scripts/audit_numbers.py`. Every number token in each deliverable was compared with the original artifacts at the report's own rounding.",
       "", "**Limitation of this check:** it asks only whether a value with the same rounding exists somewhere in the sources, so a common number can match by coincidence. It catches invented or mistyped numbers that match nothing; it does not prove a number is attached to the right quantity. The specific check is `AUDIT_KEY_CLAIMS.md`, which recomputes each key claim from its named source.",
       "", f"Data sources: {len(data_files)} files ({len(data):,} distinct values). Document sources (fallback): {len(doc_files)} files.", "",
       "| Deliverable | Matched to data | Matched to documents only | Small integers (counts, list numbers) | Explained (definitional or derived) | Unexplained |", "|---|---|---|---|---|---|"]
details, doc_only = [], []
total_unexp = 0
for t in TARGETS:
    p = os.path.join(ROOT, t)
    if not os.path.exists(p):
        continue
    r = audit(p)
    total_unexp += len(r["unexplained"])
    out.append(f"| `{t}` | {r['data']} | {r['docs']} | {r['trivial']} | {r['explained']} | **{len(r['unexplained'])}** |")
    doc_only += [f"| `{t}`:{ln} | {tok} | {ctx.replace(chr(124), chr(47))} |" for ln, tok, ctx in r["doc_only"]]
    for ln, tok, ctx in r["unexplained"]:
        details.append(f"| `{t}`:{ln} | {tok} | {ctx.replace('|', '/')} |")
out += ["", "## Numbers matched only to project documents", "", "These are stated in original project documents (plan, incident report, frozen analysis prose) but not as a value in a machine-written file.", ""]
out += (["| Location | Number | Context |", "|---|---|---|"] + doc_only) if doc_only else ["None."]
out += ["", "## Unexplained numbers", ""]
out += (["| Location | Number | Context |", "|---|---|---|"] + details) if details else ["None."]
out += ["", "## Explained numbers (definitional or arithmetic on sourced values)", "", "| Number | Explanation |", "|---|---|"]
out += [f"| {k} | {v} |" for k, v in EXPLAINED.items()]
out += ["", "## Data sources", ""] + [f"- `{os.path.relpath(p, ROOT) if p.startswith(ROOT) else p.replace(os.path.expanduser('~'), '~')}`" for p in data_files if "run.json" not in p and "DONE.json" not in p]
out += [f"- run.json and DONE.json of {sum('DONE.json' in p for p in data_files)} finished training runs (`~/a5_sessions`, `~/a5_recovery`)", "", "## Document sources (fallback)", ""]
out += [f"- `{os.path.relpath(p, ROOT)}`" for p in doc_files]
open(os.path.join(ROOT, "report", "AUDIT_REPORT_NUMBERS.md"), "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("\n".join(out[6:6 + len(TARGETS) + 2]))
for d in details:
    print(d)
print("UNEXPLAINED", total_unexp)
