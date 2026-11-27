"""Stage 0: independently validate recovered teacher traces and audit verifier false positives.

CPU only. Reads the original datasets, never modifies them.
Usage: python scripts/stage0_verify_audit.py --src <dir with sft_gpt-oss-120b*.jsonl> --out <dir>
"""
import argparse, collections, hashlib, json, os, random, re, sys, unicodedata
from fractions import Fraction

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from posttrain.verifiers.drgrpo_grader import grade, r1_zero_reward_fn  # noqa: E402


def load(p):
    t = open(p, encoding="utf-8").read()
    return json.loads(t) if t.lstrip()[0] == "[" else [json.loads(x) for x in t.splitlines() if x.strip()]


def sid(r):
    return hashlib.sha256((r["problem"] + "\0" + r["reasoning_trace"]).encode()).hexdigest()[:16]


# ---------- independent canonicalizer: shares no code with drgrpo_grader ----------
UNITS = (r"(?:degrees?|inches|inch|feet|foot|cm|mm|km|m|meters?|miles?|mph|units?|square|cubic|sq|hours?|minutes?|seconds?|"
         r"days?|weeks?|years?|dollars?|cents?|gallons?|liters?|pounds?|ounces?|grams?|points?|people|students?|ways?|"
         r"centimeters?|millimeters?|kilometers?|per|old|of)")


def canon(s):
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = s.replace("\u2212", "-").replace("\u2013", "-")
    for a in ("\\left", "\\right", "\\displaystyle", "\\!", "\\,", "\\;", "\\:", "\\ ", "\\(", "\\)", "\\[", "\\]", "$", "~"):
        s = s.replace(a, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\cdot", "*")
    s = s.replace("\u03c0", "\\pi").replace("\u221e", "\\infty").replace("\u00b0", "").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\u221a\(?(\d+)\)?", r"\\sqrt{\1}", s)
    s = re.sub(r"\\(?:text|textbf|mathrm|mbox|textrm)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\sqrt\s*(\d)", r"\\sqrt{\1}", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac\s*(\d)\s*\{", r"\\frac{\1}{", s)
    s = re.sub(r"^\(?([A-Z])\)?$", r"\1", s.strip())
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = re.sub(r"^[A-Za-z]\s*=\s*", "", s.strip())
    s = re.sub(r"(?i)(?<=[\d}\s])\s*" + UNITS + r"\b(?:\s*" + UNITS + r"\b)*\.?$", "", s)
    s = s.replace("%", "").replace("\\", "\\").rstrip(".")
    return re.sub(r"\s+", "", s)


def number(s):
    """Exact rational value of a plain number / fraction / mixed number, else None."""
    c = canon(s)
    m = re.fullmatch(r"(-?)(\d+)\\frac\{(\d+)\}\{(\d+)\}", c)
    if m and int(m[4]):
        v = int(m[2]) + Fraction(int(m[3]), int(m[4])); return -v if m[1] else v
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", c)
    if m and int(m[3]):
        v = Fraction(int(m[2]), int(m[3])); return -v if m[1] else v
    m = re.fullmatch(r"(-?\d+)/(\d+)", c)
    if m and int(m[2]):
        return Fraction(int(m[1]), int(m[2]))
    if re.fullmatch(r"-?(?:\d+\.?\d*|\.\d+)", c):
        return Fraction(c)
    return None


def second_verifier(pred, gold):
    """Hugging Face math-verify called directly (different code path from the SymPy/mathd route)."""
    try:
        from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig
        cfg = [LatexExtractionConfig(), ExprExtractionConfig()]
        wrap = lambda x: x if "$" in x or "\\(" in x else "$" + x + "$"
        g, p = parse(wrap(str(gold)), extraction_config=cfg), parse(wrap(str(pred)), extraction_config=cfg)
        return bool(g and p and verify(g, p))
    except Exception:
        return False


def ok(pred, gold, fast=True):
    try:
        return bool(grade(pred, gold, fast))
    except Exception:
        return False


ADVERSARIAL = [  # (prediction, gold, should_accept, why)
    ("6 feet", "6 \\text{ inches}", False, "conflicting units"), ("12,34", "1234", False, "comma list vs integer"),
    ("1,2", "12", False, "pair vs integer"), ("(1,2)", "(2,1)", False, "ordered pair swapped"),
    ("[0,1)", "(0,1]", False, "interval brackets"), ("(0,1)", "[0,1]", False, "open vs closed"),
    ("50", "50\\%", True, "percent sign dropped (convention)"), ("0.5", "50\\%", False, "0.5 vs 50%"),
    ("2003", "2003_6", False, "missing base"), ("1011", "1011_2", False, "missing base"),
    ("x=3 or x=4", "3", False, "extra solution"), ("3 and 4", "3", False, "extra solution"),
    ("3", "3,4", False, "missing solution"), ("not 5", "5", False, "negated text"), ("-5", "5", False, "sign"),
    ("1/2", "2", False, "reciprocal"), ("3.14", "\\pi", False, "approximation"), ("0.333", "\\frac13", False, "approximation"),
    ("x+1", "1+x", True, "commutative"), ("\\frac{2}{4}", "\\frac{1}{2}", False, "unreduced (grader policy)"),
    ("10^2", "100", True, "evaluated power"), ("2\\sqrt{2}", "\\sqrt{8}", True, "equivalent radical"),
    ("\\text{A}", "\\text{B}", False, "different choice"), ("5 or 6", "5", False, "hedge"), ("", "5", False, "empty"),
    ("5, 5", "5", False, "duplicate list"), ("$5", "5", True, "currency"), ("5 dollars", "5", True, "unit"),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    raw = load(f"{a.src}/sft_gpt-oss-120b.jsonl"); fil = load(f"{a.src}/sft_gpt-oss-120b_filtered.jsonl")
    keep = {sid(r) for r in fil}; assert len(keep) == len(fil) and keep <= {sid(r) for r in raw}
    rem = [r for r in raw if sid(r) not in keep]
    rows, tiers = [], collections.Counter()
    for r in rem:
        fast = r1_zero_reward_fn(r["reasoning_trace"], r["expected_answer"], fast=True)["answer_reward"] == 1
        p, g = r["extracted_answer"], r["expected_answer"]
        np_, ng = number(p), number(g)
        c_eq = canon(p) == canon(g); n_eq = np_ is not None and np_ == ng
        tier = "not_recovered"
        if fast:
            tier = "A_canonical_string" if c_eq else "B_exact_number" if n_eq else \
                   "C_math_verify" if second_verifier(p, g) else "D_grader_only"
        conflict = fast and np_ is not None and ng is not None and np_ != ng
        rows.append(dict(id=sid(r), problem=r["problem"], extracted=p, expected=g, recovered=fast, tier=tier,
                         conflict=conflict, second=second_verifier(p, g), canon=c_eq, num=n_eq))
        tiers[tier] += 1
    rec = [x for x in rows if x["recovered"]]
    rep = {"raw": len(raw), "filtered": len(fil), "removed": len(rem), "recovered": len(rec), "tiers": dict(tiers),
           "numeric_conflicts": sum(x["conflict"] for x in rows),
           "rejected_by_fast_but_independently_equal": sum((not x["recovered"]) and (x["canon"] or x["num"]) for x in rows),
           "rejected_by_fast_but_math_verify_accepts": sum((not x["recovered"]) and x["second"] for x in rows),
           "recovered_id_sha256": hashlib.sha256("\n".join(sorted(x["id"] for x in rec)).encode()).hexdigest()}

    # ---- false-positive audit ----
    rng = random.Random(2026)
    mut = collections.defaultdict(lambda: [0, 0]); acc = []
    for r in raw:
        p, g = str(r["extracted_answer"]), str(r["expected_answer"])
        if not ok(p, g):
            continue
        nums = list(re.finditer(r"\d+", p))
        tests = {}
        if nums:
            m = rng.choice(nums); tests["increment"] = p[:m.start()] + str(int(m[0]) + 1) + p[m.end():]
            m = nums[0]; tests["append_zero"] = p[:m.end()] + "7" + p[m.end():]
        if re.fullmatch(r"-?\d+", p) and p.strip("-0"):
            tests["sign_flip"] = p[1:] if p[0] == "-" else "-" + p
        if len(nums) >= 2 and nums[0][0] != nums[1][0]:
            x, y = nums[0], nums[1]
            tests["swap_first_two_numbers"] = p[:x.start()] + y[0] + p[x.end():y.start()] + x[0] + p[y.end():]
        for k, q in tests.items():
            mut[k][0] += 1
            if ok(q, g):
                mut[k][1] += 1; acc.append((k, q, g))
    rep["mutation_false_accepts"] = {k: {"n": n, "accepted": c} for k, (n, c) in mut.items()}
    pair_n = pair_acc = 0; pairs = []
    for _ in range(20000):
        x, y = rng.choice(raw), rng.choice(raw)
        if x is y or canon(x["extracted_answer"]) == canon(y["expected_answer"]):
            continue
        nx, ny = number(x["extracted_answer"]), number(y["expected_answer"])
        if nx is not None and nx == ny:
            continue
        pair_n += 1
        if ok(x["extracted_answer"], y["expected_answer"]):
            pair_acc += 1; pairs.append((x["extracted_answer"], y["expected_answer"]))
    rep["random_pair_false_accepts"] = {"n": pair_n, "accepted": pair_acc}
    adv = [dict(pred=p, gold=g, expected=e, fast=ok(p, g), full=ok(p, g, False), why=w) for p, g, e, w in ADVERSARIAL]
    rep["adversarial"] = {"n": len(adv), "fast_false_accept": [x["why"] for x in adv if x["fast"] and not x["expected"]],
                          "fast_false_reject": [x["why"] for x in adv if not x["fast"] and x["expected"]],
                          "full_false_accept": [x["why"] for x in adv if x["full"] and not x["expected"]]}
    multi = ["</think> <answer>3</answer> </think> <answer>4</answer>", "</answer> 5 </think> <answer>",
             "</think> <answer>4"]
    rep["format_probe"] = {t: r1_zero_reward_fn(t, "4") for t in multi}
    json.dump(rep, open(f"{a.out}/verifier_audit.json", "w"), indent=1)
    with open(f"{a.out}/recovery_audit.jsonl", "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(json.dumps(rep, indent=1))
    print("## tier D (read every one)")
    for x in rows:
        if x["tier"] == "D_grader_only":
            print(repr(x["extracted"][:90]), "||", repr(x["expected"][:90]))
    print("## mutation accepts"); [print(k, repr(q[:60]), "||", repr(g[:60])) for k, q, g in acc[:60]]
    print("## random pair accepts"); [print(repr(p[:60]), "||", repr(g[:60])) for p, g in pairs[:25]]


if __name__ == "__main__":
    main()
