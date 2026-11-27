"""Split strict-verifier failures into format vs. reasoning failures (CPU only).
Relaxed extraction is pre-declared and applied in this order; first candidate found is graded with the
same equivalence grader: (1) text in <answer>...</answer> regardless of the '</think> <answer>' boundary,
(2) last \\boxed{}, (3) last number in the response. An upper-bound style diagnostic, not a benchmark score."""
import json, os, re, sys, random, collections, math
os.environ["CUDA_VISIBLE_DEVICES"] = ""
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
from posttrain.verifiers.drgrpo_grader import grade, extract_answer
OUT = sys.argv[1]


def safe(a, g):
    try:
        return bool(a) and bool(grade(a, g, True))
    except Exception:
        return False


def relaxed(t):
    m = re.findall(r"<answer>(.*?)</answer>", t, re.S)
    if m:
        a = m[-1]
        return "answer_tag", (extract_answer(a) if "\\boxed" in a else a.strip())
    if "\\boxed" in t:
        return "boxed", extract_answer(t)
    n = re.findall(r"-?\d[\d,]*\.?\d*", t)
    return ("last_number", n[-1].rstrip(".,")) if n else ("none", None)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / 2 / n) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / 4 / n / n) / d
    return round(100 * (c - h), 2), round(100 * (c + h), 2)


rep = {}; strict = {}
for name in ("r1_zero", "question_only"):
    p = f"{OUT}/base_{name}.jsonl"
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    c = collections.Counter(); ex = collections.defaultdict(list)
    for r in rows:
        ok = r["rewards"]["answer_reward"] == 1; fm = r["rewards"]["format_reward"] == 1
        how, a = relaxed(r["model_response"]); rel = safe(a, r["ground_truth"])
        c["strict_correct"] += ok; c["format_ok"] += fm
        c["relaxed_correct"] += (ok or rel)
        if not ok:
            key = ("format_ok_wrong" if not rel else "format_ok_relaxed_right") if fm else \
                  (f"format_fail_but_right:{how}" if rel else "format_fail_and_wrong")
            c[key] += 1; ex[key].append((r["ground_truth"], r["model_response"][-110:]))
        c["hit_1024"] += r["n_tokens"] >= 1024
    n = len(rows); strict[name] = [r["rewards"]["answer_reward"] == 1 for r in rows]
    rep[name] = {"n": n, **dict(c), "strict_acc_pct": round(100 * c["strict_correct"] / n, 2),
                 "strict_ci95": wilson(c["strict_correct"], n), "format_pct": round(100 * c["format_ok"] / n, 2),
                 "relaxed_acc_pct": round(100 * c["relaxed_correct"] / n, 2),
                 "mean_tokens": round(sum(r["n_tokens"] for r in rows) / n, 1)}
    random.seed(1)
    for k, v in ex.items():
        for g, t in random.sample(v, min(4, len(v))):
            print(name, k, "| gold", g, "|", repr(t))
if len(strict) == 2:
    a, b = strict["r1_zero"], strict["question_only"]
    both = sum(x and y for x, y in zip(a, b)); ao = sum(x and not y for x, y in zip(a, b)); bo = sum(y and not x for x, y in zip(a, b))
    rep["paired"] = dict(both=both, r1_only=ao, qonly_only=bo, neither=len(a) - both - ao - bo)
json.dump(rep, open(f"{OUT}/failure_decomposition.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
