"""Verifier 1.x (strict grader) against 2.0.0 on every recorded output and every labelled SFT trace. Reads only; changes no label.

python scripts/verifier_crosscheck.py <new.json>
Every answer that 2.0.0 newly accepts is checked by methods that share no code with it: an exact rational canonicaliser
(stage0_verify_audit.number), SymPy on factorials, and math-verify. Also counts strict rejections that math-verify would accept.
"""
import collections, glob, hashlib, json, logging, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
logging.disable(logging.CRITICAL)
import sympy
import stage0_verify_audit as A
from posttrain.verifiers.drgrpo_grader import extract_answer, r1_zero_reward_fn
from posttrain.verify import FACT, LEAD, score, verifier_id

H = os.path.expanduser("~")


def independent(ans, gold):
    """(agrees, methods). Looks at the value after the lead-in, using neither the grader nor the new rules' arithmetic."""
    rest = re.sub(r"(?i)^.*?answer\s*(?:is)?\s*:?\s*", "", ans, count=1, flags=re.S).strip() if LEAD.match(ans) else ans.strip()
    f = re.fullmatch(r"[$\\()\s]*(\d+)\s*!\s*[$\\()\s]*\.?", rest)
    if f:
        g = A.number(gold)
        return g is not None and sympy.factorial(int(f[1])) == sympy.Rational(g.numerator, g.denominator), ["sympy factorial", "exact rational"]
    a, b = A.number(rest), A.number(gold)
    num = a is not None and a == b
    mv = A.second_verifier(rest, gold)
    return num or mv, [m for m, x in (("exact rational", num), ("math-verify", mv)) if x]


def strict(t, g, kind="r1_zero"):
    try:
        return r1_zero_reward_fn(t, g)["reward"] if kind == "r1_zero" else None
    except Exception:
        return 0.0


rep, flips = {}, []
files = sorted(glob.glob(H + "/a5_pilot/**/rollouts/*.jsonl", recursive=True) + glob.glob(H + "/a5_validation/*/rollouts/*.jsonl") + glob.glob(H + "/a5_pilot/**/eval/*.jsonl", recursive=True))
c = collections.Counter()
for f in files:
    if "question_only" in f:
        continue
    k = "eval " + re.sub(r"^.*a5_pilot/", "", f) if "/eval/" in f and "/rollouts/" not in f else "rollouts " + f.split("/rollouts/")[0].split("/")[-1]
    r = rep.setdefault(k, dict(n=0, formatted=0, strict=0, v2=0, rejected_but_math_verify_accepts=0))
    for x in map(json.loads, open(f, encoding="utf-8")):
        t, g = x["response"], str(x["answer"])
        s = score(t, g)
        a, b = strict(t, g), s["reward"]
        r["n"] += 1
        r["formatted"] += s["format_reward"] == 1
        r["strict"] += a
        r["v2"] += b
        c["checked"] += 1
        c["v2 rejects something strict accepts"] += a == 1 and b == 0
        ans = t.split("<answer>")[-1].replace("</answer>", "")
        if a == 0 and b == 1:
            ok, how = independent(ans, g)
            flips.append(dict(source=k, answer=ans.strip()[:120], gold=g, rule=s["rule"], independent_agrees=bool(ok), methods=how))
        if b == 0 and s["format_reward"] == 1 and A.second_verifier(ans.strip(), g):
            r["rejected_but_math_verify_accepts"] += 1
# labels: the SFT arms were built with the strict grader and stay as they are; how many labels would differ?
lab = collections.Counter()
for x in map(json.loads, open(os.path.join(ROOT, "data", "stage0", "sft_all_labeled_4836.jsonl"), encoding="utf-8")):
    a, b = strict(x["reasoning_trace"], x["expected_answer"]), score(x["reasoning_trace"], x["expected_answer"])["reward"]
    lab[("reverified" if x["reverified"] else "not reverified") + (" v2 accepts, strict rejects" if b > a else " same")] += 1
sheet = os.path.join(H, "a5_review", "result.json")
review = []
key = re.split(r"\n=+\n## Item (\d+)\n", open(H + "/a5_review/items.md", encoding="utf-8").read())
rv = json.load(open(sheet))
for i in rv["disagree"]:
    body = key[key.index(i) + 1]
    gold, resp = re.search(r"\*\*Reference answer:\*\* `(.*?)`", body)[1], re.search(r"~~~\n(.*?)\n~~~", body, re.S)[1]
    review.append(dict(item=int(i), reference=gold, strict=strict(resp, gold), v2=score(resp, gold)["reward"], rule=score(resp, gold)["rule"], answer=resp.split("<answer>")[-1].replace("</answer>", "")))
out = dict(verifier=verifier_id(), totals=dict(c), by_source=rep, newly_accepted=flips, newly_accepted_not_confirmed=[f for f in flips if not f["independent_agrees"]],
           sft_labels_if_rebuilt=dict(lab), note_labels="labels and subsets are NOT rebuilt", review_disagreements=review,
           review_record=dict(agree=rv["agree"], n=rv["n"], result_sha256=hashlib.sha256(open(sheet, "rb").read()).hexdigest()))
assert not os.path.exists(sys.argv[1]), "refusing to overwrite"
json.dump(out, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ("totals", "newly_accepted", "newly_accepted_not_confirmed", "sft_labels_if_rebuilt", "review_disagreements")}, indent=1, ensure_ascii=False))
for k, r in rep.items():
    print(k, {a: int(b) for a, b in r.items()})
sys.exit(1 if out["newly_accepted_not_confirmed"] or c["v2 rejects something strict accepts"] else 0)
