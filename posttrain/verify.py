"""Reward wrapper around the third-party strict grader: adds a timeout, error accounting and a stricter format diagnostic.

Version 2.0.0 (frozen, see verifier_lock.py) = the strict grader (1.x, unchanged) plus two additive rules for answers it rejects:
  lead-in   : a whole-string "(the) (final) answer is/:" prefix is removed, then the SAME strict grader must accept the remainder
  factorial : an answer that is exactly n! (n <= 20) is compared, as the integer, with an integer reference
Nothing the strict grader accepts is ever rejected. Other unevaluated arithmetic ("16-3-4") and prose around an answer stay rejected:
accepting them would pay a policy for not computing, or for hedging.

python -m posttrain.verify      check the frozen hashes and the golden cases
"""
import hashlib, json, math, os, re, signal, sys, threading, warnings
from .common import ROOT

warnings.filterwarnings("ignore", category=SyntaxWarning)
from .verifiers.drgrpo_grader import extract_answer, grade, question_only_reward_fn, r1_zero_reward_fn  # noqa: E402
from . import verifier_lock as lock  # noqa: E402

FNS = {"r1_zero": r1_zero_reward_fn, "question_only": question_only_reward_fn}
ZERO = {"format_reward": 0.0, "answer_reward": 0.0, "reward": 0.0}


LEAD = re.compile(r"(?i)\s*(?:(?:so|thus|therefore|hence),?\s+)?(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|:)\s*(?=\S)")
FACT = re.compile(r"(?:\$|\\\()?\s*(\d{1,2})\s*!\s*(?:\$|\\\))?\s*\.?")
INT = re.compile(r"\d+|\d{1,3}(?:(?:,|,\\!|\\,)\d{3})+")


def model_answer(text, kind):
    """The string the strict grader judges, extracted exactly as it does."""
    if kind == "r1_zero":
        a = text.split("<answer>")[-1].replace("</answer>", "")
        return extract_answer(a) if "\\boxed" in a else a
    return extract_answer(text)


def accept_v2(answer, gold):
    """Name of the additive rule that accepts `answer`, or None. Only called for answers the strict grader rejected."""
    if not isinstance(answer, str) or not isinstance(gold, str):
        return None
    m = LEAD.match(answer)
    rest = answer[m.end():].strip() if m else answer.strip()
    if m and grade(rest, gold, True):
        return "lead_in"
    f = FACT.fullmatch(rest)
    if f and int(f[1]) <= 20 and INT.fullmatch(gold.strip()) and int(re.sub(r"\D", "", gold)) == math.factorial(int(f[1])):
        return "factorial"
    return None


def verifier_id():
    """Version and content hashes of everything that decides a reward (line endings normalised)."""
    here = os.path.dirname(os.path.abspath(__file__))
    h = lambda p: hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    return dict(version=lock.VERSION, verify_py=h(os.path.join(here, "verify.py")), grader_py=h(os.path.join(here, "verifiers", "drgrpo_grader.py")))


def require_frozen():
    v = verifier_id()
    if v != dict(version=lock.VERSION, **lock.SHA256):
        raise SystemExit(f"the verifier differs from the frozen version {lock.VERSION}: {v}")
    return v


class _Timeout(BaseException):  # BaseException so the grader's own broad `except Exception` cannot swallow it
    pass


def score(text, gold, kind="r1_zero", seconds=5):
    """Never raises. Timeouts and grader errors score 0 and are flagged so they can be counted."""
    use = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()

    def handler(*_):
        raise _Timeout
    old = None
    try:
        if use:
            old = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
        r = dict(FNS[kind](text, gold, fast=True), error=None, rule=None)
        if r["format_reward"] == 1 and r["answer_reward"] == 0:
            rule = accept_v2(model_answer(text, kind), gold)
            if rule:
                r.update(answer_reward=1.0, reward=1.0, rule=rule)
        return r
    except _Timeout:
        return dict(ZERO, error="timeout", rule=None)
    except Exception as e:
        return dict(ZERO, error=type(e).__name__, rule=None)
    finally:
        if use:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)


def clean_format(text):
    """Stricter than the reward: exactly one </think>, one answer block, in order, nothing after it."""
    return (text.count("</think>") == 1 and text.count("<answer>") == 1 and text.count("</answer>") == 1
            and text.index("</think>") < text.index("<answer>") < text.index("</answer>")
            and text.rstrip().endswith("</answer>"))


if __name__ == "__main__":
    v = require_frozen()
    bad = [c for c in lock.CASES if score("r </think> <answer>" + c[0] + "</answer>", c[1])["reward"] != c[2]]
    print(json.dumps(v), f"{len(lock.CASES)} golden cases, {len(bad)} wrong", bad)
    sys.exit(1 if bad else 0)
