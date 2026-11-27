"""python -m posttrain.evaluate --config configs/eval/dev.yaml [--set init.path=... run_dir=...]"""
import os, re, time
from .common import ROOT, Run, jdump, jlines, jload, load_config, parser, seed_all, sha_file
from .stats import wilson
from .verify import clean_format, score
from .prompts import load as load_prompt


def template(name):
    return load_prompt(name)


def category(text, s, gold, finish):
    """Failure taxonomy. Relaxed extraction, in order: <answer> tag, last \\boxed, last number."""
    if s["answer_reward"] == 1:
        return "correct"
    if s["format_reward"] == 1:
        return "format_ok_answer_wrong"
    from .verifiers.drgrpo_grader import extract_answer, grade
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    a = m[-1].strip() if m else (extract_answer(text) if "\\boxed" in text else None)
    if a and "\\boxed" in a:
        a = extract_answer(a)
    if a is None:
        n = re.findall(r"-?\d[\d,]*\.?\d*", text)
        a = n[-1].rstrip(".,") if n else None
    try:
        right = bool(a) and bool(grade(a, gold, True))
    except Exception:
        right = False
    if right:
        return "format_fail_answer_right"
    return "format_fail_truncated" if finish == "length" else "format_fail_answer_wrong"


def evaluate(engine, rows, prompt="r1_zero", max_tokens=1024, temperature=0.0, seed=0):
    stop = ["</answer>"] if prompt == "r1_zero" else None
    t = time.time()
    outs = engine.generate([template(prompt).format(question=r["question"].strip()) for r in rows], 1, temperature, max_tokens, stop, seed)
    el, res, cats = time.time() - t, [], {}
    for r, (c,) in zip(rows, outs):
        s = score(c.text, r["answer"], prompt)
        k = category(c.text, s, r["answer"], c.finish)
        cats[k] = cats.get(k, 0) + 1
        res.append(dict(id=r["id"], question=r["question"], answer=r["answer"], response=c.text, n_tokens=len(c.token_ids), finish=c.finish,
                        correct=s["answer_reward"] == 1, format=s["format_reward"] == 1, clean_format=clean_format(c.text),
                        error=s["error"], rule=s["rule"], category=k))
    n = len(res)
    k = sum(x["correct"] for x in res)
    return dict(n=n, correct=k, accuracy=k / n, ci95=wilson(k, n), format_rate=sum(x["format"] for x in res) / n,
                clean_format_rate=sum(x["clean_format"] for x in res) / n, mean_tokens=sum(x["n_tokens"] for x in res) / n,
                truncated=sum(x["finish"] == "length" for x in res), verifier_errors=sum(bool(x["error"]) for x in res), additive_rule=sum(bool(x["rule"]) for x in res),
                categories=cats, gen_s=el), res


def run_eval_sets(engine, sets, run_dir, tag, prompt="r1_zero", max_tokens=1024, limit=None):
    out = {}
    os.makedirs(f"{run_dir}/eval", exist_ok=True)
    for name, path in sets.items():
        rows = jload(path)
        m, res = evaluate(engine, rows[:limit] if limit else rows, prompt, max_tokens)
        jlines(f"{run_dir}/eval/{tag}_{name}.jsonl", res)
        out[name] = dict(m, data_sha256=sha_file(path), file=f"eval/{tag}_{name}.jsonl")
    return out


def check_lock(sets):
    if any("final" in os.path.basename(str(p)) for p in sets.values()) and os.environ.get("ALLOW_FINAL_EVAL") != "1":
        raise SystemExit("final test sets are locked; set ALLOW_FINAL_EVAL=1 only for a pre-registered final run")


def main():
    a = parser(__doc__).parse_args()
    cfg = load_config(a.config, a.set)
    if a.print_config:
        print(cfg)
        return
    check_lock(cfg["sets"])
    from .engines import make_engine
    from .model_io import load_policy, load_tokenizer, resolve_parent
    run = Run(cfg, "eval", a.resume)
    seed_all(cfg["seed"])
    path, parent = resolve_parent(cfg)
    tok = load_tokenizer(path)
    model = load_policy(path, cfg["engine"].get("device", "cpu"), cfg.get("dtype", "bf16")) if cfg["engine"]["name"] == "hf" else None
    eng = make_engine(cfg, path, model, tok)
    out = {p: run_eval_sets(eng, cfg["sets"], run.dir, p, p, cfg["max_tokens"], cfg.get("limit")) for p in cfg["prompts"]}
    jdump(f"{run.dir}/summary.json", dict(results=out, parent=parent))
    run.finish(results={p: {k: v["accuracy"] for k, v in r.items()} for p, r in out.items()})


if __name__ == "__main__":
    main()
