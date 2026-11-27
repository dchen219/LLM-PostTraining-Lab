"""python -m posttrain.train_ei --config configs/ei/<name>.yaml

Expert Iteration: sample G traces per problem with the current policy, keep verified ones, fine-tune on them, repeat.
State is saved after every completed iteration, so --resume restarts at an iteration boundary.
"""
import collections, json, os, random, time
from .common import Run, jdump, jlines, jload, load_config, parser, seed_all, sha_file
from .diagnostics import gpu_mem, rollout_stats
from .engines import make_engine, verified_sync
from .evaluate import check_lock, run_eval_sets, template
from .model_io import load_policy, load_state, load_tokenizer, lr_scale, make_optimizer, resolve_parent, save_state
from .train_sft import finalize, guard_training_data, sft_step
from .verify import clean_format, score


def select(probs, groups, cfg, rng):
    """Keep verified, optionally clean-format, de-duplicated traces; cap per problem so easy problems cannot dominate."""
    keep, scores, seen = [], [], set()
    for r, cs in zip(probs, groups):
        ok = []
        for c in cs:
            s = score(c.text, r["answer"])
            scores.append(s)
            if s["reward"] == 1 and (clean_format(c.text) or not cfg["require_clean_format"]) and (r["id"], c.text) not in seen:
                seen.add((r["id"], c.text))
                ok.append(c)
        rng.shuffle(ok)
        keep += [(r, c) for c in ok[:cfg["max_per_problem"]]]
    return keep, scores


def main():
    a = parser(__doc__).parse_args()
    cfg = load_config(a.config, a.set)
    if a.print_config:
        print(json.dumps(cfg, indent=1))
        return
    check_lock(cfg["eval"]["sets"])
    guard_training_data(cfg["data"]["path"])
    run = Run(cfg, "ei", a.resume)
    seed_all(cfg["seed"])
    o, e = cfg["train"], cfg["ei"]
    path, parent = resolve_parent(cfg)
    tok = load_tokenizer(path)
    dev = cfg["device"]
    model = load_policy(path, dev, cfg.get("dtype", "fp32"), o.get("grad_ckpt", False))
    opt = make_optimizer(model, o)
    pool = jload(cfg["data"]["path"])
    tpl = template("r1_zero")
    jdump(f"{run.dir}/data.json", dict(n_pool=len(pool), file_sha256=sha_file(cfg["data"]["path"])))
    state, start, step = f"{run.dir}/state.pt", 0, 0
    if a.resume and os.path.exists(state):
        start, ex = load_state(state, model, opt)
        step = ex["step"]
    eng = make_engine(cfg, path, model, tok)
    ev = {}

    def do_eval(tag, it):
        verified_sync(eng, model)
        res = run_eval_sets(eng, cfg["eval"]["sets"], run.dir, tag, max_tokens=cfg["eval"]["max_tokens"], limit=cfg["eval"].get("limit"))
        run.log(dict(event="eval", iteration=it, step=step, **{f"{k}/{m}": v[m] for k, v in res.items() for m in ("accuracy", "format_rate", "mean_tokens", "n")}))
        return res

    if start == 0:
        ev = do_eval("iter0", 0)
    for it in range(start + 1, e["iterations"] + 1):
        rng = random.Random(f"{cfg['seed']}-ei-{it}")
        probs = rng.sample(pool, min(e["problems_per_iter"], len(pool)))
        verified_sync(eng, model)
        t = time.time()
        groups = eng.generate([tpl.format(question=r["question"].strip()) for r in probs], e["rollouts"], e["temperature"], e["max_tokens"],
                              ["</answer>"], cfg["seed"] * 1000 + it)
        gen_s = time.time() - t
        flat = [c for g in groups for c in g]
        keep, scores = select(probs, groups, e, rng)
        jlines(f"{run.dir}/accepted_iter{it}.jsonl", [dict(id=r["id"], question=r["question"], answer=r["answer"], response=c.text, n_tokens=len(c.token_ids))
                                                    for r, c in keep])
        solved = len({r["id"] for r, _ in keep})
        run.log(dict(event="rollout", iteration=it, n_rollouts=len(flat), n_verified=sum(s["reward"] == 1 for s in scores),
                     verified_rate=sum(s["reward"] == 1 for s in scores) / len(flat), accepted=len(keep), acceptance_rate=len(keep) / len(flat),
                     problems_solved=solved, problems=len(probs), gen_s=gen_s, gen_tok_per_s=sum(len(c.token_ids) for c in flat) / max(gen_s, 1e-9),
                     **rollout_stats(flat, scores), **gpu_mem()))
        if len(keep) < e.get("min_accepted", 1):
            raise SystemExit(f"iteration {it}: only {len(keep)} accepted traces; stopping instead of training on too little data")
        ex = [(tpl.format(question=r["question"].strip()), c.token_ids) for r, c in keep]
        agg = collections.defaultdict(float)
        for ep in range(e["epochs_per_iter"]):
            order = list(range(len(ex)))
            random.Random(f"{cfg['seed']}-{it}-{ep}").shuffle(order)
            for i in range(0, len(order), o["batch_size"]):
                for g in opt.param_groups:
                    g["lr"] = float(o["lr"]) * lr_scale(step, o.get("warmup", 0))
                t = time.time()
                m = sft_step(model, opt, tok, [ex[j] for j in order[i:i + o["batch_size"]]], cfg, dev)
                step += 1
                run.log(dict(event="train", iteration=it, epoch=ep, step=step, step_s=time.time() - t, **m, **gpu_mem()))
        ev = do_eval(f"iter{it}", it)
        save_state(state, model, opt, it, dict(step=step))
    lin = finalize(run, model, tok, cfg.get("stage", "ei"), parent, dict(iterations=e["iterations"], steps=step))
    if os.path.exists(state):
        os.remove(state)
    run.finish(steps=step, eval=ev, lineage=lin)


if __name__ == "__main__":
    main()
