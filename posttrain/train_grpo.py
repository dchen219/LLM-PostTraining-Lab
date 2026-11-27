"""python -m posttrain.train_grpo --config configs/grpo/<name>.yaml [--resume]

Pathways A (base -> GRPO) and B (base -> SFT -> GRPO) differ only in init.path and init.expect_stage.
"""
import json, os, random, time
import torch
from . import losses as L
from .batching import microbatches, seq_len
from .common import Run, jdump, jlines, jload, load_config, parser, seed_all, sha_file
from .common import TERMINAL
from .diagnostics import DifficultySampler, ParamProbe, adam_update_stats, gpu_mem, group_stats, reset_peaks, rollout_stats
from .engines import make_engine, verified_sync
from .evaluate import check_lock, run_eval_sets, template
from .model_io import load_policy, load_state, load_tokenizer, lr_scale, make_optimizer, resolve_parent, save_state
from .train_sft import autocast, finalize, guard_training_data
from .verify import score


def chunks(idx, n):
    return [idx[i:i + n] for i in range(0, len(idx), n)]


def logprobs_nograd(model, tok, prompts, comps, dev, mb, max_len, token_budget=None):
    """Old-policy log-probs, one 1-D tensor per rollout over response tokens (no padding assumptions downstream)."""
    out = [None] * len(prompts)
    was = model.training
    model.eval()
    order = sorted(range(len(prompts)), key=lambda i: len(comps[i].token_ids))
    total = [seq_len(tok, prompts[i], comps[i].token_ids) for i in range(len(prompts))] if token_budget else None
    with torch.no_grad(), autocast(dev):
        for ix in microbatches(order, total, mb, token_budget):
            b = L.encode(tok, [prompts[i] for i in ix], [comps[i].token_ids for i in ix], dev, max_len=max_len)
            lp, _ = L.token_stats(model, b)
            for j, i in enumerate(ix):
                out[i] = lp[j][b["response_mask"][j].bool()].cpu()
    if was:
        model.train()
    return out


def place(old, mask):
    """Scatter per-rollout old log-probs back onto the (B, T) response-mask layout."""
    z = torch.zeros_like(mask)
    for j, o in enumerate(old):
        z[j][mask[j].bool()] = o.to(mask.device, mask.dtype)
    return z


def sampling_mismatch(comps, old):
    """vLLM sampling-time log-probs vs the PyTorch old log-probs used in the ratio, on the actual rollouts."""
    for c, o in zip(comps, old):
        if c.logprobs and len(c.logprobs) != len(o):  # zip() alone would hide a token misalignment
            raise RuntimeError(f"sampling log-probs ({len(c.logprobs)}) and policy log-probs ({len(o)}) are not aligned")
    d, worst = [], None
    for i, (c, o) in enumerate(zip(comps, old)):
        if not c.logprobs:
            continue
        for j, (a, b) in enumerate(zip(c.logprobs, o.tolist())):
            x = abs(a - b)
            d.append(x)
            if worst is None or not x <= worst[0]:  # NaN counts as worst
                worst = (x, i, j, c.token_ids[j], a, b)
    if not d:
        return {}
    d.sort()
    return dict(rollout_logprob_mad=sum(d) / len(d), rollout_logprob_p99=d[int(0.99 * (len(d) - 1))], rollout_logprob_max=d[-1], rollout_logprob_tokens=len(d),
                rollout_logprob_over1=sum(x > 1 for x in d), rollout_logprob_over5=sum(x > 5 for x in d), rollout_worst_rollout=worst[1], rollout_worst_pos=worst[2],
                rollout_worst_token=worst[3], rollout_worst_sampling_lp=worst[4], rollout_worst_policy_lp=worst[5])


def check_rollouts(comps, old, tol, shares_policy=False):
    """Correctness gate on the ACTUAL rollouts, before any update: the engine's sampling log-probs must agree with the policy that will be trained.

    Separate from the weight check: it also covers token alignment, stale caches, prompts and kernels.
    """
    if shares_policy:
        return {}
    d = sampling_mismatch(comps, old)
    if not d or d["rollout_logprob_tokens"] < sum(len(c.token_ids) for c in comps):
        raise SystemExit("rollout log-prob check failed: the engine did not return a sampling log-prob for every rollout token")
    if not d["rollout_logprob_mad"] <= tol:  # also catches NaN
        raise SystemExit(f"rollout log-prob check failed: mean |sampling - policy| = {d['rollout_logprob_mad']:.4f} exceeds {tol}")
    return d


def grpo_update(model, opt, tok, prompts, comps, adv, rewards, old, cfg, dev, seed, probe=None):
    """Policy update(s) on one rollout batch. Returns diagnostics.

    group_filter: none | compute_only (skip zero-advantage rollouts, same denominator => identical gradient, less compute)
                  | renormalize (also divide by the number of informative rollouts => larger effective step; an intervention)
    """
    g, o = cfg["grpo"], cfg["train"]
    N, kind, filt = len(prompts), g["loss"], g.get("group_filter", "none")
    if filt != "none" and kind == "no_baseline":
        raise ValueError("group filtering is only valid for advantage-based losses")
    use = [i for i in range(N) if filt == "none" or adv[i] != 0]
    tb = o.get("token_budget")
    total = [seq_len(tok, prompts[i], comps[i].token_ids) for i in range(N)] if tb else None
    d = dict(updates=0, trained_rollouts=0, trained_tokens=0, loss=0.0, entropy_sum=0.0, tok_sum=0.0, clip_outside_tokens=0.0, clip_active_tokens=0.0,
             ratio_sum=0.0, kl_sum=0.0, ratio_min=float("inf"), ratio_max=0.0, grad_norms=[], row_entropy={})
    if not use:
        return d
    model.train()
    ub = g["update_batch"]
    for ep in range(g["epochs"]):
        order = list(range(N))
        random.Random(f"{seed}-{ep}").shuffle(order)
        keep = set(use)
        for ix_all in chunks(order, ub):
            ix = [i for i in ix_all if i in keep]
            if not ix:
                continue
            denom = len(ix) if filt == "renormalize" else len(ix_all)
            ix.sort(key=lambda i: len(comps[i].token_ids))
            for m_ix in microbatches(ix, total, o["micro_batch"], tb):
                b = L.encode(tok, [prompts[i] for i in m_ix], [comps[i].token_ids for i in m_ix], dev, max_len=o.get("max_len"))
                with autocast(dev):
                    lp, ent = L.token_stats(model, b, entropy=True)
                mk = b["response_mask"]
                seq, m = L.policy_loss(kind, lp, mk, adv[m_ix].to(dev), rewards[m_ix].to(dev),
                                       place([old[i] for i in m_ix], mk) if old else None, g["clip"], g["length_norm"], g["norm_const"])
                loss = seq.sum() / denom
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite GRPO loss")
                loss.backward()
                d["loss"] += loss.item()
                d["entropy_sum"] += (ent * mk).sum().item()
                for j, i in enumerate(m_ix):  # per-rollout mean entropy: diagnostic, computed without gradient
                    d["row_entropy"][i] = ((ent[j] * mk[j]).sum() / mk[j].sum().clamp(min=1)).item()
                d["tok_sum"] += mk.sum().item()
                for k in ("clip_outside_tokens", "clip_active_tokens", "ratio_sum", "kl_sum"):
                    d[k] += m.get(k, 0.0)
                d["ratio_min"] = min(d["ratio_min"], m.get("ratio_min", 1.0))
                d["ratio_max"] = max(d["ratio_max"], m.get("ratio_max", 1.0))
                d["trained_rollouts"] += len(m_ix)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), o["max_grad_norm"])
            if not torch.isfinite(gn):
                raise RuntimeError("non-finite gradient norm; stopping")
            if probe:
                probe.before()
            opt.step()
            if probe:
                d["moved"] = probe.after()
            opt.zero_grad(set_to_none=True)
            d["upd"] = adam_update_stats(opt)  # after the gradients are freed: the temporaries are chunked and small
            d["grad_norms"].append(gn.item())
            d["updates"] += 1
    return d


def summarize(d):
    t = max(d["tok_sum"], 1)
    gn = d["grad_norms"]
    out = dict(updates=d["updates"], trained_rollouts=d["trained_rollouts"], trained_tokens=int(d["tok_sum"]),
               loss=d["loss"] / max(d["updates"], 1), entropy=d["entropy_sum"] / t, grad_norm=sum(gn) / len(gn) if gn else None,
               grad_norm_max=max(gn) if gn else None)
    out.update(d.get("moved", {}))
    out.update(d.get("upd", {}))
    if d["ratio_sum"]:
        out.update(ratio_mean=d["ratio_sum"] / t, ratio_min=d["ratio_min"], ratio_max=d["ratio_max"], approx_kl=d["kl_sum"] / t,
                   clip_frac=d["clip_outside_tokens"] / t, clip_frac_active=d["clip_active_tokens"] / t)
    return out


def rollout_records(step, ids, rows, comps, rw, adv, sc, pol, G, entropy=None, worst=0, tok=None):
    """One record per rollout: prompt, response, token IDs, reward, format, truncation, entropy and the per-token sampling/policy log-probs.

    Written from values the step already computed; it draws no random numbers and touches neither the model nor the optimizer.
    """
    def piece(t):
        try:
            return tok.decode([t])
        except Exception:
            return None
    recs, gaps = [], []
    for i, c in enumerate(comps):
        p = pol[i].tolist() if pol and i < len(pol) and pol[i] is not None else []
        s = list(c.logprobs)
        ok = len(p) == len(s) == len(c.token_ids) and len(s) > 0
        g = [abs(a - b) for a, b in zip(s, p)] if ok else []
        gaps += g
        r = dict(step=step, i=i, group=i // G, id=ids[i // G], prompt_id=ids[i // G], answer=rows[i // G]["answer"], response=c.text,
                 token_ids=list(c.token_ids), n_tokens=len(c.token_ids), finish=c.finish, truncated=c.finish == "length", reward=rw[i],
                 format_reward=sc[i].get("format_reward"), answer_reward=sc[i].get("answer_reward"), adv=float(adv[i]), rule=sc[i].get("rule"),
                 verifier_error=sc[i].get("error"), entropy=(entropy or {}).get(i), aligned=ok, gap_mean=sum(g) / len(g) if g else None,
                 gap_max=max(g) if g else None, sampling_logprobs=s, policy_logprobs=p)
        if worst:
            w = sorted(range(len(g)), key=lambda j: -g[j] if g[j] == g[j] else float("-inf"))[:worst]
            r["worst"] = [dict(pos=j, token_id=c.token_ids[j], piece=piece(c.token_ids[j]), sampling_lp=s[j], policy_lp=p[j]) for j in w]
        recs.append(r)
    return recs, gaps


def dump_failed_batch(run_dir, meta, ids, rows, comps, rw, adv, sc, pol, G, tok):
    """Everything about a batch a correctness gate rejected, written BEFORE exiting. No optimizer update was made on it."""
    out = os.path.join(run_dir, f"failed_batch_step{meta['attempted_step']}")
    k = 1
    while os.path.exists(out + (f"_{k}" if k > 1 else "")):
        k += 1
    out += f"_{k}" if k > 1 else ""
    os.makedirs(out)
    recs, gaps = rollout_records(meta["attempted_step"], ids, rows, comps, rw, adv, sc, pol, G, worst=5, tok=tok)
    jlines(os.path.join(out, "rollouts.jsonl"), recs)
    gs = sorted(x for x in gaps if x == x)
    st = dict(tokens=len(gaps), nan=len(gaps) - len(gs), mean=sum(gaps) / len(gaps) if gaps else None, p99=gs[int(0.99 * (len(gs) - 1))] if gs else None,
              max=gs[-1] if gs else None, over1=sum(x > 1 for x in gs), over5=sum(x > 5 for x in gs), rollouts_misaligned=sum(not r["aligned"] for r in recs))
    jdump(os.path.join(out, "meta.json"), dict(meta, optimizer_updated=False, mismatch=st, prompt_ids=ids, **group_stats(rw, G), **rollout_stats(comps, sc)))
    return out


def main():
    a = parser(__doc__).parse_args()
    cfg = load_config(a.config, a.set)
    if a.print_config:
        print(json.dumps(cfg, indent=1))
        return
    if "fork" in cfg:
        raise SystemExit("a fork config is started with python -m posttrain.fork_grpo")
    train(cfg, a.resume)


def train(cfg, resume=False, fork=None):
    """fork: a posttrain.fork_grpo.Fork. A new branch starts from its verified source checkpoint; an interrupted branch resumes from its own state."""
    check_lock(cfg["eval"]["sets"])
    guard_training_data(cfg["data"]["path"])
    g, o = cfg["grpo"], cfg["train"]
    if g["epochs"] > 1 or g["update_batch"] < g["prompts_per_step"] * g["group_size"]:
        assert g["loss"].startswith("grpo"), "multiple updates per rollout batch need an importance-ratio loss"
    run = Run(cfg, "grpo", resume)
    seed_all(cfg["seed"])
    path, parent = resolve_parent(cfg)
    tok = load_tokenizer(path)
    dev = cfg["device"]
    model = load_policy(path, dev, cfg.get("dtype", "fp32"), o.get("grad_ckpt", False))
    opt = make_optimizer(model, o)
    pool = {r["id"]: r for r in jload(cfg["data"]["path"])}
    jdump(f"{run.dir}/data.json", dict(n_pool=len(pool), file_sha256=sha_file(cfg["data"]["path"]), parent=parent, fork=cfg.get("fork")))
    sampler = DifficultySampler(pool, random.Random(f"{cfg['seed']}-prompts"), g.get("difficulty_sampling", False))
    tpl = template("r1_zero")
    state, step, ex = f"{run.dir}/state.pt", 0, None
    if resume and os.path.exists(state):
        step, ex = load_state(state, model, opt)
    elif fork is not None:  # first start, or a branch interrupted before its first save: (re)start from the verified source
        step, ex, proof = fork.start(model, opt)
        run.log(dict(event="fork_start", step=step, **proof))
    if ex is not None:
        sampler.rng.setstate(ex["sampler_rng"])
        sampler.p = ex["sampler_p"]
        moved = run.rewind(step)  # nothing after the restored step stays in the active log
        if moved["rows"] or moved["files"]:
            print(f"resumed at step {step}: moved {moved['rows']} later log rows and {moved['files']} files to superseded/", flush=True)
    eng = make_engine(cfg, path, model, tok)
    pprobe = ParamProbe(model)
    probe = [tpl.format(question=r["question"].strip()) for r in list(pool.values())[:4]]

    if not eng.shares_policy and not g.get("check_rollout_logprobs", True):
        raise SystemExit("check_rollout_logprobs=false is not allowed with a separate rollout engine: it is a correctness gate")

    def do_eval(tag):
        verified_sync(eng, model)
        res = run_eval_sets(eng, cfg["eval"]["sets"], run.dir, tag, max_tokens=cfg["eval"]["max_tokens"], limit=cfg["eval"].get("limit"))
        run.log(dict(event="eval", step=step, **{f"{k}/{m}": v[m] for k, v in res.items() for m in ("accuracy", "format_rate", "mean_tokens", "n")}))
        return res

    ev = {}
    if step == 0 and cfg["eval"].get("at_start", True):
        ev = do_eval("step0")
    if fork is not None and step == fork.step and not any(r.get("event") == "eval" and r.get("step") == step for r in jload(f"{run.dir}/metrics.jsonl")):
        do_eval(f"fork_step{step}")  # the verified source checkpoint, before this branch changes it
    try:
        G = g["group_size"]
        while step < g["steps"]:
            reset_peaks()
            T = {}
            t = time.time()
            sync = verified_sync(eng, model)  # primary gate, every step: exact equality with the dtype-converted policy weights
            if g.get("sync_check_every") and step % g["sync_check_every"] == 0:
                sync.update(eng.check_sync(model, probe))  # diagnostic only: mixes bf16 and kernel noise with any weight error, and misses small errors
                sync["sync_probe_over_tol"] = sync["sync_mean_abs_logprob_diff"] > g["sync_tol"]
            T["sync_s"] = time.time() - t
            ids = sampler.sample(g["prompts_per_step"])
            rows = [pool[i] for i in ids]
            base = [tpl.format(question=r["question"].strip()) for r in rows]
            t = time.time()
            groups = eng.generate(base, G, g["temperature"], g["max_tokens"], ["</answer>"], cfg["seed"] * 100003 + step,
                                  want_logprobs=not eng.shares_policy)
            T["rollout_s"] = time.time() - t
            comps = [c for x in groups for c in x]
            prompts = [p for p in base for _ in range(G)]
            t = time.time()
            sc = [score(c.text, r["answer"]) for r, x in zip(rows, groups) for c in x]
            T["reward_s"] = time.time() - t
            rw = [s["reward"] for s in sc]
            rules = sum(bool(s["rule"]) for s in sc)
            adv = L.group_advantages(rw, G, g["std_norm"])
            sampler.update(ids, [sum(rw[i * G:(i + 1) * G]) / G for i in range(len(ids))])
            t = time.time()
            ratio_loss = g["loss"].startswith("grpo")
            now = logprobs_nograd(model, tok, prompts, comps, dev, o["micro_batch"], o.get("max_len"), o.get("token_budget")) if ratio_loss or not eng.shares_policy else None
            T["old_lp_s"] = time.time() - t
            try:
                mism = check_rollouts(comps, now, g.get("rollout_tol", 0.05), eng.shares_policy)
            except (SystemExit, RuntimeError) as e:  # rejected BEFORE grpo_update(): keep the evidence, then stop
                nxt = float(o["lr"]) * lr_scale(step, o.get("warmup", 0))
                meta = dict(attempted_step=step + 1, completed_updates=step, error=str(e), tol=g.get("rollout_tol", 0.05), lr_for_attempt=nxt,
                            generation_seed=cfg["seed"] * 100003 + step, config_sha256=run.meta["config_sha256"], time=time.time(),
                            sync={k: v for k, v in sync.items() if k != "mismatch_names"}, timings=T, **gpu_mem())
                fb = dump_failed_batch(run.dir, meta, ids, rows, comps, rw, adv, sc, now, G, tok)
                run.log(dict(event="gate_failed", step=step, attempted_step=step + 1, error=str(e), failed_batch=os.path.relpath(fb, run.dir)))
                raise
            old = now if ratio_loss else None
            for pg in opt.param_groups:
                pg["lr"] = float(o["lr"]) * lr_scale(step, o.get("warmup", 0))
            t = time.time()
            try:
                d = grpo_update(model, opt, tok, prompts, comps, adv, torch.tensor(rw), old, cfg, dev, cfg["seed"] * 7919 + step, pprobe)
            except (SystemExit, RuntimeError) as e:  # non-finite loss or gradient: raised before opt.step(), so this batch changed nothing either
                if any(x in str(e) for x in TERMINAL):
                    meta = dict(attempted_step=step + 1, completed_updates=step, error=str(e), stage="update", lr_for_attempt=opt.param_groups[0]["lr"],
                                generation_seed=cfg["seed"] * 100003 + step, config_sha256=run.meta["config_sha256"], time=time.time(), rollout_check=mism, **gpu_mem())
                    fb = dump_failed_batch(run.dir, meta, ids, rows, comps, rw, adv, sc, now, G, tok)
                    run.log(dict(event="update_failed", step=step, attempted_step=step + 1, error=str(e), failed_batch=os.path.relpath(fb, run.dir)))
                raise
            T["update_s"] = time.time() - t
            step += 1
            ntok = sum(len(c.token_ids) for c in comps)
            run.log(dict(event="train", step=step, rollouts=len(comps), lr=opt.param_groups[0]["lr"], rollout_tok_per_s=ntok / max(T["rollout_s"], 1e-9),
                         rollouts_per_s=len(comps) / max(T["rollout_s"], 1e-9), step_s=sum(T.values()), **T, **group_stats(rw, G), **rollout_stats(comps, sc),
                         **summarize(d), **mism, **{k: v for k, v in sync.items() if k != "sync_s"}, verifier_additive_rule=rules, **gpu_mem()))
            if g.get("save_rollouts_every") and step % g["save_rollouts_every"] == 0:
                os.makedirs(f"{run.dir}/rollouts", exist_ok=True)
                jlines(f"{run.dir}/rollouts/step{step}.jsonl", rollout_records(step, ids, rows, comps, rw, adv, sc, now, G, d.get("row_entropy"))[0])
            if o.get("save_every") and step % o["save_every"] == 0 and step < g["steps"]:
                save_state(state, model, opt, step, dict(sampler_rng=sampler.rng.getstate(), sampler_p=sampler.p))
            if cfg["eval"].get("every") and step % cfg["eval"]["every"] == 0 and step < g["steps"]:
                do_eval(f"step{step}")
        ev = do_eval("final_step")
        extra = dict(steps=step, **({"fork": cfg["fork"]} if fork is not None else {}))
        lin = finalize(run, model, tok, cfg.get("stage", "grpo"), parent, extra)
        if os.path.exists(state):
            os.remove(state)
        run.finish(steps=step, eval=ev, lineage=lin, **({"source_check": fork.check_source()} if fork is not None else {}))
    except (SystemExit, RuntimeError) as e:
        if any(t in str(e) for t in TERMINAL):  # a correctness gate: a finished outcome, never retried
            run.fail(e, step, **({"source_check": fork.check_source()} if fork is not None else {}))
        raise


if __name__ == "__main__":
    main()
