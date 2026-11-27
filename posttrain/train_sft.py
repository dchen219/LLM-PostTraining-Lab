"""python -m posttrain.train_sft --config configs/sft/<name>.yaml [--set key=value ...] [--resume]"""
import json, math, os, random, time
import torch
from . import losses as L
from .batching import microbatches, seq_len
from .common import Run, jdump, jload, load_config, parser, seed_all, sha_file, sha_obj
from .diagnostics import ParamProbe, gpu_mem, reset_peaks
from .engines import make_engine, verified_sync
from .evaluate import check_lock, run_eval_sets, template
from .model_io import (load_policy, load_state, load_tokenizer, lr_scale, make_optimizer, read_lineage, resolve_parent, save_model,
                       save_state, weights_hash)


def autocast(device, on=True):
    kind = "cuda" if str(device).startswith("cuda") else "cpu"
    return torch.autocast(kind, torch.bfloat16, enabled=on and kind == "cuda")


def guard_training_data(path):
    """Evaluation isolation: an eval_* file can never be used as training data."""
    if os.path.basename(str(path)).startswith("eval_"):
        raise SystemExit(f"refusing to train on an evaluation file: {path}")


def finalize(run, model, tok, stage, parent, extra):
    """Save once. If a previous attempt already saved this run's checkpoint before crashing, reuse it."""
    path = f"{run.dir}/checkpoint"
    lin = read_lineage(path) if os.path.isdir(path) else None
    if lin and lin["config_sha256"] == run.meta["config_sha256"] and lin["weights_sha256"] == weights_hash(path) and lin["extra"] == extra:
        return lin
    return save_model(model, tok, path, stage, parent, run.meta, extra)


def sft_step(model, opt, tok, examples, cfg, device, probe=None):
    """One optimizer step over `examples` [(prompt, response)]; loss is the token mean over the WHOLE batch."""
    o = cfg["train"]
    mb = o["micro_batch"]
    ex = sorted(examples, key=lambda e: len(e[0]) + len(e[1]))
    eos, tb = o.get("append_eos", True), o.get("token_budget")
    total = [seq_len(tok, p, r, eos) for p, r in ex] if tb else None
    parts = [L.encode(tok, [ex[i][0] for i in ix], [ex[i][1] for i in ix], device, eos, o.get("max_len"))
             for ix in microbatches(list(range(len(ex))), total, mb, tb)]
    total = sum(b["response_mask"].sum() for b in parts)
    loss_sum = ent_sum = 0.0
    model.train()
    for b in parts:
        with autocast(device):
            lp, ent = L.token_stats(model, b, entropy=o.get("log_entropy", True))
        s, _ = L.sft_loss_sum(lp, b["response_mask"])
        if not torch.isfinite(s):
            raise RuntimeError("non-finite SFT loss")
        (s / total).backward()
        loss_sum += s.item()
        if ent is not None:
            ent_sum += (ent * b["response_mask"]).sum().item()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), o["max_grad_norm"])
    if not torch.isfinite(gn):
        raise RuntimeError("non-finite gradient norm; stopping rather than corrupting the checkpoint")
    if probe:
        probe.before()
    opt.step()
    moved = probe.after() if probe else {}
    opt.zero_grad(set_to_none=True)
    return dict(**moved, loss=loss_sum / total.item(), entropy=ent_sum / total.item(), grad_norm=gn.item(), tokens=int(total.item()))


def load_examples(d):
    rows = {r["id"]: r for r in jload(d["path"])}
    if d.get("subset"):
        ids = json.load(open(d["subset"], encoding="utf-8"))["ids"]
    else:
        ids = list(rows)
    missing = [i for i in ids if i not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} subset ids are missing from {d['path']}")
    return [rows[i] for i in ids]


def main():
    a = parser(__doc__).parse_args()
    cfg = load_config(a.config, a.set)
    if a.print_config:
        print(json.dumps(cfg, indent=1))
        return
    check_lock((cfg.get("eval") or {}).get("sets") or {})
    guard_training_data(cfg["data"]["path"])
    run = Run(cfg, "sft", a.resume)
    seed_all(cfg["seed"])
    o = cfg["train"]
    path, parent = resolve_parent(cfg)
    tok = load_tokenizer(path)
    dev = cfg["device"]
    model = load_policy(path, dev, cfg.get("dtype", "fp32"), o.get("grad_ckpt", False))
    opt = make_optimizer(model, o)
    rows = load_examples(cfg["data"])
    tpl = template(cfg["data"].get("prompt", "r1_zero"))
    ex = [(tpl.format(question=r["problem"].strip()), r["reasoning_trace"]) for r in rows]
    per_epoch = math.ceil(len(ex) / o["batch_size"])
    total = o.get("max_steps") or per_epoch * o["epochs"]
    jdump(f"{run.dir}/data.json", dict(n=len(ex), file_sha256=sha_file(cfg["data"]["path"]), ids_sha256=sha_obj([r["id"] for r in rows]),
                                     ids=[r["id"] for r in rows], steps_total=total))
    state, step = f"{run.dir}/state.pt", 0
    if a.resume and os.path.exists(state):
        step, _ = load_state(state, model, opt)
    eng = make_engine(cfg, path, model, tok) if cfg.get("eval") else None
    probe = ParamProbe(model)

    def do_eval(tag):
        verified_sync(eng, model)
        res = run_eval_sets(eng, cfg["eval"]["sets"], run.dir, tag, max_tokens=cfg["eval"]["max_tokens"], limit=cfg["eval"].get("limit"))
        run.log(dict(event="eval", step=step, **{f"{k}/{m}": v[m] for k, v in res.items() for m in ("accuracy", "format_rate", "mean_tokens", "n")}))
        return res

    if eng and step == 0 and cfg["eval"].get("at_start", True):
        do_eval("step0")
    every = (cfg.get("eval") or {}).get("every")
    while step < total:
        epoch, k = divmod(step, per_epoch)
        order = list(range(len(ex)))
        random.Random(f"{cfg['seed']}-{epoch}").shuffle(order)
        batch = [ex[i] for i in order[k * o["batch_size"]:(k + 1) * o["batch_size"]]]
        for g in opt.param_groups:
            g["lr"] = float(o["lr"]) * lr_scale(step, o.get("warmup", 0))
        t = time.time()
        reset_peaks()
        m = sft_step(model, opt, tok, batch, cfg, dev, probe)
        step += 1
        dt = time.time() - t
        run.log(dict(event="train", step=step, epoch=epoch, lr=opt.param_groups[0]["lr"], step_s=dt, tok_per_s=m["tokens"] / dt, **m, **gpu_mem()))
        if o.get("save_every") and step % o["save_every"] == 0 and step < total:
            save_state(state, model, opt, step, {})
        if eng and every and step % every == 0 and step < total:
            do_eval(f"step{step}")
    final = do_eval("final_step") if eng else {}
    lin = finalize(run, model, tok, cfg.get("stage", "sft"), parent, dict(steps=step, n_train=len(ex)))
    if os.path.exists(state):
        os.remove(state)
    run.finish(steps=step, eval=final, lineage=lin)


if __name__ == "__main__":
    main()
