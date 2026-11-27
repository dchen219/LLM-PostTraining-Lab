"""Model loading, checkpointing, resume and lineage tracking."""
import inspect, json, os, random, shutil, time
import torch
from .common import jdump, sha_file


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_policy(path, device, dtype="fp32", grad_ckpt=False):
    """fp32 master weights by default: in pure bf16 most lr=1e-5 updates round to zero (results/stage0/bf16_update_check.json)."""
    import transformers
    from transformers import AutoModelForCausalLM
    t = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    m = AutoModelForCausalLM.from_pretrained(path, attn_implementation="sdpa", **{key: t}).to(device)
    if grad_ckpt:
        m.gradient_checkpointing_enable()
    return m


def make_optimizer(model, o):
    kw = dict(lr=float(o["lr"]), betas=tuple(o.get("betas", (0.9, 0.95))), weight_decay=float(o.get("weight_decay", 0.0)))
    if "fused" in inspect.signature(torch.optim.AdamW).parameters and next(model.parameters()).is_cuda:
        kw["fused"] = True
    return torch.optim.AdamW(model.parameters(), **kw)


def lr_scale(step, warmup):
    return min(1.0, (step + 1) / warmup) if warmup else 1.0


def read_lineage(path):
    f = os.path.join(path, "lineage.json")
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


def weights_hash(path):
    fs = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    if not fs:
        raise SystemExit(f"no .safetensors weights in {path}")
    return {f: sha_file(os.path.join(path, f)) for f in fs}


def resolve_parent(cfg):
    """Initial weights plus provenance. init.path is a checkpoint dir or a hub id (with a pinned init.revision)."""
    init = cfg["init"]
    p = init["path"]
    if os.path.isdir(p):
        lin, actual = read_lineage(p), weights_hash(p)
        if lin and lin["weights_sha256"] != actual:
            raise SystemExit(f"checkpoint {p} does not match its lineage record")
        stage = lin["stage"] if lin else init.get("declared_stage", "base")
        if not lin:  # a folder without our lineage record (a cached base model, or a checkpoint from other code) must prove its identity
            pin = init["expect_sha256"] if "expect_sha256" in init else os.environ.get("PT_BASE_SHA256")
            if "expect_sha256" in init and not pin:
                raise SystemExit(f"{p} has no lineage record and expect_sha256 is empty; refusing unidentified weights")
            if pin and list(actual.values()) != [pin]:
                raise SystemExit(f"{p} has no lineage record and its weights are not the expected file (sha256 differs)")
        parent = dict(path=p, stage=stage, weights_sha256=actual, chain=lin["chain"] if lin else [stage])
    else:
        from huggingface_hub import snapshot_download
        p = snapshot_download(p, revision=init.get("revision"), local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")
        parent = dict(path=p, hub_id=init["path"], revision=init.get("revision"), stage="base", weights_sha256=weights_hash(p), chain=["base"])
    want = init.get("expect_stage")
    if want and parent["stage"] != want:
        raise SystemExit(f"init stage is {parent['stage']!r} but the config expects {want!r}; refusing a wrong-lineage run")
    return p, parent


def save_model(model, tok, path, stage, parent, run_meta, extra=None):
    """Atomic: temp dir, hash, rename. Weights stored in bf16 for inference; resume state keeps fp32."""
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    tmp = path + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    sd = {k: v.detach().to("cpu", torch.bfloat16) if v.is_floating_point() else v.detach().cpu() for k, v in model.state_dict().items()}
    if getattr(model.config, "tie_word_embeddings", False):
        sd.pop("lm_head.weight", None)
    model.save_pretrained(tmp, state_dict=sd, safe_serialization=True)
    tok.save_pretrained(tmp)
    cj = os.path.join(tmp, "config.json")
    c = json.load(open(cj, encoding="utf-8"))
    for k in ("torch_dtype", "dtype"):
        if k in c:
            c[k] = "bfloat16"
    json.dump(c, open(cj, "w", encoding="utf-8"), indent=1)
    lin = dict(stage=stage, chain=parent["chain"] + [stage], parent=parent, weights_sha256=weights_hash(tmp),
               config_sha256=run_meta["config_sha256"], code_sha256=run_meta["code_sha256"], created=time.time(), extra=extra or {})
    jdump(os.path.join(tmp, "lineage.json"), lin)
    os.replace(tmp, path)
    return lin


def save_state(path, model, opt, step, extra):
    import numpy as np
    tmp = path + ".tmp"
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, extra=extra, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    py_rng=random.getstate(), np_rng=np.random.get_state()), tmp)
    os.replace(tmp, path)


def load_state(path, model, opt):
    """weights_only=False is required for RNG/optimizer state; only load state files written by this code on this machine."""
    return restore_state(torch.load(path, map_location="cpu", weights_only=False), model, opt)


STATE_KEYS = ("model", "opt", "step", "extra", "torch_rng", "cuda_rng", "py_rng", "np_rng")


def restore_state(s, model, opt):
    """Model, optimizer and every RNG, from a dict written by save_state."""
    import numpy as np
    missing = [k for k in STATE_KEYS if k not in s]
    if missing:
        raise SystemExit(f"saved state is incomplete: {missing}")
    model.load_state_dict(s["model"])
    opt.load_state_dict(s["opt"])
    torch.set_rng_state(s["torch_rng"])
    random.setstate(s["py_rng"])
    np.random.set_state(s["np_rng"])
    if s["cuda_rng"] and len(s["cuda_rng"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(s["cuda_rng"])
    return s["step"], s["extra"]
