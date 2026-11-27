import hashlib, json, os, platform, random, re, sys, time
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENV = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def expand(v):
    """Expand ${VAR:default} recursively so every path is configurable without editing code."""
    if isinstance(v, dict):
        return {k: expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [expand(x) for x in v]
    if isinstance(v, str):
        def sub(m):
            got = os.environ.get(m[1], m[2])
            if got is None:
                raise KeyError(f"environment variable {m[1]} is required by the config")
            return got
        return _ENV.sub(sub, v)
    return v


def deep_merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_raw(path):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    base = cfg.pop("base", None)
    return deep_merge(load_raw(os.path.join(os.path.dirname(path), base)), cfg) if base else cfg


def load_config(path, overrides=()):
    cfg = load_raw(path)
    for o in overrides:
        k, v = o.split("=", 1)
        node = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"override {k} does not exist in the config")
        node[leaf] = yaml.safe_load(v)
    return expand(cfg)


def jload(path):
    t = open(path, encoding="utf-8").read()
    return json.loads(t) if t.lstrip()[:1] == "[" else [json.loads(x) for x in t.splitlines() if x.strip()]


def jdump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def jlines(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append(path, row):
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def code_hash():
    h = hashlib.sha256()
    # Release layout: every .py under posttrain/ (subpackages included), keyed by its path. The hash of the original layout,
    # recorded by the runs and the checkpoint lock, is reproduced by scripts/verify_code_lineage.py.
    fs = sorted(os.path.relpath(os.path.join(d, n), ROOT).replace(os.sep, "/") for d, _, ns in os.walk(os.path.join(ROOT, "posttrain")) for n in ns if n.endswith(".py"))
    for f in fs:
        h.update(f.encode() + open(os.path.join(ROOT, f), "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def seed_all(s):
    import numpy as np
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def env_info():
    import importlib.metadata as m
    import torch

    def v(p):
        try:
            return m.version(p)
        except m.PackageNotFoundError:
            return None
    return dict(python=sys.version.split()[0], platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
                gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                **{p: v(p) for p in ("transformers", "vllm", "math-verify", "sympy", "numpy")})


def incidents():
    """Recorded failed runs (recovery/incident_*.json). They are kept as evidence and are never resumed, overwritten or completed."""
    import glob
    return [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(ROOT, "recovery", "incident_*.json")))]


def refuse_failed_run(run_dir):
    runs = os.environ.get("PT_RUNS")
    for inc in incidents():
        if runs and os.path.realpath(run_dir) == os.path.realpath(os.path.join(runs, inc["run_dir_rel"])) and os.path.exists(os.path.join(run_dir, "run.json")):
            raise SystemExit(f"{run_dir} is the failed run {inc['experiment_id']} ({inc['status']}); it is preserved as evidence and is never resumed. "
                             "Post-hoc branches start from its checkpoint in new folders: python -m posttrain.fork_grpo")


TERMINAL = ("rollout log-prob check failed", "weight-sync check failed", "non-finite", "are not aligned")


class Run:
    """One immutable run directory: a finished run is never overwritten, an unfinished one resumes only on request.

    A run that stopped at a correctness gate writes FAILED.json; that is a finished outcome and is not resumed either.
    """

    def __init__(self, cfg, kind, resume=False):
        self.cfg, self.dir, self.t0 = cfg, cfg["run_dir"], time.time()
        if os.path.exists(os.path.join(self.dir, "DONE.json")):
            raise SystemExit(f"{self.dir} is already complete; refusing to overwrite")
        if os.path.exists(os.path.join(self.dir, "FAILED.json")):
            raise SystemExit(f"{self.dir} stopped at a correctness gate (FAILED.json); that is its outcome and it is not resumed")
        refuse_failed_run(self.dir)
        if os.path.isdir(self.dir) and os.listdir(self.dir) and not resume:
            raise SystemExit(f"{self.dir} exists; pass --resume or choose a new run_dir")
        os.makedirs(self.dir, exist_ok=True)
        start = os.path.join(self.dir, "run.json")
        from .verify import require_frozen  # every run records, and refuses to start without, the frozen verifier
        verifier = require_frozen()
        self.meta = dict(kind=kind, config=cfg, verifier=verifier, config_sha256=sha_obj(cfg), code_sha256=code_hash(), env=env_info(),
                         started=time.time(), argv=sys.argv)
        self.prior_wall = 0.0
        if os.path.exists(start):
            old = json.load(open(start, encoding="utf-8"))
            if old["config_sha256"] != self.meta["config_sha256"]:
                raise SystemExit("resume refused: config differs from the original run")
            append(os.path.join(self.dir, "resumes.jsonl"), dict(time=time.time(), code_sha256=self.meta["code_sha256"]))
            self.meta = old
            mp = os.path.join(self.dir, "metrics.jsonl")
            if os.path.exists(mp):
                self.prior_wall = max([r.get("wall_s", 0.0) for r in jload(mp)] or [0.0])
        else:
            jdump(start, self.meta)
        self.price = (cfg.get("cost") or {}).get("usd_per_hour")

    def log(self, row):
        row = dict(row, wall_s=round(self.prior_wall + time.time() - self.t0, 2))
        if self.price not in (None, "", "None"):
            row["usd"] = round(row["wall_s"] / 3600 * float(self.price), 4)
        append(os.path.join(self.dir, "metrics.jsonl"), row)
        print(json.dumps({k: (float(f"{v:.4g}") if isinstance(v, float) else v) for k, v in row.items()}, default=str), flush=True)

    def finish(self, **kw):
        jdump(os.path.join(self.dir, "DONE.json"), dict(finished=time.time(), wall_s=self.prior_wall + time.time() - self.t0, **kw))

    def fail(self, error, step, **kw):
        jdump(os.path.join(self.dir, "FAILED.json"), dict(failed=time.time(), wall_s=self.prior_wall + time.time() - self.t0, error=str(error),
                                                          completed_updates=step, attempted_step=step + 1, optimizer_updated_on_attempt=False, **kw))

    def rewind(self, step):
        """On resume from a state saved at `step`: rows and files written after it are moved aside (never deleted), so no step is logged twice.

        wall_s keeps counting the discarded time, so cost includes work that had to be repeated.
        """
        mp = os.path.join(self.dir, "metrics.jsonl")
        rows = jload(mp) if os.path.exists(mp) else []
        late = [r for r in rows if (r.get("step") or 0) > step]
        tag = f"superseded/resume_at_step{step}_{int(time.time())}"
        if late:
            for r in late:
                append(os.path.join(self.dir, "metrics.superseded.jsonl"), dict(r, superseded_by_resume_at=step))
            jlines(mp, [r for r in rows if (r.get("step") or 0) <= step])
        moved = 0
        for sub, rx in (("rollouts", r"step(\d+)\.jsonl$"), ("eval", r"step(\d+)_.*\.jsonl$")):
            d = os.path.join(self.dir, sub)
            for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
                m = re.match(rx, f)
                if m and int(m[1]) > step:
                    os.makedirs(os.path.join(self.dir, tag, sub), exist_ok=True)
                    os.replace(os.path.join(d, f), os.path.join(self.dir, tag, sub, f))
                    moved += 1
        return dict(rows=len(late), files=moved)


def parser(desc):
    import argparse
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--print-config", action="store_true")
    p.add_argument("--set", nargs="*", default=[], metavar="key=value")
    return p
