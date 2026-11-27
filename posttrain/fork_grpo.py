"""Post-hoc recovery branches of a failed GRPO run, each in a NEW folder, started from one of its verified saved states.

python -m posttrain.fork_grpo --spec recovery/<name>.yaml --check     verify the source and print every difference; trains nothing
python -m posttrain.fork_grpo --spec recovery/<name>.yaml [--resume]  train the branch (an interrupted branch resumes from its own state)

What it guarantees
- The branch configuration IS the source run's recorded configuration (run.json, whose hash is re-checked), plus only the changes the spec
  lists, from ALLOWED. Every difference is computed, refused if not allowed, and recorded in the branch's run.json and lineage.
- The source state file must have the recorded sha256 and the recorded step, and contain model, optimizer, step, sampler and all RNG states.
  It is only read. At the end (finished or stopped), its hash and every file of the source run folder are checked again.
- The source run folder is never opened for writing; the original Run class protections are unchanged (the failed run itself is refused).
- The branch folder must be new and under $PT_RUNS/recovery/. A finished or gate-stopped branch is never resumed or overwritten.
- No duplicate step rows: the branch logs only steps after the source step; on resume, rows after the restored step are moved aside.
These branches are diagnostic. They do not replace, and are not reported as, the pre-registered run.
"""
import argparse, copy, hashlib, json, os, sys
import yaml
from .common import expand, sha_file, sha_obj

ALLOWED = {"train.lr": "intervention (learning-rate sensitivity)",
           "train.save_every": "operational: more resume points inside the branch; no effect on any update",
           "grpo.save_rollouts_every": "instrumentation: every step's rollouts are written to disk; no effect on any update or random number"}
CRITERIA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recovery", "diagnostic_criteria.yaml")


def get(cfg, key):
    for k in key.split("."):
        cfg = cfg[k]
    return cfg


def put(cfg, key, v):
    *ps, leaf = key.split(".")
    for k in ps:
        cfg = cfg[k]
    if leaf not in cfg:
        raise SystemExit(f"{key} does not exist in the source configuration")
    cfg[leaf] = v


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def differences(a, b):
    fa, fb = flat(a), flat(b)
    return [dict(key=k, source=fa.get(k), branch=fb.get(k)) for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]


def folder_digest(d, skip=("state.pt",)):
    """sha256 of every file in the source run folder (the large state file by size and mtime; it is hashed separately)."""
    rows = []
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            st = os.stat(p)
            rows.append([os.path.relpath(p, d).replace(os.sep, "/"), st.st_size, st.st_mtime_ns if f in skip else sha_file(p)])
    return dict(files=len(rows), sha256=sha_obj(rows))


def state_digest(s):
    """Content hash of the model and optimizer tensors as loaded: two branches that start from identical states have equal digests."""
    import torch
    h = hashlib.sha256()

    def add(x):
        if torch.is_tensor(x):
            t = x.detach().cpu().contiguous()
            h.update(f"{t.dtype}{tuple(t.shape)}".encode())
            h.update(t.view(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        elif isinstance(x, dict):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                add(x[k])
        elif isinstance(x, (list, tuple)):
            for y in x:
                add(y)
        else:
            h.update(repr(x).encode())
    add(s["model"])
    add(s["opt"])
    return h.hexdigest()


class Fork:
    def __init__(self, info):
        self.info, self.step, self.state = info, info["source_step"], info["source_state"]

    def verify(self):
        """Hash, load and check the source state BEFORE the branch folder is created. It is then held in memory, so it cannot change after the check."""
        import torch
        from .model_io import STATE_KEYS
        h = sha_file(self.state)
        if h != self.info["source_state_sha256"]:
            raise SystemExit(f"source state {self.state} has sha256 {h}, expected {self.info['source_state_sha256']}; refusing")
        s = torch.load(self.state, map_location="cpu", weights_only=False)
        if s.get("step") != self.step:
            raise SystemExit(f"source state is at step {s.get('step')}, expected {self.step}; refusing")
        missing = [k for k in STATE_KEYS if k not in s] + [k for k in ("sampler_rng", "sampler_p") if k not in (s.get("extra") or {})]
        if missing:
            raise SystemExit(f"source state is incomplete: {missing}; refusing")
        self.s, self.h = s, h
        return h

    def start(self, model, opt):
        """Restore model, optimizer, step, sampler and every RNG from the verified source state. Returns (step, extra, proof)."""
        from .model_io import restore_state
        if getattr(self, "s", None) is None:
            self.verify()
        s, self.s = self.s, None
        dig = state_digest(s)
        lr_saved = [g["lr"] for g in s["opt"]["param_groups"]]
        step, extra = restore_state(s, model, opt)
        return step, extra, dict(source_state=self.state, source_state_sha256=self.h, state_digest=dig, saved_optimizer_lr=lr_saved,
                                 optimizer_tensors=len(s["opt"]["state"]))

    def check_source(self):
        """After the branch: the source state and the source run folder must be byte-for-byte what they were before it."""
        h = sha_file(self.state)
        f = folder_digest(self.info["source_run_dir"])
        ok = h == self.info["source_state_sha256"] and f == self.info["source_folder"]
        if not ok:
            print("WARNING: the source state or the source run folder changed while this branch existed", flush=True)
        return dict(source_unchanged=ok, source_state_sha256_after=h, source_folder_after=f)


def build(spec_path):
    """(branch config, Fork). Refuses anything that would make the branch differ from the source in a way the spec does not declare."""
    from .verify import require_frozen
    sp = expand(yaml.safe_load(open(spec_path, encoding="utf-8")))
    src = sp["source"]
    sd = src["run_dir"]
    rj = json.load(open(os.path.join(sd, "run.json"), encoding="utf-8"))
    if sha_obj(rj["config"]) != rj["config_sha256"]:
        raise SystemExit("the source run.json configuration does not match its own recorded hash; refusing")
    if os.path.exists(os.path.join(sd, "DONE.json")):
        raise SystemExit(f"{sd} completed; branches are only for runs that stopped")
    for k, v in (sp.get("expect_source") or {}).items():
        if get(rj["config"], k) != v:
            raise SystemExit(f"source {k} is {get(rj['config'], k)!r}, the spec expects {v!r}; refusing")
    if rj.get("verifier") != require_frozen():
        raise SystemExit("the source run used another verifier; branches must score with the same frozen verifier")
    runs = os.path.realpath(os.environ["PT_RUNS"])
    rd = os.path.realpath(sp["run_dir"])
    if not rd.startswith(os.path.join(runs, "recovery") + os.sep):
        raise SystemExit(f"branch folders live under $PT_RUNS/recovery/, not {rd}")
    if os.path.realpath(sd) in (rd, os.path.dirname(rd)) or rd.startswith(os.path.realpath(sd) + os.sep):
        raise SystemExit("the branch folder must be separate from the source run")
    sp_state = os.path.realpath(src["state"])
    if sp_state.startswith(rd + os.sep):
        raise SystemExit("the source state may not live inside the branch folder")
    cfg = copy.deepcopy(rj["config"])
    for k, v in (sp.get("changes") or {}).items():
        if k not in ALLOWED:
            raise SystemExit(f"{k} may not be changed in a recovery branch; allowed: {sorted(ALLOWED)}")
        put(cfg, k, v)
    cfg["run_dir"] = sp["run_dir"]
    diff = differences(rj["config"], cfg)
    bad = [d["key"] for d in diff if d["key"] not in ALLOWED and d["key"] != "run_dir"]
    if bad:
        raise SystemExit(f"undeclared differences from the source: {bad}")
    met = [json.loads(x) for x in open(os.path.join(sd, "metrics.jsonl"), encoding="utf-8") if x.strip()]
    logged = sorted({r["step"] for r in met if r.get("event") == "train"})
    info = dict(id=sp["id"], purpose=sp.get("purpose", ""), posthoc=True, replaces_registered_run=False, source_experiment=src["experiment_id"],
                source_run_dir=sd, source_config_sha256=rj["config_sha256"], source_code_sha256=rj["code_sha256"], source_state=src["state"],
                source_state_sha256=src["state_sha256"], source_step=int(src["step"]), source_completed_updates=int(src["completed_updates"]),
                source_logged_train_steps=[logged[0], logged[-1], len(logged)] if logged else None, source_folder=folder_digest(sd),
                differences=[dict(d, why=ALLOWED.get(d["key"], "new folder")) for d in diff])
    crit = yaml.safe_load(open(CRITERIA, encoding="utf-8"))
    info["criteria"] = dict(file="recovery/diagnostic_criteria.yaml", sha256=sha_file(CRITERIA), version=crit["version"], frozen=crit["frozen"])
    if not logged or logged[-1] != info["source_completed_updates"]:
        raise SystemExit(f"source metrics end at step {logged[-1] if logged else None}, the spec says {info['source_completed_updates']} updates were completed")
    if info["source_step"] >= info["source_completed_updates"] or info["source_step"] % int(get(rj["config"], "train.save_every")):
        raise SystemExit("source_step must be a save point before the failure")
    cfg["fork"] = info
    return cfg, Fork(info)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if os.environ.get("PT_PLATFORM") == "research_server" and not a.check:
        if os.environ.get("A5_AUTHORIZED") != "yes" or os.environ.get("A5_RECOVERY_APPROVED") != "yes":
            raise SystemExit("REFUSED: recovery branches need A5_AUTHORIZED=yes and A5_RECOVERY_APPROVED=yes")
    cfg, fork = build(a.spec)
    if a.check:
        h = sha_file(fork.state)
        ok = h == fork.info["source_state_sha256"]
        print(json.dumps(dict(id=fork.info["id"], run_dir=cfg["run_dir"], exists=os.path.exists(cfg["run_dir"]), source_state_sha256_ok=ok,
                              differences=fork.info["differences"], lr=cfg["train"]["lr"], micro_batch=cfg["train"]["micro_batch"],
                              token_budget=cfg["train"]["token_budget"], steps=cfg["grpo"]["steps"], rollout_tol=cfg["grpo"]["rollout_tol"]), indent=1))
        sys.exit(0 if ok else 1)
    if not (a.resume and os.path.exists(os.path.join(cfg["run_dir"], "state.pt"))):  # the branch will start from the source: check it first
        fork.verify()
    from .train_grpo import train
    train(cfg, a.resume, fork)


if __name__ == "__main__":
    main()
