"""Held-out (final) evaluation of the checkpoints frozen in final/freeze_v1.yaml. Nothing else can be evaluated on the locked sets from here.

python -m posttrain.final_eval --plan            list the jobs; reads nothing on disk but the freeze file
python -m posttrain.final_eval --check           every listed checkpoint: DONE.json, no FAILED.json, lineage stage, weights sha256 = DONE.json lineage
python -m posttrain.final_eval --lock            write $PT_RUNS/final_eval/FINAL_CHECKPOINTS.json (freeze sha256 + every checkpoint's sha256) BEFORE any evaluation
python -m posttrain.final_eval --run --worker 0 --workers 2     evaluate this worker's share; needs the lock and A5_FINAL_EVAL_APPROVED=yes

The lock is written once. An existing lock that differs (other freeze file, other weights) is refused, never replaced. Before each checkpoint is
evaluated its weights are hashed again and must equal the lock. ALLOW_FINAL_EVAL=1 is set only in the evaluation subprocess.
"""
import argparse, json, os, subprocess, sys, time

from .common import ROOT, code_hash, jdump, sha_file

FREEZE = os.path.join(ROOT, "final", "freeze_v1.yaml")
TUNING = ("SFT-tune_", "GRPO-tune_")
FAILED_ORIGINAL = "GRPO-A_base_s2026"


def load_freeze(path=FREEZE):
    import yaml
    return yaml.safe_load(open(path, encoding="utf-8")), sha_file(path)


def validate(fz):
    """Structural rules of the frozen plan. Returns a list of problems (empty = valid)."""
    bad = []
    ck = {c["id"]: c for c in fz["checkpoints"]}
    if len(ck) != len(fz["checkpoints"]):
        bad.append("duplicate checkpoint id")
    for c in fz["checkpoints"]:
        if c["id"].startswith(TUNING):
            bad.append(f"{c['id']}: tuning runs are not final checkpoints")
        if c["id"] == FAILED_ORIGINAL:
            bad.append(f"{FAILED_ORIGINAL} failed and has no final step; it cannot be evaluated")
        if c["class"] not in fz["classes"]:
            bad.append(f"{c['id']}: unknown class {c['class']}")
        if c["class"] == "posthoc" and not str(c.get("dir") or "").startswith("recovery/"):
            bad.append(f"{c['id']}: post-hoc checkpoints live under recovery/")
        if c["class"] == "preregistered" and str(c.get("dir") or "").startswith("recovery/"):
            bad.append(f"{c['id']}: a recovery branch cannot be pre-registered")
        if (c["id"] == "BASE") != (c["dir"] is None):
            bad.append(f"{c['id']}: only BASE has no run folder")
    if FAILED_ORIGINAL not in {x["id"] for x in fz.get("not_evaluated", [])}:
        bad.append(f"{FAILED_ORIGINAL} must be listed as not evaluated")
    for fam, f in fz["families"].items():
        comps = f["comparisons"]
        if f["holm_m"] is not None and f["holm_m"] != len(comps):
            bad.append(f"{fam}: holm_m {f['holm_m']} != {len(comps)} comparisons")
        for c in comps:
            for side in ("a", "b"):
                cid, prompt = c[side].split(":")
                if cid not in ck:
                    if not (c.get("unavailable") and cid == FAILED_ORIGINAL):
                        bad.append(f"{fam}/{c['name']}: {cid} is not a frozen checkpoint")
                    continue
                if prompt not in ck[cid]["prompts"]:
                    bad.append(f"{fam}/{c['name']}: {cid} is not evaluated with {prompt}")
                if fam != "posthoc" and ck[cid]["class"] == "posthoc":
                    bad.append(f"{fam}/{c['name']}: a post-hoc checkpoint appears in a pre-registered family")
    return bad


def jobs(fz):
    return [dict(id=c["id"], dir=c["dir"], stage=c["stage"], prompts=c["prompts"], cls=c["class"]) for c in fz["checkpoints"]]


def out_dir(runs, j):
    return os.path.join(runs, "final_eval", j["id"])


def ckpt_path(runs, j):
    return None if j["dir"] is None else os.path.join(runs, j["dir"], "checkpoint")


def check_one(runs, j):
    """Identity of one frozen checkpoint. Returns a record; raises SystemExit on any problem."""
    from .model_io import read_lineage, weights_hash
    if j["dir"] is None:
        from .common import load_config
        from .model_io import resolve_parent
        _, parent = resolve_parent(load_config(os.path.join(ROOT, "configs", "eval", "final.yaml")))
        pin = os.environ.get("PT_BASE_SHA256")
        if pin and list(parent["weights_sha256"].values()) != [pin]:
            raise SystemExit("BASE: weights do not match PT_BASE_SHA256")
        return dict(id=j["id"], path=parent["path"], stage="base", weights_sha256=parent["weights_sha256"])
    d = os.path.join(runs, j["dir"])
    if os.path.exists(os.path.join(d, "FAILED.json")):
        raise SystemExit(f"{j['id']}: FAILED.json present; a failed run is never evaluated")
    if not os.path.exists(os.path.join(d, "DONE.json")):
        raise SystemExit(f"{j['id']}: no DONE.json; only finished runs are evaluated")
    done = json.load(open(os.path.join(d, "DONE.json"), encoding="utf-8"))
    lin, p = done["lineage"], ckpt_path(runs, j)
    if lin["stage"] != j["stage"]:
        raise SystemExit(f"{j['id']}: lineage stage {lin['stage']!r}, expected {j['stage']!r}")
    disk = read_lineage(p)
    if disk is None or disk["weights_sha256"] != lin["weights_sha256"]:
        raise SystemExit(f"{j['id']}: checkpoint lineage.json does not match DONE.json")
    actual = weights_hash(p)
    if actual != lin["weights_sha256"]:
        raise SystemExit(f"{j['id']}: checkpoint weights differ from DONE.json (sha256)")
    return dict(id=j["id"], path=p, stage=lin["stage"], weights_sha256=actual, steps=done.get("steps"), wall_s=done.get("wall_s"))


def lock_path(runs):
    return os.path.join(runs, "final_eval", "FINAL_CHECKPOINTS.json")


def make_lock(runs, fz, fsha, records):
    return dict(freeze_file="final/freeze_v1.yaml", freeze_sha256=fsha, freeze_version=fz["version"], code_sha256=code_hash(),
                checkpoints={r["id"]: dict(path=r["path"], stage=r["stage"], weights_sha256=r["weights_sha256"]) for r in records})


def write_lock(runs, lock):
    p = lock_path(runs)
    if os.path.exists(p):
        old = json.load(open(p, encoding="utf-8"))
        same = old["freeze_sha256"] == lock["freeze_sha256"] and old["checkpoints"] == lock["checkpoints"]
        if not same:
            raise SystemExit(f"{p} exists and differs (freeze or weights); refusing to replace a lock")
        return old
    os.makedirs(os.path.dirname(p), exist_ok=True)
    jdump(p, dict(lock, locked=time.time()))
    return lock


def read_lock(runs, fsha):
    p = lock_path(runs)
    if not os.path.exists(p):
        raise SystemExit("no FINAL_CHECKPOINTS.json: run --lock first")
    lk = json.load(open(p, encoding="utf-8"))
    if lk["freeze_sha256"] != fsha:
        raise SystemExit("the lock was written for a different freeze file; refusing")
    return lk


def eval_cmd(runs, j, rec):
    s = [f"run_dir={out_dir(runs, j)}", f"prompts={json.dumps(j['prompts'])}"]
    if j["dir"] is not None:
        s += [f"init.path={rec['path']}", f"init.expect_stage={j['stage']}"]
    return [sys.executable, "-m", "posttrain.evaluate", "--config", os.path.join(ROOT, "configs", "eval", "final.yaml"), "--set"] + s


def run_worker(runs, fz, fsha, worker, workers, call=subprocess.call):
    if os.environ.get("A5_FINAL_EVAL_APPROVED") != "yes":
        raise SystemExit("REFUSED: final evaluation has not been approved (A5_FINAL_EVAL_APPROVED is not 'yes')")
    lk = read_lock(runs, fsha)
    mine = [j for i, j in enumerate(jobs(fz)) if i % workers == worker]
    for j in mine:
        od = out_dir(runs, j)
        if os.path.exists(os.path.join(od, "DONE.json")):
            print(f"skip {j['id']} (evaluated)")
            continue
        rec = check_one(runs, j)
        if rec["weights_sha256"] != lk["checkpoints"][j["id"]]["weights_sha256"]:
            raise SystemExit(f"{j['id']}: weights changed since the lock; refusing")
        cmd = eval_cmd(runs, j, rec) + (["--resume"] if os.path.isdir(od) and os.listdir(od) else [])
        print(f"=== final evaluation {j['id']} ({j['cls']}) {time.strftime('%F %T')}", flush=True)
        rc = call(cmd, cwd=ROOT, env=dict(os.environ, ALLOW_FINAL_EVAL="1"))
        if rc != 0:
            raise SystemExit(f"{j['id']}: evaluation exited with {rc}")
    return [j["id"] for j in mine]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    for m in ("plan", "check", "lock", "run"):
        g.add_argument("--" + m, action="store_true")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    fz, fsha = load_freeze()
    bad = validate(fz)
    if bad:
        raise SystemExit("REFUSED: freeze file invalid: " + "; ".join(bad))
    print(f"freeze {fz['version']} sha256 {fsha}")
    if a.plan:
        for i, j in enumerate(jobs(fz)):
            print(f"  worker {i % max(1, a.workers)}  {j['cls']:<13} {j['id']:<32} {','.join(j['prompts'])}")
        for x in fz["not_evaluated"]:
            print(f"  not evaluated  {x['id']}: {x['why']}")
        return
    runs = os.environ.get("PT_RUNS") or sys.exit("PT_RUNS is not set: source scripts/server_env.sh")
    if a.check or a.lock:
        recs = [check_one(runs, j) for j in jobs(fz)]
        for r in recs:
            print(f"  ok  {r['id']:<32} {r['stage']:<5} {next(iter(r['weights_sha256'].values()))[:16]}")
        if a.lock:
            if os.environ.get("A5_FINAL_EVAL_APPROVED") != "yes":
                raise SystemExit("REFUSED: final evaluation has not been approved (A5_FINAL_EVAL_APPROVED is not 'yes')")
            write_lock(runs, make_lock(runs, fz, fsha, recs))
            print("LOCKED", lock_path(runs))
        else:
            print("CHECK_OK (final)")
        return
    done = run_worker(runs, fz, fsha, a.worker, a.workers)
    print(f"WORKER {a.worker} DONE: {', '.join(done)}")


if __name__ == "__main__":
    main()
