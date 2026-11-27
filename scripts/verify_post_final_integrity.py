"""Verify the post-evaluation integrity evidence (read-only).

Inputs (recovered from the server; never modified here):
  <final>/BEFORE_FINAL.sha256       `find sft grpo recovery -type f -exec sha256sum {} +` run in $PT_RUNS
                                    before the final evaluation (FINAL_EVAL_PLAN.md step 3)
  <final>/post_final_integrity.log  output of `sha256sum -c --quiet BEFORE_FINAL.sha256 && echo
                                    TRAINED_RUNS_UNCHANGED` after the evaluation (step 4)
  <final>/FINAL_CHECKPOINTS.json    the lock
  training-time manifests and local copies of the run trees (--runs, repeatable)

The server-side check itself cannot be re-run from the laptop. What this script establishes is
that the manifest the server checked against is the right one: complete, well-formed, consistent
with the lock and with the training-time manifests and the local copies of the run files.

Usage:
  python scripts/verify_post_final_integrity.py --final ~/a5_final --runs ~/a5_sessions \
      --runs ~/a5_recovery --out results/research_analysis/post_final_integrity
"""
import argparse
import hashlib
import json
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--final", required=True)
ap.add_argument("--runs", action="append", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
FIN = os.path.expanduser(a.final)
TREES = [os.path.expanduser(t) for t in a.runs]
os.makedirs(a.out, exist_ok=True)

checks, notes = [], []


def check(name, ok, detail=""):
    checks.append({"check": name, "ok": bool(ok), "detail": detail})


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_manifest(p):
    out, bad = {}, []
    for i, line in enumerate(open(p, encoding="utf-8").read().splitlines(), 1):
        m = re.fullmatch(r"([0-9a-f]{64})  (\S.*)", line)
        if not m:
            bad.append(i)
            continue
        out.setdefault(m.group(2), []).append(m.group(1))
    return out, bad


bf_path = os.path.join(FIN, "BEFORE_FINAL.sha256")
log_path = os.path.join(FIN, "post_final_integrity.log")
for p in (bf_path, log_path):
    check(f"present: {os.path.basename(p)}", os.path.isfile(p), p)
if not all(c["ok"] for c in checks):
    sys.exit("missing input")

# --- 1. the log -------------------------------------------------------------------------------
log = open(log_path, "rb").read()
check("log: exactly one line 'TRAINED_RUNS_UNCHANGED'", log.strip() == b"TRAINED_RUNS_UNCHANGED",
      repr(log[:200]))
check("log: no FAILED / WARNING lines from sha256sum --quiet",
      not re.search(rb"FAILED|WARNING|No such file", log))
notes.append("`sha256sum -c --quiet` prints nothing for matching files and `&& echo` runs only on exit "
             "status 0, so a log holding only TRAINED_RUNS_UNCHANGED is the complete expected output.")

# --- 2. the manifest's form ---------------------------------------------------------------------
bf, bad = read_manifest(bf_path)
n_lines = sum(len(v) for v in bf.values())
check("BEFORE_FINAL: every line is '<sha256>  <path>'", not bad, f"bad lines: {bad[:10]}")
check("BEFORE_FINAL: no duplicate paths", all(len(v) == 1 for v in bf.values()),
      str([k for k, v in bf.items() if len(v) > 1][:5]))
bf = {k: v[0] for k, v in bf.items()}
check("BEFORE_FINAL: paths relative, under sft/ grpo/ recovery/ only",
      all(re.match(r"(sft|grpo|recovery)/", k) and ".." not in k for k in bf), "")
check("BEFORE_FINAL: no final_eval outputs listed", not any("final_eval" in k for k in bf))
runs = sorted({"/".join(k.split("/")[:2]) for k in bf})

# --- 3. the lock ----------------------------------------------------------------------------------
lock = json.load(open(os.path.join(FIN, "FINAL_CHECKPOINTS.json")))
lock_rows = []
for name, c in lock["checkpoints"].items():
    w = c["weights_sha256"]["model.safetensors"]
    if "/runs/" not in c["path"]:
        lock_rows.append({"checkpoint": name, "path": None, "lock": w, "before_final": None,
                          "match": None})
        notes.append(f"{name} lives outside the runs tree ({c['path'].split('/hub/')[-1]}); it is not in "
                     "BEFORE_FINAL by design. Its hash is covered by the pre-run `check` and by every "
                     "summary.json recording the locked hash.")
        continue
    rel = c["path"].split("/runs/", 1)[1] + "/model.safetensors"
    got = bf.get(rel)
    lock_rows.append({"checkpoint": name, "path": rel, "lock": w, "before_final": got, "match": got == w})
    check(f"lock = BEFORE_FINAL weights: {name}", got == w, f"{rel}: lock {w[:16]} vs {str(got)[:16]}")
check("every locked trained run fully listed (DONE.json, run.json, checkpoint/)",
      all(any(k.startswith(r["path"].rsplit("/checkpoint/", 1)[0] + "/" + f) for k in bf)
          for r in lock_rows if r["path"] for f in ("DONE.json", "run.json", "checkpoint/config.json")))

# --- 4. training-time manifests and local copies --------------------------------------------------
manifests = {}
for t in TREES:
    for f in sorted(os.listdir(t)):
        if f.endswith(".sha256"):
            manifests[os.path.join(os.path.basename(t), f)] = read_manifest(os.path.join(t, f))[0]
cmp_rows = []
for mname, m in manifests.items():
    shared = [k for k in m if k in bf]
    diff = [k for k in shared if bf[k] not in m[k]]
    missing = [k for k in m if k not in bf]
    cmp_rows.append({"manifest": mname, "entries": len(m), "in_before_final": len(shared),
                     "hash_differs": len(diff), "absent_from_before_final": len(missing),
                     "absent_examples": missing[:8], "differs_examples": diff[:8]})
    check(f"{mname}: shared entries hash-identical to BEFORE_FINAL", not diff,
          f"{len(shared)} shared, {len(diff)} differ {diff[:3]}")

session_like = [m for n, m in manifests.items() if "BEFORE_RECOVERY" not in n]
union = set().union(*[set(m) for m in session_like]) if session_like else set()
extra = sorted(k for k in bf if k not in union)
absent = sorted(k for k in union if k not in bf)
check("every file in the session/recovery manifests is in BEFORE_FINAL", not absent, str(absent[:5]))

local_rows, n_local, n_local_bad = [], 0, []
for k, h in sorted(bf.items()):
    for t in TREES:
        p = os.path.join(t, k)
        if os.path.isfile(p):
            n_local += 1
            got = sha(p)
            if got != h:
                n_local_bad.append(k)
            break
    else:
        local_rows.append(k)
check("every BEFORE_FINAL file with a local copy re-hashes identically",
      not n_local_bad, f"{n_local} re-hashed locally, mismatches {n_local_bad[:5]}")

# files without a local copy (weights, optimizer state) against hashes recorded at training time
lineage_rows = []
for k in local_rows:
    run = k.split("/checkpoint/")[0] if "/checkpoint/" in k else k.rsplit("/", 1)[0]
    rec = None
    if k.endswith("checkpoint/model.safetensors"):
        for t in TREES:
            p = os.path.join(t, run, "DONE.json")
            if os.path.isfile(p):
                rec = json.load(open(p))["lineage"]["weights_sha256"]["model.safetensors"]
                src = "DONE.json lineage"
                break
    elif k == "grpo/A_base_s2026/state.pt":
        recs = set()
        for t in TREES:
            for r in ("recovery/A_recovery80_lr5e-6_s2026", "recovery/A_replay80_lr1e-5_s2026"):
                p = os.path.join(t, r, "run.json")
                if os.path.isfile(p):
                    recs |= set(re.findall(r'state_sha256": "([0-9a-f]{64})"', open(p).read()))
        rec = recs.pop() if len(recs) == 1 else None
        src = "branch run.json state_sha256 (step-80 parent)"
    lineage_rows.append({"path": k, "before_final": bf[k], "recorded": rec, "source": rec and src,
                         "match": rec == bf[k]})
    check(f"no local copy, matches training-time record: {k}", rec == bf[k],
          f"{bf[k][:16]} vs {str(rec)[:16]}")

# --- 5. the failed run is preserved ---------------------------------------------------------------
a_files = [k for k in bf if k.startswith("grpo/A_base_s2026/")]
check("failed GRPO-A_base_s2026 still present and has no DONE.json",
      a_files and "grpo/A_base_s2026/DONE.json" not in bf, f"{len(a_files)} files")

res = {
    "inputs": {os.path.basename(p): {"sha256": sha(p), "bytes": os.path.getsize(p)} for p in (bf_path, log_path)},
    "before_final": {"lines": n_lines,"runs": runs,
                     "files_per_run": {r: sum(k.startswith(r + "/") for k in bf) for r in runs}},
    "lock_vs_before_final": lock_rows,
    "manifest_comparison": cmp_rows,
    "extra_in_before_final_vs_session_manifests": extra,
    "absent_from_before_final_vs_session_manifests": absent,
    "local_rehash": {"rehashed": n_local, "mismatched": n_local_bad, "no_local_copy": local_rows},
    "no_local_copy_vs_training_records": lineage_rows,
    "checks": checks, "notes": notes,
    "failed": sum(not c["ok"] for c in checks),
}
json.dump(res, open(os.path.join(a.out, "post_final_integrity.json"), "w"), indent=1)
for c in checks:
    print(("ok   " if c["ok"] else "FAIL ") + c["check"] + (f"  [{c['detail']}]" if c["detail"] and not c["ok"] else ""))
print(f"\nlines {n_lines}; runs {len(runs)}; local re-hash {n_local}; no local copy {len(local_rows)}; "
      f"extra vs session manifests {len(extra)}")
for r in cmp_rows:
    print(r["manifest"], {k: v for k, v in r.items() if k not in ("manifest",)})
print("extra:", extra)
print("no local copy:", local_rows)
print("FAILED:", res["failed"])
