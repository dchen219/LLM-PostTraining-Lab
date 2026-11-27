"""Check that this release is the code that produced the frozen results, up to the documented package reorganisation.

The release moved the grader to posttrain/verifiers/ and the prompt templates to posttrain/prompts/, and edited the import lines
that refer to them (docs/PACKAGE_MIGRATION.md). docs/code_lineage.json lists, for every file of the original layout, its release
path and the exact edits. This script reverses those edits in memory (nothing is written) and checks:

  1. every reconstructed file has its original sha256;
  2. the original code hash, computed as the original posttrain.common.code_hash did, equals the code_sha256 in the checkpoint
     lock (results/run_records/a5_final/FINAL_CHECKPOINTS.json), i.e. the code version that ran the held-out evaluation;
  3. the reconstructed reward code (verify.py, drgrpo_grader.py) has the verifier hashes recorded by every training run.

python scripts/verify_code_lineage.py
"""
import glob, hashlib, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
L = json.load(open(os.path.join(ROOT, "docs", "code_lineage.json"), encoding="utf-8"))
sha = lambda b: hashlib.sha256(b).hexdigest()
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'ok' if cond else 'FAIL'}] {name}{': ' + detail if detail else ''}")


orig = {}
for f in L["files"]:
    b = b"" if f["release"] is None else open(os.path.join(ROOT, f["release"]), "rb").read().replace(b"\r\n", b"\n")
    t = b.decode("utf-8")
    for e in f["edits"]:
        n = t.count(e["release"])
        if n != 1:
            check(f"{f['original']}: edit applies once", False, f"found {n} times: {e['release'][:60]!r}")
        t = t.replace(e["release"], e["original"])
    orig[f["original"]] = t.encode("utf-8")
    check(f"{f['original']} <- {f['release'] or '(empty file, not shipped)'}", sha(orig[f["original"]]) == f["original_sha256"], f"{len(f['edits'])} edit(s) reversed")

h = hashlib.sha256()
for d in ("posttrain", "cs336_alignment"):  # the original algorithm: per directory, sorted file names, name + content
    for k in sorted(k for k in orig if k.startswith(d + "/")):
        h.update(k.split("/", 1)[1].encode() + orig[k])
lock = json.load(open(os.path.join(ROOT, "results", "run_records", "a5_final", "FINAL_CHECKPOINTS.json"), encoding="utf-8"))
check("original code hash == checkpoint lock", h.hexdigest() == lock["code_sha256"] == L["lock_code_sha256"], h.hexdigest()[:16] + "…")

ver = dict(verify_py=sha(orig["posttrain/verify.py"]), grader_py=sha(orig["cs336_alignment/drgrpo_grader.py"]))
runs = sorted(glob.glob(os.path.join(ROOT, "results", "run_records", "*", "*", "*", "run.json")))
recorded = {json.dumps({k: json.load(open(p))["verifier"][k] for k in ("verify_py", "grader_py")}, sort_keys=True) for p in runs}
check(f"reward code == verifier hashes recorded by all {len(runs)} run records", recorded == {json.dumps(ver, sort_keys=True)}, f"verify.py {ver['verify_py'][:12]}…, grader {ver['grader_py'][:12]}…")

from posttrain import verifier_lock  # noqa: E402
from posttrain.common import code_hash  # noqa: E402
from posttrain.verify import verifier_id  # noqa: E402
check("verifier_lock.ORIGINAL_SHA256 == reconstructed reward code", verifier_lock.ORIGINAL_SHA256 == ver)
check("release verifier == verifier_lock.SHA256", {k: v for k, v in verifier_id().items() if k != "version"} == verifier_lock.SHA256)
print(f"release code hash (posttrain/**/*.py): {code_hash()}")
print("LINEAGE_OK" if ok else "LINEAGE_FAILED")
sys.exit(0 if ok else 1)
