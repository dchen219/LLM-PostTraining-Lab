"""Run one registered experiment by id. Prints the command by default; --go executes it. GATED, HOLD_B and DEFERRED experiments also need --approve.

python scripts/run_experiment.py --id SFT-pilot_sft [--go] [--resume] [--set key=value ...]
"""
import argparse, csv, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from posttrain.hw import hw_overrides  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--id", required=True)
ap.add_argument("--go", action="store_true")
ap.add_argument("--approve", action="store_true")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
rows = {r["experiment_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "EXPERIMENT_REGISTRY.csv"), encoding="utf-8"))}
r = rows.get(a.id) or sys.exit(f"unknown id {a.id}")
if not r["module"].startswith("posttrain."):
    sys.exit(f"{a.id} is not launched from here ({r['module'] or 'already done locally'})")
if r["status"] in ("GATED", "HOLD_B", "DEFERRED") and not a.approve:
    sys.exit(f"{a.id} is {r['status']} (depends on: {r['depends_on']}). Pass --approve only after that decision has been made.")
sets = hw_overrides(r["module"], os.environ.get("PT_HW")) + a.set  # an explicit --set wins
if os.environ.get("PT_PLATFORM") == "research_server":
    if not os.environ.get("PT_HW"):
        sys.exit("PT_HW is not set: refusing to use the 80 GB settings on this machine")
    if a.go and os.environ.get("A5_AUTHORIZED") != "yes":
        sys.exit("REFUSED: server use has not been authorized (A5_AUTHORIZED is not 'yes')")
cmd = [sys.executable, "-m", r["module"], "--config", os.path.join(ROOT, r["config"])] + (["--resume"] if a.resume else []) + (["--set"] + sets if sets else [])
print("depends on:", r["depends_on"] or "nothing", "\n" + " ".join(cmd))
if a.go:
    sys.exit(subprocess.call(cmd, cwd=ROOT))
