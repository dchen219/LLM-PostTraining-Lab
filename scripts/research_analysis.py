"""Post-evaluation research analysis: audit, independent reproduction of the frozen held-out statistics, and analysis tables. Reads only.

python scripts/research_analysis.py --final ~/a5_final --runs ~/a5_sessions --runs ~/a5_recovery \
    --report results/final_eval/report.json --out results/research_analysis
Writes AUDIT.md, audit.json, analysis.json, consolidated_results.csv, statistical_comparisons.csv and data/*.csv (the plotting data for
scripts/research_figures.py). Nothing under --final or --runs is written. The frozen plan (final/freeze_v1.yaml) is read, never changed.

Labels used in every table:
  confirmatory            pre-registered family "confirmatory" (C1-C4), Holm m = 4 per set
  confirmatory-secondary  pre-registered family "sft_vs_base", Holm m = 7 per set
  descriptive             pre-registered descriptive contrasts: CI and unadjusted p, no claim rests on them
  post-hoc (frozen)       the frozen post-hoc family (A1/A2), Holm m = 4 per set
  post-hoc (unregistered) contrasts chosen after seeing the held-out results (this script), own Holm family per set
  sensitivity             the frozen confirmatory family re-computed on relaxed accuracy
  exploratory (dev)       development-set analyses (training curves, compute)
McNemar p-values and CIs cover sampling of evaluation questions only. One training seed per arm: no CI here covers training-seed variance.
"""
import argparse, csv, hashlib, json, logging, os, statistics as st, subprocess, sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
logging.disable(logging.CRITICAL)
import numpy as np  # noqa: E402
from scipy.stats import binomtest, spearmanr, kendalltau  # noqa: E402
from posttrain.common import jload  # noqa: E402
from posttrain.final_eval import FREEZE, load_freeze  # noqa: E402
from posttrain import verifier_lock  # noqa: E402
import stage0_verify_audit as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--final", required=True)
ap.add_argument("--runs", action="append", required=True)
ap.add_argument("--report", default=os.path.join(ROOT, "results/final_eval/report.json"))
ap.add_argument("--out", required=True)
ap.add_argument("--skip-training-audit", action="store_true")
a = ap.parse_args()
FIN = os.path.expanduser(a.final)
TREES = [os.path.expanduser(t) for t in a.runs]
O = a.out
assert not os.path.exists(O), "refusing to overwrite"
os.makedirs(O + "/data")
J = lambda p: json.load(open(p, encoding="utf-8"))
SETS = ("gsm8k_test_final", "math_val_final")
DEVSETS = ("gsm8k_dev", "math_dev")
EXPECTED_FREEZE = "ff0f0ce574bcb47c9dafedc77bab1654362bb955a81c593960108ca682b7a453"
BASE_SHA = "f7aa8d665ab44d3b461bed742fe93cdcedc15d2e31e50dc372c126e6f720857b"
SHORT = {"BASE": "Base", "SFT-exact_full_s0": "SFT exact full", "SFT-reverified_full_s0": "SFT re-verified full", "SFT-reverified_sizecontrol_s0": "SFT re-verified size control",
         "SFT-exact_n512_s0": "SFT exact n512", "SFT-reverified_n512_s0": "SFT re-verified n512", "SFT-reverified_n128_s0": "SFT re-verified n128",
         "SFT-reverified_n2048_s0": "SFT re-verified n2048", "GRPO-B_sft512_s2026": "GRPO-B (sequential)", "GRPO-A-replay80-lr1e-5-s2026": "A1 (post-hoc, lr 1e-5)",
         "GRPO-A-recovery80-lr5e-6-s2026": "A2 (post-hoc, lr 5e-6)"}
META = {  # data source, SFT examples, stage; from the registry (EXPERIMENT_PLAN.md sections 4 and 10)
    "BASE": ("-", 0), "SFT-exact_full_s0": ("exact", 3420), "SFT-reverified_full_s0": ("reverified", 4423), "SFT-reverified_sizecontrol_s0": ("reverified", 3420),
    "SFT-exact_n512_s0": ("exact", 512), "SFT-reverified_n512_s0": ("reverified", 512), "SFT-reverified_n128_s0": ("reverified", 128), "SFT-reverified_n2048_s0": ("reverified", 2048),
    "GRPO-B_sft512_s2026": ("reverified", 512), "GRPO-A-replay80-lr1e-5-s2026": ("-", 0), "GRPO-A-recovery80-lr5e-6-s2026": ("-", 0)}

checks, problems, unverifiable, notes = [], [], [], []


def check(area, name, ok, detail=""):
    checks.append(dict(area=area, check=name, ok=bool(ok), detail=str(detail)))
    if not ok:
        problems.append(f"[{area}] {name}: {detail}")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find(rel):
    for t in TREES:
        if os.path.isdir(os.path.join(t, rel)):
            return os.path.join(t, rel)
    return None


def wcsv(path, rows, cols=None):
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})


# ======================================================================= 1. audit of the held-out evaluation artifacts
fz, fsha = load_freeze(FREEZE)
check("freeze", "final/freeze_v1.yaml sha256 equals the frozen value", fsha == EXPECTED_FREEZE, fsha)
lock = J(os.path.join(FIN, "FINAL_CHECKPOINTS.json"))
check("freeze", "lock written for this freeze file", lock["freeze_sha256"] == fsha, lock["freeze_sha256"])
REP = J(a.report)
check("freeze", "report.json written for this freeze file", REP["freeze_sha256"] == fsha, REP["freeze_sha256"])
check("report", "report lists no missing evaluation and no problem", not REP["missing"] and not REP["problems"], f"missing={REP['missing']} problems={REP['problems']}")

# manifest of the evaluation outputs (written on the server after the run)
man = {}
for line in open(os.path.join(FIN, "FINAL_EVAL.sha256"), encoding="utf-8"):
    h, p = line.split(maxsplit=1)
    man[p.strip().removeprefix("final_eval/")] = h
on_disk = sorted(os.path.relpath(os.path.join(dp, f), FIN).replace(os.sep, "/") for dp, _, fs in os.walk(FIN) for f in fs if f != "FINAL_EVAL.sha256")
check("manifest", "FINAL_EVAL.sha256 lists exactly the recovered files", sorted(man) == on_disk, f"{len(man)} listed, {len(on_disk)} on disk")
bad = [p for p, h in man.items() if not os.path.exists(os.path.join(FIN, p)) or sha_file(os.path.join(FIN, p)) != h]
check("manifest", "every recovered evaluation file matches FINAL_EVAL.sha256", not bad, f"{len(man) - len(bad)}/{len(man)} match; bad={bad}")
check("manifest", "FINAL_CHECKPOINTS.json (the lock) is covered by FINAL_EVAL.sha256", "FINAL_CHECKPOINTS.json" in man)

# lock contents against the freeze and the training lineage
ids = [c["id"] for c in fz["checkpoints"]]
check("lock", "lock lists exactly the frozen checkpoints", sorted(lock["checkpoints"]) == sorted(ids), sorted(lock["checkpoints"]))
check("lock", "failed GRPO-A_base_s2026 absent from the lock and from the evaluation folder",
      "GRPO-A_base_s2026" not in lock["checkpoints"] and not os.path.exists(os.path.join(FIN, "GRPO-A_base_s2026")))
INC = J(os.path.join(ROOT, "recovery", "incident_GRPO-A_base_s2026.json"))
gA = find("grpo/A_base_s2026")
check("lock", "GRPO-A_base_s2026 recorded FAILED (incident record) with no DONE.json", INC["status"] == "FAILED" and gA and not os.path.exists(gA + "/DONE.json"),
      f"incident status {INC['status']}, attempted step {INC['attempted_step']}, gate {INC['gate']['value']} > {INC['gate']['tolerance']}")
check("lock", "BASE weights equal the registered base model", lock["checkpoints"]["BASE"]["weights_sha256"]["model.safetensors"] == BASE_SHA)
for c in fz["checkpoints"]:
    if not c["dir"]:
        continue
    d = find(c["dir"])
    if d is None or not os.path.exists(d + "/DONE.json"):
        check("lock", f"{c['id']}: training DONE.json present", False, c["dir"])
        continue
    lin = J(d + "/DONE.json")["lineage"]
    check("lock", f"{c['id']}: locked weights = training DONE.json lineage weights", lin["weights_sha256"] == lock["checkpoints"][c["id"]]["weights_sha256"],
          lin["weights_sha256"]["model.safetensors"][:16])
    ck = d + "/checkpoint/lineage.json"
    if os.path.exists(ck):
        check("lock", f"{c['id']}: checkpoint/lineage.json agrees", J(ck).get("weights_sha256") == lin["weights_sha256"])
    if os.path.exists(d + "/checkpoint/model.safetensors"):
        check("lock", f"{c['id']}: local weight file hashes to the locked value", sha_file(d + "/checkpoint/model.safetensors") == lin["weights_sha256"]["model.safetensors"])
    if c["id"] == "GRPO-B_sft512_s2026":
        check("lock", "GRPO-B parent = SFT-reverified_n512_s0 weights (chain base>sft>grpo)",
              lin["parent"]["weights_sha256"] == lock["checkpoints"]["SFT-reverified_n512_s0"]["weights_sha256"] and lin["chain"] == ["base", "sft", "grpo"])
if not any(os.path.exists((find(c["dir"]) or "") + "/checkpoint/model.safetensors") for c in fz["checkpoints"] if c["dir"]):
    unverifiable.append("Weight files are not on the laptop. Weight identity rests on the server-side `check` (11 hashes printed and matched before the run) "
                        "and on every evaluation's summary.json recording the locked hash; it was not re-hashed locally.")

# per-evaluation metadata
manifest = J(os.path.join(ROOT, "data/stage0/MANIFEST.json"))["output_sha256"]
started, cfgs, codes, ids_by_set = {}, {}, set(), {s: None for s in SETS}
VL = verifier_lock.SHA256
for c in fz["checkpoints"]:
    d = os.path.join(FIN, c["id"])
    run, summ, done = J(d + "/run.json"), J(d + "/summary.json"), J(d + "/DONE.json")
    started[c["id"]] = run["started"]
    codes.add(run["code_sha256"])
    check("eval", f"{c['id']}: evaluated weights = lock", summ["parent"]["weights_sha256"] == lock["checkpoints"][c["id"]]["weights_sha256"])
    check("eval", f"{c['id']}: lineage stage = frozen stage", summ["parent"]["stage"] == c["stage"], summ["parent"]["stage"])
    check("eval", f"{c['id']}: verifier {fz['verifier']} with locked file hashes",
          run["verifier"]["version"] == fz["verifier"] == verifier_lock.VERSION and all(run["verifier"][k] == v for k, v in VL.items()))
    e = run["config"]["engine"]
    check("eval", f"{c['id']}: engine settings = freeze", e["name"] == fz["engine"]["name"] and e["max_model_len"] == fz["engine"]["max_model_len"]
          and e["prefix_caching"] == fz["engine"]["prefix_caching"] and e["gpu_mem"] == fz["engine"]["gpu_mem"] and run["config"]["max_tokens"] == fz["engine"]["max_tokens"]
          and run["config"]["seed"] == fz["engine"]["seed"] and run["env"]["gpus"] == [fz["engine"]["gpu_type"]], e)
    check("eval", f"{c['id']}: prompts = freeze", run["config"]["prompts"] == c["prompts"], run["config"]["prompts"])
    rest = {k: v for k, v in run["config"].items() if k not in ("run_dir", "init", "prompts")}
    cfgs[c["id"]] = json.dumps(rest, sort_keys=True)
    for pr in c["prompts"]:
        for s in SETS:
            g = summ["results"][pr][s]
            check("eval", f"{c['id']} {pr} {s}: data sha256 = MANIFEST.json", g["data_sha256"] == manifest[f"eval_{s}.jsonl"], g["data_sha256"][:16])
            x = jload(os.path.join(d, "eval", f"{pr}_{s}.jsonl"))
            iset = [y["id"] for y in x]
            check("eval", f"{c['id']} {pr} {s}: n = {fz['sets'][s]['n']}, ids unique", len(iset) == fz["sets"][s]["n"] == len(set(iset)), len(iset))
            if ids_by_set[s] is None:
                ids_by_set[s] = set(iset)
            check("eval", f"{c['id']} {pr} {s}: same item ids as every other evaluation", set(iset) == ids_by_set[s])
            k = sum(bool(y["correct"]) for y in x)
            check("eval", f"{c['id']} {pr} {s}: item count = summary.json = DONE.json", k == g["correct"] and abs(done["results"][pr][s] - k / len(x)) < 1e-12, f"{k}/{len(x)}")
            check("eval", f"{c['id']} {pr} {s}: no verifier errors", sum(bool(y["error"]) for y in x) == 0)
check("eval", "one code version for every evaluation, equal to the lock", codes == {lock["code_sha256"]}, codes)
check("eval", "identical evaluation config across checkpoints (apart from run folder, weights and prompts)", len(set(cfgs.values())) == 1)
check("eval", "lock written before the first evaluation started", lock["locked"] < min(started.values()), f"locked {lock['locked']:.0f}, first start {min(started.values()):.0f}")
span = (max(J(os.path.join(FIN, c, 'DONE.json'))["finished"] for c in ids) - min(started.values())) / 3600
notes.append(f"Evaluation wall-clock span (first start to last finish): {span:.2f} h.")

# held-out ids are disjoint from the development ids and from training ids recorded in data.json
dev_ids = {}
for s in DEVSETS:
    p = find("sft/exact_n512_s0")
    dev_ids[s] = {y["id"] for y in jload(f"{p}/eval/final_step_{s}.jsonl")}
check("isolation", "held-out item ids disjoint from development item ids", not (ids_by_set["gsm8k_test_final"] | ids_by_set["math_val_final"]) & (dev_ids["gsm8k_dev"] | dev_ids["math_dev"]))
train_ids = set()
for c in fz["checkpoints"]:
    if c["dir"] and c["stage"] == "sft":
        train_ids |= set(J(find(c["dir"]) + "/data.json")["ids"])
check("isolation", "held-out item ids disjoint from SFT training ids", not (ids_by_set["gsm8k_test_final"] | ids_by_set["math_val_final"]) & train_ids, f"{len(train_ids)} SFT ids")
notes.append("Held-out problem text was not compared with training text in this analysis; text-level decontamination is the Stage 0 result "
             "(data/stage0/contamination.json: GSM8K 0 exact / 0 13-gram / 0 fuzzy; MATH val near-duplicates removed from SFT and RL pools).")

# training-side manifests and post-evaluation integrity
for tree, names in ((TREES[0], ("SESSION_A.sha256", "SESSION_B.sha256")), (TREES[-1], ("RECOVERY.sha256",))):
    for n in names:
        p = os.path.join(tree, n)
        if os.path.exists(p):
            r = subprocess.run(["sha256sum", "-c", "--quiet", n], cwd=tree, capture_output=True, text=True)
            check("training manifests", f"{n} re-verified", r.returncode == 0, (r.stdout + r.stderr).strip()[-300:] or f"{sum(1 for _ in open(p))} entries OK")
bf = [os.path.join(t, "BEFORE_FINAL.sha256") for t in [FIN] + TREES if os.path.exists(os.path.join(t, "BEFORE_FINAL.sha256"))]
if bf:
    notes.append(f"BEFORE_FINAL.sha256 found at {bf[0]}; it lists server paths and can only be checked on the server.")
unverifiable.append("Post-evaluation integrity of the trained runs (`sha256sum -c BEFORE_FINAL.sha256` on the server, expected output TRAINED_RUNS_UNCHANGED): "
                    "neither BEFORE_FINAL.sha256 nor the terminal output was recovered to the laptop, so it is NOT verified here." if not bf else
                    "BEFORE_FINAL.sha256 is present but lists server paths; the TRAINED_RUNS_UNCHANGED result itself was not recovered.")
if not any(os.path.exists(os.path.join(t, "final_eval.log")) for t in [FIN] + TREES):
    unverifiable.append("final_eval.log (launcher output) was not recovered; launch order and the absence of retries are inferred from run.json/DONE.json timestamps only.")
# training-artifact audit (read-only script from the preparation phase), re-run on the same trees
if not a.skip_training_audit:
    cmd = [sys.executable, os.path.join(ROOT, "scripts/final_audit.py")] + sum([["--runs", t] for t in TREES], []) + ["--out", O + "/training_audit"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ta = J(O + "/training_audit/audit.json") if os.path.exists(O + "/training_audit/audit.json") else {}
    check("training audit", "scripts/final_audit.py re-run: 0 unresolved problems", r.returncode == 0 and ta.get("problems") == [], f"problems={ta.get('problems')}, explained={len(ta.get('explained', []))}")

# ======================================================================= 2. item-level data and independent reproduction
ans = lambda t: t.split("<answer>")[-1].replace("</answer>", "").strip()
ITEMS = {}
for c in fz["checkpoints"]:
    for pr in c["prompts"]:
        for s in SETS:
            x = jload(os.path.join(FIN, c["id"], "eval", f"{pr}_{s}.jsonl"))
            strict = {y["id"]: bool(y["correct"]) for y in x}
            sec = {y["id"]: strict[y["id"]] or bool(pr == "r1_zero" and y["format"] and A.second_verifier(ans(y["response"]), str(y["answer"]))) for y in x}
            rel = {y["id"]: sec[y["id"]] or y["category"] == "format_fail_answer_right" for y in x}
            ITEMS[(c["id"], pr, s)] = dict(strict=strict, second=sec, relaxed=rel, cat={y["id"]: y["category"] for y in x}, fmt={y["id"]: bool(y["format"]) if pr == "r1_zero" else True for y in x},
                                           clean={y["id"]: bool(y.get("clean_format")) for y in x}, tok={y["id"]: y["n_tokens"] for y in x}, trunc={y["id"]: y["finish"] == "length" for y in x},
                                           dbl={y["id"]: y["response"].count("</think>") >= 2 for y in x})


def wilson_sp(k, n):
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return float(ci.low), float(ci.high)


def paired_sp(a_, b_):
    """Independent re-implementation: numpy arrays, scipy exact binomial test (two-sided, p = 0.5) for McNemar."""
    keys = sorted(a_)
    x, y = np.array([a_[k] for k in keys]), np.array([b_[k] for k in keys])
    n, ao, bo = len(keys), int((x & ~y).sum()), int((y & ~x).sum())
    d = (bo - ao) / n
    var = ((ao + bo) / n - d * d) / n  # variance of the paired difference in proportions (Wald)
    p = 1.0 if ao + bo == 0 else float(binomtest(min(ao, bo), ao + bo, 0.5).pvalue)
    return dict(n=n, a_only=ao, b_only=bo, both=int((x & y).sum()), neither=int((~x & ~y).sum()), diff=d, se=float(np.sqrt(max(var, 0.0))),
                ci95=(d - 1.96 * np.sqrt(max(var, 0.0)), d + 1.96 * np.sqrt(max(var, 0.0))), p=p)


def holm_np(names, ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps, kind="stable")
    adj = np.minimum(1.0, np.maximum.accumulate((len(ps) - np.arange(len(ps))) * ps[o]))
    out = np.empty_like(adj)
    out[o] = adj
    return dict(zip(names, out.tolist()))


dev_rep = dict(metric_fields=0, comparison_fields=0, max_abs_dev=0.0, mismatches=[])


def cmpv(where, got, want, tol=1e-9):
    dev_rep["max_abs_dev"] = max(dev_rep["max_abs_dev"], abs(got - want))
    if abs(got - want) > tol * max(1.0, abs(want)):
        dev_rep["mismatches"].append(f"{where}: reproduced {got!r} vs report {want!r}")


repm = {(m["id"], m["prompt"], m["set"]): m for m in REP["metrics"]}
rows = []
for (cid, pr, s), it in ITEMS.items():
    n, k = len(it["strict"]), sum(it["strict"].values())
    lo, hi = wilson_sp(k, n)
    inc = [i for i in it["strict"] if not it["strict"][i]]
    cats = Counter(it["cat"].values())
    r = dict(checkpoint=cid, label=SHORT[cid], cls=next(c["class"] for c in fz["checkpoints"] if c["id"] == cid), stage=next(c["stage"] for c in fz["checkpoints"] if c["id"] == cid),
             data_source=META[cid][0], sft_examples=META[cid][1], prompt=pr, set=s, n=n, correct=k, strict=k / n, strict_ci_lo=lo, strict_ci_hi=hi,
             second=sum(it["second"].values()) / n, relaxed=sum(it["relaxed"].values()) / n, format=sum(it["fmt"].values()) / n,
             clean_format=(sum(it["clean"].values()) / n) if pr == "r1_zero" else None, truncated_rate=sum(it["trunc"].values()) / n,
             truncated_share_of_errors=(sum(it["trunc"][i] for i in inc) / len(inc)) if inc else None, double_think=sum(it["dbl"].values()),
             tokens_mean=st.mean(it["tok"].values()), tokens_correct=st.mean(it["tok"][i] for i in it["strict"] if it["strict"][i]),
             tokens_incorrect=st.mean(it["tok"][i] for i in inc) if inc else None,
             acc_given_format=(sum(it["strict"][i] for i in it["fmt"] if it["fmt"][i]) / max(1, sum(it["fmt"].values()))),
             **{f"cat_{kk}": v for kk, v in sorted(cats.items())})
    rows.append(r)
    m = repm[(cid, pr, s)]
    for f_, g_ in (("strict", r["strict"]), ("second", r["second"]), ("relaxed", r["relaxed"]), ("format", r["format"]), ("tokens_mean", r["tokens_mean"]),
                   ("truncated_rate", r["truncated_rate"])):
        cmpv(f"{cid} {pr} {s} {f_}", g_, m[f_])
        dev_rep["metric_fields"] += 1
    cmpv(f"{cid} {pr} {s} ci_lo", lo, m["ci95"][0], 1e-5); cmpv(f"{cid} {pr} {s} ci_hi", hi, m["ci95"][1], 1e-5); dev_rep["metric_fields"] += 2  # scipy uses z = 1.959964, the report 1.96
    if pr == "r1_zero":
        cmpv(f"{cid} {pr} {s} clean_format", r["clean_format"], m["clean_format"]); dev_rep["metric_fields"] += 1
    cmpv(f"{cid} {pr} {s} double_think", r["double_think"], m["double_think"]); dev_rep["metric_fields"] += 1

# training compute (GPU-hours = 2 GPUs x wall-clock, the convention of the report; includes in-training dev evaluations)
def train_rows(d):
    keep = {}
    for x in jload(d + "/metrics.jsonl"):
        if x["event"] == "train":
            keep[x["step"]] = x
    return [keep[k] for k in sorted(keep)]


GH = {}
for c in fz["checkpoints"]:
    if c["dir"]:
        GH[c["id"]] = 2 * J(find(c["dir"]) + "/DONE.json")["wall_s"] / 3600
tA = train_rows(gA)
A_upto80 = 2 * sum(x["step_s"] for x in tA if x["step"] <= 80) / 3600
A_total = 2 * max(x.get("wall_s", 0) for x in jload(gA + "/metrics.jsonl")) / 3600
CUM = {cid: v for cid, v in GH.items()}
CUM["GRPO-B_sft512_s2026"] = GH["GRPO-B_sft512_s2026"] + GH["SFT-reverified_n512_s0"]
for b in ("GRPO-A-replay80-lr1e-5-s2026", "GRPO-A-recovery80-lr5e-6-s2026"):
    CUM[b] = A_upto80 + GH[b]
for r in rows:
    r["train_gpu_h"] = GH.get(r["checkpoint"])
    r["cumulative_gpu_h"] = CUM.get(r["checkpoint"], 0.0)

# dev results of the same checkpoints (final step) for the consolidated table and the dev-vs-final analysis
DEVDIR = {"SFT-exact_full_s0": ("sft/exact_full_s0", "final_step"), "SFT-reverified_full_s0": ("sft/reverified_full_s0", "final_step"),
          "SFT-reverified_sizecontrol_s0": ("sft/reverified_sizecontrol_s0", "final_step"), "SFT-exact_n512_s0": ("sft/exact_n512_s0", "final_step"),
          "SFT-reverified_n512_s0": ("sft/reverified_n512_s0", "final_step"), "SFT-reverified_n128_s0": ("sft/reverified_n128_s0", "final_step"),
          "SFT-reverified_n2048_s0": ("sft/reverified_n2048_s0", "final_step"), "GRPO-B_sft512_s2026": ("grpo/B_sft512_s2026", "final_step"),
          "GRPO-A-replay80-lr1e-5-s2026": ("recovery/A_replay80_lr1e-5_s2026", "final_step"), "GRPO-A-recovery80-lr5e-6-s2026": ("recovery/A_recovery80_lr5e-6_s2026", "final_step"),
          "BASE": ("grpo/A_base_s2026", "step0")}  # base, r1_zero: GRPO-A's step-0 evaluation (no update yet; same engine settings as the dev evaluations)
DEV = {}
for cid, (rel, tag) in DEVDIR.items():
    for s in DEVSETS:
        x = jload(f"{find(rel)}/eval/{tag}_{s}.jsonl")
        DEV[(cid, s)] = {y["id"]: bool(y["correct"]) for y in x}
for r in rows:
    if r["prompt"] == "r1_zero":
        s = "gsm8k_dev" if r["set"].startswith("gsm8k") else "math_dev"
        dv = DEV.get((r["checkpoint"], s))
        if dv:
            r["dev_set"], r["dev_strict"] = s, sum(dv.values()) / len(dv)
            r["dev_minus_final_pp"] = 100 * (r["dev_strict"] - r["strict"])
wcsv(O + "/consolidated_results.csv", rows)

# comparisons: frozen families (reproduced), then post-hoc unregistered and sensitivity
LAB = dict(confirmatory="confirmatory", sft_vs_base="confirmatory-secondary", descriptive="descriptive", posthoc="post-hoc (frozen)")
key = lambda x, s: tuple(x.split(":")) + (s,)
HYP = {  # plain reading of each detected direction; the frozen plan's tests are two-sided
    "C1 data quality, full": ("re-verified worse than exact-match", "re-verified better than exact-match"),
    "C2 data quality, matched 512": ("re-verified worse than exact-match", "re-verified better than exact-match"),
    "C3 pathway B vs its SFT parent": ("GRPO worse than its SFT parent", "GRPO better than its SFT parent"),
}


def direction(name, diff):
    if name in HYP:
        return HYP[name][diff > 0]
    if " vs base question-only" in name and name.startswith("SFT"):
        return "SFT better than base question-only" if diff > 0 else "SFT WORSE than base question-only"
    return "b > a" if diff > 0 else "b < a"


comps = []
repc = {(g["family"], g["name"], g["set"]): g for g in REP["comparisons"]}
for fam, f in fz["families"].items():
    for s in SETS:
        grp = []
        for cm in f["comparisons"]:
            row = dict(family=LAB[fam], frozen_family=fam, name=cm["name"], set=s, a=cm["a"], b=cm["b"], metric="strict")
            if cm.get("unavailable"):
                row.update(unavailable=cm["unavailable"], p=1.0)
            else:
                r = paired_sp(ITEMS[key(cm["a"], s)]["strict"], ITEMS[key(cm["b"], s)]["strict"])
                row.update(r)
                row["relaxed_diff"] = paired_sp(ITEMS[key(cm["a"], s)]["relaxed"], ITEMS[key(cm["b"], s)]["relaxed"])["diff"]
                row["second_diff"] = paired_sp(ITEMS[key(cm["a"], s)]["second"], ITEMS[key(cm["b"], s)]["second"])["diff"]
            grp.append(row)
        if f["holm_m"]:
            assert len(grp) == f["holm_m"]
            h = holm_np([g["name"] for g in grp], [g["p"] for g in grp])
            for g in grp:
                g["p_holm"] = h[g["name"]]
        for g in grp:
            rg = repc[(fam, g["name"], s)]
            if g.get("diff") is not None:
                for f_ in ("a_only", "b_only", "diff", "p", "relaxed_diff", "second_diff"):
                    cmpv(f"{fam} {g['name']} {s} {f_}", g[f_], rg[f_]); dev_rep["comparison_fields"] += 1
                cmpv(f"{fam} {g['name']} {s} ci_lo", g["ci95"][0], rg["ci95"][0]); cmpv(f"{fam} {g['name']} {s} ci_hi", g["ci95"][1], rg["ci95"][1])
                dev_rep["comparison_fields"] += 2
            if "p_holm" in g:
                cmpv(f"{fam} {g['name']} {s} p_holm", g["p_holm"], rg["p_holm"]); dev_rep["comparison_fields"] += 1
            g["report_verdict"] = rg["verdict"]
            if g.get("diff") is None:
                g["reading"] = "unavailable (pre-registered run failed); counted p = 1"
            else:
                adj = g.get("p_holm", g["p"])
                big = s != "gsm8k_test_final" or abs(100 * g["diff"]) >= fz["thresholds"]["detectable_pp"]
                sig = adj < fz["thresholds"]["alpha"] and big
                g["reading"] = (f"difference detected: {direction(g['name'], g['diff'])}" if sig else "no difference detected (not evidence of equivalence)")
        comps += grp

EXTRA = [  # chosen after the held-out results were seen: post-hoc, unregistered, own Holm family per set
    ("Direct (A1, post-hoc branch) vs sequential (GRPO-B)", "GRPO-A-replay80-lr1e-5-s2026:r1_zero", "GRPO-B_sft512_s2026:r1_zero"),
    ("Direct (A2, post-hoc branch) vs sequential (GRPO-B)", "GRPO-A-recovery80-lr5e-6-s2026:r1_zero", "GRPO-B_sft512_s2026:r1_zero"),
    ("GRPO-B vs SFT exact n512 (other data source, same size)", "SFT-exact_n512_s0:r1_zero", "GRPO-B_sft512_s2026:r1_zero"),
    ("Exact-match scaling 512 -> 3,420", "SFT-exact_n512_s0:r1_zero", "SFT-exact_full_s0:r1_zero"),
    ("Composition at 512 vs at full size: exact n512 -> re-verified n2048", "SFT-exact_n512_s0:r1_zero", "SFT-reverified_n2048_s0:r1_zero"),
]
for s in SETS:
    grp = []
    for name, x, y in EXTRA:
        r = paired_sp(ITEMS[key(x, s)]["strict"], ITEMS[key(y, s)]["strict"])
        grp.append(dict(family="post-hoc (unregistered)", name=name, set=s, a=x, b=y, metric="strict", **r,
                        relaxed_diff=paired_sp(ITEMS[key(x, s)]["relaxed"], ITEMS[key(y, s)]["relaxed"])["diff"]))
    h = holm_np([g["name"] for g in grp], [g["p"] for g in grp])
    for g in grp:
        g["p_holm"] = h[g["name"]]
        g["reading"] = ("exploratory: difference (Holm p < 0.05)" if g["p_holm"] < 0.05 else "exploratory: no difference detected") + f", {direction(g['name'], g['diff'])}"
    comps += grp
# sensitivity: the confirmatory family on relaxed accuracy (C4 still p = 1)
for s in SETS:
    grp = []
    for cm in fz["families"]["confirmatory"]["comparisons"]:
        row = dict(family="sensitivity", name=cm["name"] + " [relaxed accuracy]", set=s, a=cm["a"], b=cm["b"], metric="relaxed")
        if cm.get("unavailable"):
            row.update(p=1.0, unavailable=cm["unavailable"])
        else:
            row.update(paired_sp(ITEMS[key(cm["a"], s)]["relaxed"], ITEMS[key(cm["b"], s)]["relaxed"]))
        grp.append(row)
    h = holm_np([g["name"] for g in grp], [g["p"] for g in grp])
    for g in grp:
        g["p_holm"] = h[g["name"]]
        if g.get("diff") is not None:
            g["reading"] = "sensitivity: " + ("difference" if g["p_holm"] < 0.05 and (s != "gsm8k_test_final" or abs(100 * g["diff"]) >= 1.4) else "no difference detected")
    comps += grp

# dev-set value of each held-out comparison whose two sides have dev evaluations (BASE question-only has none)
for g in comps:
    xa, xb = g["a"].split(":"), g["b"].split(":")
    if xa[1] == "r1_zero" and xb[1] == "r1_zero" and g.get("metric") == "strict":
        ds = "gsm8k_dev" if g["set"].startswith("gsm8k") else "math_dev"
        if (xa[0], ds) in DEV and (xb[0], ds) in DEV:
            r = paired_sp(DEV[(xa[0], ds)], DEV[(xb[0], ds)])
            g["dev_diff"], g["dev_p"] = r["diff"], r["p"]

for g in comps:
    if "ci95" in g:
        g["ci_lo"], g["ci_hi"] = g["ci95"]
    for f_ in ("diff", "ci_lo", "ci_hi", "relaxed_diff", "second_diff", "dev_diff", "se"):
        if g.get(f_) is not None:
            g[f_ + "_pp"] = 100 * g[f_]
CC = ["family", "name", "set", "metric", "a", "b", "n", "a_only", "b_only", "both", "neither", "diff_pp", "ci_lo_pp", "ci_hi_pp", "se_pp", "p", "p_holm",
      "second_diff_pp", "relaxed_diff_pp", "dev_diff_pp", "dev_p", "reading", "report_verdict", "unavailable"]
wcsv(O + "/statistical_comparisons.csv", comps, CC)
check("reproduction", "every metric and comparison in report.json reproduced by an independent implementation (scipy binomtest, numpy Holm)",
      not dev_rep["mismatches"], f"{dev_rep['metric_fields']} metric and {dev_rep['comparison_fields']} comparison fields; max |diff| {dev_rep['max_abs_dev']:.2e}; "
      f"mismatches: {dev_rep['mismatches'][:5]}")

# ======================================================================= 3. analysis tables
AN = dict(gpu_h=GH, cumulative_gpu_h=CUM, failed_A=dict(gpu_h_total=A_total, gpu_h_steps_1_80=A_upto80, incident=INC))
# 3a. seed sensitivity: the only same-data, same-lr, different-training-seed pair is SFT-reverified_n512_s0 (seed 0) vs SFT-tune_lr1e-5 (seed 100), dev sets only
seedpair = {}
tune = find("sft/tune_lr1e-5")
for s in DEVSETS:
    t = {y["id"]: bool(y["correct"]) for y in jload(f"{tune}/eval/final_step_{s}.jsonl")}
    r = paired_sp(DEV[("SFT-reverified_n512_s0", s)], t)
    sig2 = max(0.0, r["diff"] ** 2 - r["se"] ** 2)  # seed variance of a two-arm contrast, method of moments, 1 degree of freedom
    seedpair[s] = dict(diff=r["diff"], se_item=r["se"], p=r["p"], sigma_contrast=sig2 ** 0.5)
same_data = J(find("sft/tune_lr1e-5") + "/data.json")["ids_sha256"] == J(find("sft/reverified_n512_s0") + "/data.json")["ids_sha256"]
AN["seed_pair"] = dict(note="SFT-reverified_n512_s0 (seed 0) vs SFT-tune_lr1e-5 (seed 100): same data ids and lr; development sets only; one pair = 1 degree of freedom",
                       same_training_ids=same_data, **seedpair)
seed_rows = []
for g in comps:
    if g.get("diff") is None or g["family"] not in ("confirmatory", "descriptive", "post-hoc (unregistered)") or g.get("metric") != "strict":
        continue
    sg = seedpair["gsm8k_dev" if g["set"].startswith("gsm8k") else "math_dev"]["sigma_contrast"]
    two_trained = "BASE" not in g["a"] and "BASE" not in g["b"]
    parent_child = g["name"].startswith("C3")
    tot = (g["se"] ** 2 + (sg ** 2 if two_trained and not parent_child else 0.0)) ** 0.5
    seed_rows.append(dict(family=g["family"], name=g["name"], set=g["set"], diff_pp=100 * g["diff"], item_se_pp=100 * g["se"],
                          seed_sd_contrast_pp=100 * sg if two_trained and not parent_child else None, total_se_pp=100 * tot,
                          approx_ci_lo_pp=100 * (g["diff"] - 1.96 * tot), approx_ci_hi_pp=100 * (g["diff"] + 1.96 * tot), ratio_diff_to_total_se=g["diff"] / tot,
                          applies=("seed SD from the dev pair applied" if two_trained and not parent_child else
                                   "not applied: GRPO-seed variance unknown (no GRPO seed replicate)" if parent_child else "not applied")))
wcsv(O + "/data/seed_sensitivity.csv", seed_rows)
AN["seed_sensitivity"] = seed_rows

# 3b. cross-benchmark and dev-to-final agreement
tr = [c["id"] for c in fz["checkpoints"] if c["id"] != "BASE"]
acc = {(r["checkpoint"], r["set"]): r["strict"] for r in rows if r["prompt"] == "r1_zero"}
g_ = [acc[(c, "gsm8k_test_final")] for c in tr]
m_ = [acc[(c, "math_val_final")] for c in tr]
xb = dict(n_checkpoints=len(tr), spearman_gsm8k_math=float(spearmanr(g_, m_)[0]), spearman_p=float(spearmanr(g_, m_)[1]), kendall_gsm8k_math=float(kendalltau(g_, m_)[0]))
for s, ds in (("gsm8k_test_final", "gsm8k_dev"), ("math_val_final", "math_dev")):
    dv = [sum(DEV[(c, ds)].values()) / len(DEV[(c, ds)]) for c in tr]
    fv = [acc[(c, s)] for c in tr]
    xb[f"spearman_dev_final_{s}"] = float(spearmanr(dv, fv)[0])
    xb[f"kendall_dev_final_{s}"] = float(kendalltau(dv, fv)[0])
    xb[f"mean_dev_minus_final_pp_{s}"] = 100 * float(np.mean(np.array(dv) - np.array(fv)))
    bdev = sum(DEV[("BASE", ds)].values()) / len(DEV[("BASE", ds)])
    xb[f"base_r1_zero_dev_minus_final_pp_{s}"] = 100 * (bdev - acc[("BASE", s)])
agree = []
for name in {g["name"] for g in comps if g.get("diff") is not None and g.get("metric") == "strict"}:
    gs = [g for g in comps if g["name"] == name and g.get("metric") == "strict" and g.get("diff") is not None]
    if len(gs) == 2:
        dg = next(g for g in gs if g["set"].startswith("gsm8k"))["diff"]
        dm = next(g for g in gs if g["set"].startswith("math"))["diff"]
        agree.append(dict(name=name, family=gs[0]["family"], gsm8k_pp=100 * dg, math_pp=100 * dm, same_sign=(dg > 0) == (dm > 0)))
xb["comparisons_same_sign"] = f"{sum(x['same_sign'] for x in agree)}/{len(agree)}"
devfin = [dict(name=g["name"], family=g["family"], set=g["set"], final_pp=100 * g["diff"], dev_pp=100 * g["dev_diff"], same_sign=(g["diff"] > 0) == (g["dev_diff"] > 0))
          for g in comps if g.get("dev_diff") is not None]
xb["dev_final_same_sign"] = f"{sum(x['same_sign'] for x in devfin)}/{len(devfin)}"
wcsv(O + "/data/cross_benchmark_signs.csv", agree)
wcsv(O + "/data/dev_vs_final_comparisons.csv", devfin)
wcsv(O + "/data/dev_vs_final_accuracy.csv", [dict(checkpoint=c, label=SHORT[c], set=s, dev=sum(DEV[(c, ds)].values()) / len(DEV[(c, ds)]), final=acc[(c, s)])
                                              for c in ["BASE"] + tr for s, ds in (("gsm8k_test_final", "gsm8k_dev"), ("math_val_final", "math_dev"))])
AN["cross_benchmark"] = xb

# 3c. where gains come from: category transitions on discordant items
TRANS = [("C3: SFT parent -> GRPO-B", "SFT-reverified_n512_s0:r1_zero", "GRPO-B_sft512_s2026:r1_zero"),
         ("Base r1_zero -> base question-only", "BASE:r1_zero", "BASE:question_only"),
         ("Base question-only -> SFT re-verified n512", "BASE:question_only", "SFT-reverified_n512_s0:r1_zero"),
         ("Base r1_zero -> SFT re-verified n512", "BASE:r1_zero", "SFT-reverified_n512_s0:r1_zero"),
         ("SFT exact n512 -> SFT re-verified n512 (C2)", "SFT-exact_n512_s0:r1_zero", "SFT-reverified_n512_s0:r1_zero"),
         ("A1 -> GRPO-B", "GRPO-A-replay80-lr1e-5-s2026:r1_zero", "GRPO-B_sft512_s2026:r1_zero")]
trans = []
for lab, x, y in TRANS:
    for s in SETS:
        ia, ib = ITEMS[key(x, s)], ITEMS[key(y, s)]
        gained = Counter(ia["cat"][i] for i in ia["strict"] if ib["strict"][i] and not ia["strict"][i])
        lost = Counter(ib["cat"][i] for i in ia["strict"] if ia["strict"][i] and not ib["strict"][i])
        for kind, cn in (("gained: category before", gained), ("lost: category after", lost)):
            for cat, v in sorted(cn.items()):
                trans.append(dict(transition=lab, set=s, kind=kind, category=cat, count=v))
wcsv(O + "/data/gain_transitions.csv", trans)

# 3d. item difficulty: how many of the nine pre-registered trained checkpoints solve each item
pre = [c["id"] for c in fz["checkpoints"] if c["class"] == "preregistered" and c["id"] != "BASE"]
diff_rows = []
for s in SETS:
    cnt = Counter(sum(ITEMS[(c, "r1_zero", s)]["strict"][i] for c in pre) for i in ids_by_set[s])
    qo = ITEMS[("BASE", "question_only", s)]["strict"]
    for k_ in range(len(pre) + 1):
        its = [i for i in ids_by_set[s] if sum(ITEMS[(c, "r1_zero", s)]["strict"][i] for c in pre) == k_]
        diff_rows.append(dict(set=s, models_solving=k_, items=len(its), base_question_only_solves=sum(qo[i] for i in its)))
wcsv(O + "/data/item_difficulty.csv", diff_rows)
AN["never_solved"] = {s: next(r["items"] for r in diff_rows if r["set"] == s and r["models_solving"] == 0) for s in SETS}
AN["solved_by_base_qo_not_by_B"] = {s: sum(ITEMS[("BASE", "question_only", s)]["strict"][i] and not ITEMS[("GRPO-B_sft512_s2026", "r1_zero", s)]["strict"][i] for i in ids_by_set[s]) for s in SETS}

# 3e. A2's doubled </think>: accuracy with and without it
dbl = {}
for s in SETS:
    it = ITEMS[("GRPO-A-recovery80-lr5e-6-s2026", "r1_zero", s)]
    w = [i for i in it["dbl"] if it["dbl"][i]]
    wo = [i for i in it["dbl"] if not it["dbl"][i]]
    dbl[s] = dict(doubled=len(w), acc_doubled=sum(it["strict"][i] for i in w) / max(1, len(w)), not_doubled=len(wo), acc_not_doubled=sum(it["strict"][i] for i in wo) / max(1, len(wo)),
                  format_pass_with_doubled=sum(it["fmt"][i] for i in w), clean_with_doubled=sum(it["clean"][i] for i in w))
AN["A2_doubled_think"] = dbl

# 3f. GRPO curves (dev), per-step dynamics, SFT loss, compute windows
CURVES = [("GRPO-B (sequential)", "grpo/B_sft512_s2026", ["step0", "step25", "step50", "step75", "final_step"], [0, 25, 50, 75, 100]),
          ("GRPO-A (direct, FAILED at 98)", "grpo/A_base_s2026", ["step0", "step25", "step50", "step75"], [0, 25, 50, 75]),
          ("A1 (post-hoc, lr 1e-5)", "recovery/A_replay80_lr1e-5_s2026", ["fork_step80", "final_step"], [80, 100]),
          ("A2 (post-hoc, lr 5e-6)", "recovery/A_recovery80_lr5e-6_s2026", ["fork_step80", "final_step"], [80, 100])]
tB = train_rows(find("grpo/B_sft512_s2026"))
cumB = lambda k: GH["SFT-reverified_n512_s0"] + 2 * sum(x["step_s"] for x in tB if x["step"] <= k) / 3600
cumA = lambda k: 2 * sum(x["step_s"] for x in tA if x["step"] <= k) / 3600
curve_rows = []
for lab, rel, tags, steps in CURVES:
    d = find(rel)
    tb = train_rows(d)
    for tag, k_ in zip(tags, steps):
        for s in DEVSETS:
            x = jload(f"{d}/eval/{tag}_{s}.jsonl")
            kk = sum(bool(y["correct"]) for y in x)
            lo, hi = wilson_sp(kk, len(x))
            if lab.startswith("GRPO-B"):
                gh = cumB(k_)
            elif lab.startswith("GRPO-A"):
                gh = cumA(k_)
            else:
                gh = A_upto80 + 2 * sum(x_["step_s"] for x_ in tb if x_["step"] <= k_) / 3600
            curve_rows.append(dict(run=lab, step=k_, set=s, strict=kk / len(x), ci_lo=lo, ci_hi=hi, format=st.mean(bool(y["format"]) for y in x),
                                   tokens=st.mean(y["n_tokens"] for y in x), cumulative_gpu_h=gh))
for lab, rel in (("GRPO tune lr 2e-6", "grpo/tune_lr2e-6"), ("GRPO tune lr 5e-6", "grpo/tune_lr5e-6"), ("GRPO tune lr 1e-5", "grpo/tune_lr1e-5")):
    d = find(rel)
    n_ = len(train_rows(d))
    for s in DEVSETS:
        x = jload(f"{d}/eval/final_step_{s}.jsonl")
        kk = sum(bool(y["correct"]) for y in x)
        lo, hi = wilson_sp(kk, len(x))
        curve_rows.append(dict(run=lab, step=n_, set=s, strict=kk / len(x), ci_lo=lo, ci_hi=hi, format=st.mean(bool(y["format"]) for y in x), tokens=st.mean(y["n_tokens"] for y in x),
                               cumulative_gpu_h=2 * J(d + "/DONE.json")["wall_s"] / 3600))
wcsv(O + "/data/grpo_dev_curves.csv", curve_rows)

DYN = [("GRPO-B (sequential)", "grpo/B_sft512_s2026"), ("GRPO-A (direct, FAILED at 98)", "grpo/A_base_s2026"), ("A1 (post-hoc, lr 1e-5)", "recovery/A_replay80_lr1e-5_s2026"),
       ("A2 (post-hoc, lr 5e-6)", "recovery/A_recovery80_lr5e-6_s2026"), ("GRPO tune lr 1e-5", "grpo/tune_lr1e-5"), ("GRPO tune lr 5e-6", "grpo/tune_lr5e-6"),
       ("GRPO tune lr 2e-6", "grpo/tune_lr2e-6")]
FIELDS = ["reward_mean", "entropy", "len_mean", "len_correct", "len_incorrect", "grad_norm", "rollout_logprob_mad", "rollout_logprob_p99", "rollout_logprob_max",
          "truncated_rate", "format_rate", "step_s", "rollout_s", "update_s", "params_mismatched"]
dyn_rows = []
for lab, rel in DYN:
    for x in train_rows(find(rel)):
        dyn_rows.append(dict(run=lab, step=x["step"], zero_adv_frac=x["zero_adv"] / x["groups"], all_correct_frac=x["all_correct"] / x["groups"],
                             all_incorrect_frac=x["all_incorrect"] / x["groups"], groups=x["groups"], **{f_: x.get(f_) for f_ in FIELDS}))
dyn_rows.append(dict(run="GRPO-A (direct, FAILED at 98)", step=INC["attempted_step"], rollout_logprob_mad=INC["gate"]["value"], rejected_by_gate=True))
wcsv(O + "/data/grpo_dynamics.csv", dyn_rows)

sft_rows = []
for c in fz["checkpoints"]:
    if c["stage"] == "sft":
        for x in train_rows(find(c["dir"])):
            sft_rows.append(dict(run=SHORT[c["id"]], checkpoint=c["id"], step=x["step"], loss=x["loss"], grad_norm=x.get("grad_norm")))
wcsv(O + "/data/sft_loss.csv", sft_rows)

win_rows = []
for lab, t, cum, evs in (("GRPO-B (sequential)", tB, cumB, [0, 25, 50, 75, 100]), ("GRPO-A (direct, FAILED at 98)", tA, cumA, [0, 25, 50, 75])):
    for lo, hi in zip(evs[:-1], evs[1:]):
        seg = [x for x in t if lo < x["step"] <= hi]
        row = dict(run=lab, window=f"{lo}-{hi}", gpu_h=cum(hi) - cum(lo), zero_adv_frac=sum(x["zero_adv"] for x in seg) / sum(x["groups"] for x in seg),
                   entropy=st.mean(x["entropy"] for x in seg), reward=st.mean(x["reward_mean"] for x in seg), len_mean=st.mean(x["len_mean"] for x in seg))
        for s in DEVSETS:
            a_ = next(r for r in curve_rows if r["run"] == lab and r["step"] == lo and r["set"] == s)
            b_ = next(r for r in curve_rows if r["run"] == lab and r["step"] == hi and r["set"] == s)
            row[f"{s}_gain_pp"] = 100 * (b_["strict"] - a_["strict"])
            row[f"{s}_pp_per_gpu_h"] = row[f"{s}_gain_pp"] / row["gpu_h"]
        win_rows.append(row)
wcsv(O + "/data/grpo_compute_windows.csv", win_rows)

# 3g. efficiency on the held-out sets: gain over base (question-only, the reasoning baseline; r1_zero, the protocol baseline) per cumulative GPU-hour
eff = []
for r in rows:
    if r["prompt"] != "r1_zero" or r["checkpoint"] == "BASE":
        continue
    b_qo = acc_qo = next(x["strict"] for x in rows if x["checkpoint"] == "BASE" and x["prompt"] == "question_only" and x["set"] == r["set"])
    b_r1 = next(x["strict"] for x in rows if x["checkpoint"] == "BASE" and x["prompt"] == "r1_zero" and x["set"] == r["set"])
    eff.append(dict(checkpoint=r["checkpoint"], label=r["label"], cls=r["cls"], set=r["set"], strict=r["strict"], cumulative_gpu_h=r["cumulative_gpu_h"], own_gpu_h=r["train_gpu_h"],
                    gain_vs_base_qo_pp=100 * (r["strict"] - b_qo), gain_vs_base_r1_pp=100 * (r["strict"] - b_r1),
                    pp_per_gpu_h_vs_qo=100 * (r["strict"] - b_qo) / r["cumulative_gpu_h"]))
wcsv(O + "/data/efficiency.csv", eff)
AN["efficiency_note"] = ("GPU-hours = 2 x wall-clock of the training run (the report's convention), including in-training dev evaluations. GRPO-B includes its SFT parent; "
                         "A1/A2 include GRPO-A steps 1-80 (step time only). The failed GRPO-A consumed about %.2f GPU-h in total." % A_total)

AN["notes"], AN["unverifiable"] = notes, unverifiable
AN["reproduction"] = dev_rep
json.dump(AN, open(O + "/analysis.json", "w"), indent=1, default=float)
json.dump(dict(checks=checks, problems=problems, unverifiable=unverifiable, notes=notes), open(O + "/audit.json", "w"), indent=1, default=str)

# ======================================================================= 4. AUDIT.md (generated)
L = ["# Final artifact audit (generated by scripts/research_analysis.py)", "",
     f"Checks: {len(checks)}; failed: {len(problems)}; not verifiable from the laptop: {len(unverifiable)}.", "",
     "Read-only. Inputs: `~/a5_final` (held-out outputs), `~/a5_sessions` (Session A/B), `~/a5_recovery` (failed run and post-hoc branches), "
     "`results/final_eval/report.json`, `final/freeze_v1.yaml`, `data/stage0/MANIFEST.json`, `recovery/incident_GRPO-A_base_s2026.json`.", "",
     "## Failed checks", ""] + ([f"- {p}" for p in problems] or ["- none"]) + ["", "## Not verifiable from the laptop", ""] + [f"- {u}" for u in unverifiable] + \
    ["", "## Notes", ""] + [f"- {n}" for n in notes] + ["", "## All checks", "", "| area | check | result | detail |", "|---|---|---|---|"]
for c in checks:
    L.append(f"| {c['area']} | {c['check']} | {'ok' if c['ok'] else 'FAIL'} | {c['detail'][:160].replace('|', '/')} |")
open(O + "/AUDIT_GENERATED.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"checks {len(checks)}, failed {len(problems)}, unverifiable {len(unverifiable)}")
for p in problems:
    print("FAIL", p)
print(json.dumps(dev_rep, default=str)[:600])
