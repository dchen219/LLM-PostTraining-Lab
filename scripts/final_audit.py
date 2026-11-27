"""Audit of every recovered Tier A, failed-original and recovery-branch artifact, plus the consolidated development results. Reads only.

python scripts/final_audit.py --runs ~/a5_sessions [--runs ~/a5_recovery] --out results/final_prep/audit
Several --runs trees are merged (the first one holding a run wins). For each run: folder, DONE/FAILED status against the expected outcome,
run.json config hash, config differences from the registered config, code and verifier hashes, metric rows (duplicates, gaps, count against
DONE.json), resumes and superseded rows, evaluation files against DONE.json, lineage (GRPO-B's parent is the registered SFT parent).
Every *.sha256 manifest at a tree's root is verified file by file. Development sets only: no held-out file is opened.
"""
import argparse, csv, json, logging, os, statistics as st, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
logging.disable(logging.CRITICAL)
from posttrain.common import jload, load_raw, sha_file, sha_obj  # noqa: E402
from posttrain.hw import hw_overrides  # noqa: E402
from posttrain.stats import wilson  # noqa: E402
from tier_ids import A as TIER_A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", action="append", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--no-second", action="store_true", help="skip the math-verify second verifier (faster)")
a = ap.parse_args()
TREES = [os.path.expanduser(r) for r in a.runs]
assert not os.path.exists(a.out), "refusing to overwrite"
J = lambda p: json.load(open(p, encoding="utf-8"))
REG = {r["experiment_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "EXPERIMENT_REGISTRY.csv"), encoding="utf-8"))}
from posttrain.verifier_lock import SHA256 as LOCK  # noqa: E402
BRANCHES = {"GRPO-A-replay80-lr1e-5-s2026": "recovery/A_replay80_lr1e-5_s2026", "GRPO-A-recovery80-lr5e-6-s2026": "recovery/A_recovery80_lr5e-6_s2026"}
EXPECT = {i: "done" for i in TIER_A}
EXPECT["GRPO-A_base_s2026"] = "failed"
EXPECT.update({b: "done" for b in BRANCHES})


def rel(i):
    return BRANCHES.get(i) or f"{'sft' if i.startswith('SFT') else 'grpo'}/{i.split('-', 1)[1]}"


def find(i):
    for t in TREES:
        if os.path.isdir(os.path.join(t, rel(i))):
            return os.path.join(t, rel(i))
    return None


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def config_diffs(i, cfg):
    """Leaves of the recorded config that differ from the registered config file (path templates and run_dir are not compared)."""
    if i not in REG or not REG[i]["config"]:
        return None
    reg = flat(load_raw(os.path.join(ROOT, REG[i]["config"])))
    for o in hw_overrides(REG[i]["module"], "a6000"):
        k, v = o.split("=", 1)
        reg[k] = type(reg.get(k, v))(v) if reg.get(k) is not None else v
    got, d = flat(cfg), []
    for k in sorted(set(reg) | set(got)):
        r, g = reg.get(k, "<absent>"), got.get(k, "<absent>")
        if k == "run_dir" or (isinstance(r, str) and "${" in r):
            continue
        try:
            same = float(r) == float(g)
        except (TypeError, ValueError):
            same = r == g
        if not same:
            d.append(dict(key=k, registered=r, recorded=g))
    return d


KNOWN = {"train.lr": "main runs take the learning rate chosen by scripts/choose_lr.py from the tuning runs; re-derived below from the tuning dev results"}
MAN = json.load(open(os.path.join(ROOT, "data", "stage0", "MANIFEST.json"), encoding="utf-8"))
MAN_FILES = next(v for v in MAN.values() if isinstance(v, dict) and "rl_train_pool.jsonl" in v)
BASE_SHA = "f7aa8d665ab44d3b461bed742fe93cdcedc15d2e31e50dc372c126e6f720857b"  # pinned base weights (scripts/server_env.sh PT_BASE_SHA256)

runs, problems, explained = [], [], []
for i in list(EXPECT):
    d = find(i)
    r = dict(id=i, expected=EXPECT[i], folder=d is not None)
    if not d:
        r["status"] = "MISSING"
        problems.append(f"{i}: not recovered")
        runs.append(r)
        continue
    has = lambda f: os.path.exists(os.path.join(d, f))
    incident = os.path.exists(os.path.join(ROOT, "recovery", f"incident_{i}.json"))  # failed before FAILED.json existed: the incident record is its outcome
    r["status"] = "done" if has("DONE.json") else "failed" if (has("FAILED.json") or incident) else "incomplete"
    r["incident_record"] = incident
    if r["status"] != EXPECT[i]:
        problems.append(f"{i}: status {r['status']}, expected {EXPECT[i]}")
    if has("DONE.json") and has("FAILED.json"):
        problems.append(f"{i}: both DONE.json and FAILED.json")
    run = J(d + "/run.json")
    r["config_sha256_ok"] = sha_obj(run["config"]) == run["config_sha256"]
    r["code_sha256"] = run["code_sha256"][:12]
    v = run.get("verifier") or {}
    r["verifier"] = v.get("version")
    if LOCK:
        r["verifier_hashes_ok"] = all(v.get(k) == LOCK.get(k) for k in ("verify_py", "grader_py") if k in LOCK)
    if not r["config_sha256_ok"]:
        problems.append(f"{i}: run.json config does not hash to its config_sha256")
    if r["verifier"] != "2.0.0":
        problems.append(f"{i}: verifier {r['verifier']}")
    diffs = config_diffs(i, run["config"])
    if diffs is not None:
        r["config_diffs"] = [dict(x, why=KNOWN.get(x["key"], "UNEXPECTED")) for x in diffs]
        for x in r["config_diffs"]:
            if x["why"] == "UNEXPECTED":
                problems.append(f"{i}: config {x['key']} registered {x['registered']} recorded {x['recorded']}")
    rows = jload(d + "/metrics.jsonl") if has("metrics.jsonl") else []
    tr = [x for x in rows if x["event"] == "train"]
    steps = [x["step"] for x in tr]
    r.update(train_rows=len(tr), duplicated_step_rows=len(steps) - len(set(steps)), first_step=min(steps) if steps else None, last_step=max(steps) if steps else None,
             resumes=len(jload(d + "/resumes.jsonl")) if has("resumes.jsonl") else 0, superseded_rows=len(jload(d + "/metrics.superseded.jsonl")) if has("metrics.superseded.jsonl") else 0)
    if steps and sorted(set(steps)) != list(range(min(steps), max(steps) + 1)):
        problems.append(f"{i}: gaps in logged steps")
    if r["duplicated_step_rows"]:
        # Benign only if: a resume is recorded, the duplicated steps form one block that starts right after a save point, and the first
        # replayed step equals the original row (the replay restarted from the saved state). Reports then keep the last row per step.
        dup = sorted({x for x in steps if steps.count(x) > 1})
        se = run["config"]["train"].get("save_every") or 0
        block = dup == list(range(dup[0], dup[-1] + 1))
        pair = lambda s_: [x for x in tr if x["step"] == s_]
        same_first = abs(pair(dup[0])[0]["loss"] - pair(dup[0])[-1]["loss"]) < 1e-9
        maxdiff = max(abs(pair(s_)[0]["loss"] - pair(s_)[-1]["loss"]) for s_ in dup)
        ok = bool(r["resumes"] >= 1 and block and se and (dup[0] - 1) % se == 0 and same_first)
        r["duplicates"] = dict(steps=[dup[0], dup[-1]], resumed_from=dup[0] - 1, first_replayed_identical=same_first, max_loss_diff=maxdiff, benign=ok)
        msg = (f"{i}: {r['duplicated_step_rows']} duplicated step rows = steps {dup[0]}-{dup[-1]} replayed after one resume from the step-{dup[0] - 1} save "
               f"(before the rewind fix); first replayed step identical, later losses differ by at most {maxdiff:.1e} (GPU nondeterminism); reports keep the last row per step, "
               f"which is the trajectory that produced the checkpoint")
        (explained if ok else problems).append(msg)
    if has("DONE.json"):
        done = J(d + "/DONE.json")
        r["steps"] = done.get("steps")
        r["gpu_h"] = 2 * done["wall_s"] / 3600
        if steps and done.get("steps") and max(steps) != done["steps"]:
            problems.append(f"{i}: last logged step {max(steps)} != DONE steps {done['steps']}")
        lin = done.get("lineage") or {}
        r["weights_sha256"] = (lin.get("weights_sha256") or {}).get("model.safetensors", "")[:16]
        r["parent_sha256"] = next(iter(((lin.get("parent") or {}).get("weights_sha256") or {"": ""}).values()))[:16]
        r["lineage_config_ok"] = lin.get("config_sha256") in (None, run["config_sha256"])
        if not r["lineage_config_ok"]:
            problems.append(f"{i}: DONE lineage config_sha256 differs from run.json")
        # checkpoint metadata (the weights themselves are not downloaded: their hashes can only be confirmed on the server)
        cl = os.path.join(d, "checkpoint", "lineage.json")
        r["checkpoint_lineage_ok"] = os.path.exists(cl) and J(cl).get("weights_sha256") == lin.get("weights_sha256") and J(cl).get("config_sha256") == lin.get("config_sha256")
        if not r["checkpoint_lineage_ok"]:
            problems.append(f"{i}: checkpoint/lineage.json missing or differs from DONE.json")
        r["weights_full"] = (lin.get("weights_sha256") or {}).get("model.safetensors")
        r["parent_full"] = next(iter(((lin.get("parent") or {}).get("weights_sha256") or {"": ""}).values()))
        r["chain"] = ">".join(lin.get("chain") or [])
        # dataset identity against data/stage0 (MANIFEST.json and the local file)
        dj = J(d + "/data.json") if has("data.json") else {}
        dp = os.path.basename(run["config"]["data"]["path"])
        r["data_file_ok"] = dj.get("file_sha256") == MAN_FILES.get(dp) == sha_file(os.path.join(ROOT, "data", "stage0", dp))
        if i.startswith("SFT"):
            sub = J(os.path.join(ROOT, "data", "stage0", "subsets", os.path.basename(run["config"]["data"]["subset"])))
            r["data_ids_ok"] = set(dj.get("ids", [])) == set(sub["ids"]) and len(dj["ids"]) == dj["n"] == sub["n"] and sha_obj(dj["ids"]) == dj["ids_sha256"]
            r["data"] = f"{sub['name']} n={dj['n']}"
        else:
            r["data_ids_ok"] = dj.get("n_pool") == MAN["counts"]["rl_pool"]
            r["data"] = f"rl_train_pool n={dj.get('n_pool')}"
            if (dj.get("parent") or {}).get("weights_sha256") != (lin.get("parent") or {}).get("weights_sha256"):
                problems.append(f"{i}: data.json parent differs from lineage parent")
        if not (r["data_file_ok"] and r["data_ids_ok"]):
            problems.append(f"{i}: dataset identity does not match data/stage0 (file {r['data_file_ok']}, ids {r['data_ids_ok']})")
        for name, e in (done.get("eval") or {}).items():
            p = f"{d}/{e['file']}" if "file" in e else f"{d}/eval/final_step_{name}.jsonl"
            if not os.path.exists(p):
                problems.append(f"{i}: missing {p}")
                continue
            x = jload(p)
            if len(x) != e["n"] or sum(bool(y["correct"]) for y in x) != e["correct"]:
                problems.append(f"{i}: {name} items disagree with DONE.json")
    runs.append(r)

# lineage: pathway B starts from the registered SFT parent; every other trained run starts from the pinned base weights
byid = {r["id"]: r for r in runs}
if byid.get("GRPO-B_sft512_s2026", {}).get("parent_full") and byid.get("SFT-reverified_n512_s0", {}).get("weights_full"):
    ok = byid["GRPO-B_sft512_s2026"]["parent_full"] == byid["SFT-reverified_n512_s0"]["weights_full"] and byid["GRPO-B_sft512_s2026"]["chain"] == "base>sft>grpo"
    byid["GRPO-B_sft512_s2026"]["parent_is_registered_sft"] = ok
    if not ok:
        problems.append("GRPO-B_sft512_s2026: parent weights are not SFT-reverified_n512_s0")
for r in runs:
    if r.get("parent_full") and r["id"] != "GRPO-B_sft512_s2026" and r["parent_full"] != BASE_SHA:
        problems.append(f"{r['id']}: parent is not the pinned base model")


# learning rates of the main runs, re-derived with the pre-registered rule from the tuning runs' final-step GSM8K dev accuracy:
# take the highest; if another rate is within 1.0 pp of it, take the smallest such rate (EXPERIMENT_PLAN.md section 10)
def rule(cands):
    best = max(a_ for _, a_ in cands)
    return min(l_ for l_, a_ in cands if best - a_ <= 0.010 + 1e-12)


lr_rule = {}
for kind, tune in (("SFT", ["SFT-tune_lr1e-5", "SFT-tune_lr2e-5", "SFT-tune_lr5e-5"]), ("GRPO", ["GRPO-tune_lr2e-6", "GRPO-tune_lr5e-6", "GRPO-tune_lr1e-5"])):
    c = []
    for t in tune:
        d = find(t)
        if d and os.path.exists(d + "/DONE.json"):
            c.append((float(J(d + "/run.json")["config"]["train"]["lr"]), J(d + "/DONE.json")["eval"]["gsm8k_dev"]["accuracy"]))
    if len(c) == 3:
        want = rule(c)
        got = {r["id"]: float(J(find(r["id"]) + "/run.json")["config"]["train"]["lr"]) for r in runs
               if r["id"].startswith(kind) and "tune" not in r["id"] and r.get("folder") and r["id"] not in BRANCHES}
        lr_rule[kind] = dict(tuning={f"{l_:g}": a_ for l_, a_ in c}, rule_choice=want, recorded=got)
        for k, v in got.items():
            if abs(v - want) > 1e-15:
                problems.append(f"{k}: lr {v:g} is not the rule's choice {want:g}")

# manifests
WEIGHTS = (".safetensors", "state.pt")  # never downloaded: confirmed only on the server
manifests, MF, provenance = [], {}, {}
for t in TREES:
    for f in sorted(os.listdir(t)):
        if f.endswith(".txt") and "PROVENANCE" in f:
            provenance[f] = open(os.path.join(t, f), encoding="utf-8").read().strip()
        if not f.endswith(".sha256"):
            continue
        ent, n, bad, miss, excl = {}, 0, [], [], []
        for line in open(os.path.join(t, f), encoding="utf-8"):
            if not line.strip():
                continue
            h, p = line.rstrip("\n").split(None, 1)
            p = p.lstrip("*")
            ent[p] = h
            q = os.path.join(t, p)
            n += 1
            if not os.path.exists(q):
                (excl if p.endswith(WEIGHTS) else miss).append(p)
            elif sha_file(q) != h:
                bad.append(p)
        MF[f] = (t, ent)
        manifests.append(dict(tree=t, file=f, entries=n, verified=n - len(miss) - len(bad) - len(excl), missing=len(miss), changed=len(bad),
                              weights_not_downloaded=len(excl)))
        if miss or bad:
            problems.append(f"{t}/{f}: {len(miss)} missing, {len(bad)} changed: {(miss + bad)[:5]}")
        if excl:
            explained.append(f"{f}: {len(excl)} entries are model weights or optimizer states ({', '.join(sorted({os.path.basename(x) for x in excl}))}), excluded from the download "
                             "by design; their recorded hashes are cross-checked against DONE.json below and must be confirmed on the server")
for need in ("SESSION_A.sha256", "SESSION_B.sha256", "RECOVERY.sha256"):
    if need not in MF:
        problems.append(f"manifest {need} not recovered")

# cross-checks between manifests: SESSION_A/B were written retrospectively for the transfer; BEFORE_RECOVERY.sha256 was written on the server
# before the recovery branches ran. Every file both list must have the same hash, so the transferred files are the pre-recovery files.
cross = {}
if "BEFORE_RECOVERY.sha256" in MF:
    br = MF["BEFORE_RECOVERY.sha256"][1]
    for f in ("SESSION_A.sha256", "SESSION_B.sha256"):
        if f in MF:
            e = MF[f][1]
            common = set(e) & set(br)
            diff = sorted(p for p in common if e[p] != br[p])
            cross[f] = dict(entries=len(e), also_in_before_recovery=len(common), only_in_session_manifest=sorted(set(e) - set(br))[:10], differing=diff)
            if diff:
                problems.append(f"{f} and BEFORE_RECOVERY.sha256 disagree on {len(diff)} files: {diff[:5]}")
    # files listed before the recovery that no session manifest covers (other than weights)
    cov = set().union(*[set(MF[f][1]) for f in ("SESSION_A.sha256", "SESSION_B.sha256") if f in MF]) if any(f in MF for f in ("SESSION_A.sha256", "SESSION_B.sha256")) else set()
    cross["before_recovery_not_in_session_manifests"] = sorted(p for p in br if p not in cov and not p.endswith(WEIGHTS))[:20]
    # recorded server-side weight hashes against each run's DONE.json lineage
    wchk = []
    for r in runs:
        if r.get("weights_full") and r["id"] not in BRANCHES:
            p = f"{rel(r['id'])}/checkpoint/model.safetensors"
            ok = br.get(p) == r["weights_full"]
            wchk.append(dict(id=r["id"], manifest=br.get(p, "absent")[:16], done=r["weights_full"][:16], match=ok))
            r["weights_in_before_recovery"] = ok
            if not ok:
                problems.append(f"{r['id']}: weights hash in BEFORE_RECOVERY.sha256 ({br.get(p, 'absent')[:16]}) differs from DONE.json ({r['weights_full'][:16]})")
    cross["weights"] = wchk
    st_ = br.get("grpo/A_base_s2026/state.pt")
    cross["original_state_pt"] = st_
    if st_ and st_ != "73114307c664b55c05b6b6eddb326d0179525c89944dac10daef48f2a424e1bb":
        problems.append("grpo/A_base_s2026/state.pt in BEFORE_RECOVERY.sha256 differs from the incident backup hash")
# the failed original as copied in both trees must be identical where both have the file
o1, o2 = [os.path.join(t, "grpo", "A_base_s2026") for t in TREES[:2]] if len(TREES) > 1 else (None, None)
if o1 and o2 and os.path.isdir(o1) and os.path.isdir(o2):
    fl = [os.path.relpath(os.path.join(dp_, f_), o1) for dp_, _, fs in os.walk(o1) for f_ in fs]
    both = [x for x in fl if os.path.exists(os.path.join(o2, x))]
    dif = [x for x in both if sha_file(os.path.join(o1, x)) != sha_file(os.path.join(o2, x))]
    cross["original_copies"] = dict(files=len(fl), compared=len(both), differing=dif)
    if dif:
        problems.append(f"GRPO-A_base_s2026 copies differ between trees: {dif[:5]}")

# consolidated development results (final step; the failed original's last evaluation is step 75 and is diagnostic only)
A = None
if not a.no_second:
    import stage0_verify_audit as A  # noqa: E402
ans = lambda t: t.split("<answer>")[-1].replace("</answer>", "").strip()
dev = []
for r in runs:
    d = find(r["id"])
    if not d:
        continue
    tag = "step75" if r["id"] == "GRPO-A_base_s2026" else "final_step"
    row = dict(id=r["id"], point=tag, kind="post-hoc branch" if r["id"] in BRANCHES else ("FAILED run, diagnostic" if r["id"] == "GRPO-A_base_s2026" else "registered"),
               lr=None, gpu_h=r.get("gpu_h"))
    try:
        row["lr"] = J(d + "/run.json")["config"]["train"]["lr"]
    except KeyError:
        pass
    for name in ("gsm8k_dev", "math_dev"):
        p = f"{d}/eval/{tag}_{name}.jsonl"
        if not os.path.exists(p):
            continue
        x = jload(p)
        n, k = len(x), sum(bool(y["correct"]) for y in x)
        sec = [bool(y["correct"]) or (A is not None and bool(y["format"] and A.second_verifier(ans(y["response"]), str(y["answer"])))) for y in x]
        relx = [s or y["category"] == "format_fail_answer_right" for s, y in zip(sec, x)]
        row[name] = dict(n=n, strict=k / n, ci95=wilson(k, n), second=sum(sec) / n if A else None, relaxed=sum(relx) / n, format=st.mean(bool(y["format"]) for y in x),
                         clean_format=st.mean(bool(y.get("clean_format", y["format"])) for y in x), double_think=sum(y["response"].count("</think>") >= 2 for y in x),
                         tokens=st.mean(y["n_tokens"] for y in x), truncated=sum(y["finish"] == "length" for y in x))
    dev.append(row)
for t in TREES:  # registered base dev evaluation, if it was run
    p = os.path.join(t, "eval", "base_dev")
    if os.path.exists(p + "/DONE.json"):
        for pr in ("r1_zero", "question_only"):
            row = dict(id="EVAL-base-dev", point=pr, kind="registered", lr=None, gpu_h=None)
            for name in ("gsm8k_dev", "math_dev"):
                f = f"{p}/eval/{pr}_{name}.jsonl"
                if os.path.exists(f):
                    x = jload(f)
                    n, k = len(x), sum(bool(y["correct"]) for y in x)
                    row[name] = dict(n=n, strict=k / n, ci95=wilson(k, n), second=None, relaxed=None, format=st.mean(bool(y["format"]) for y in x),
                                     clean_format=None, double_think=None, tokens=st.mean(y["n_tokens"] for y in x), truncated=sum(y["finish"] == "length" for y in x))
            dev.append(row)
        break

os.makedirs(a.out)
json.dump(dict(trees=TREES, runs=runs, manifests=manifests, cross_checks=cross, lr_rule=lr_rule, provenance=provenance, dev=dev, problems=problems, explained=explained),
          open(a.out + "/audit.json", "w"), indent=1, default=str)
pc = lambda v: "-" if v is None else f"{100 * v:.1f}"
yn = lambda v: "-" if v is None else ("yes" if v else "**NO**")
L = ["# Artifact audit and consolidated development results (generated)", "", f"Trees: {', '.join(TREES)}", "",
     f"**Unresolved problems: {len(problems)}**", ""] + [f"- {p}" for p in problems] + ["", f"Explained (not problems): {len(explained)}", ""] + [f"- {p}" for p in explained]
L += ["", "Manifest provenance: " + ("; ".join(f"{k}: {v.replace(chr(10), ' ')}" for k, v in provenance.items()) or "none recorded"),
      "SESSION_A.sha256 and SESSION_B.sha256 were generated retrospectively for transfer verification (after the sessions, not during them). "
      "Their evidence that the transferred files are the pre-recovery files is the agreement with BEFORE_RECOVERY.sha256, written on the server before the recovery branches ran.", "",
      "| run | expected | status | config hash | verifier | data file | data ids | ckpt lineage | weights = server manifest | code | rows | dup | last | resumes | GPU-h | weights | chain | config differences |", "|" + "---|" * 18]
for r in runs:
    cd = "; ".join(f"{x['key']} {x['registered']}->{x['recorded']}" for x in r.get("config_diffs") or []) or "-"
    L.append(f"| {r['id']} | {r['expected']} | {r['status']} | {yn(r.get('config_sha256_ok'))} | {r.get('verifier', '-')} | {yn(r.get('data_file_ok'))} | {yn(r.get('data_ids_ok'))} | {yn(r.get('checkpoint_lineage_ok'))} | "
             f"{yn(r.get('weights_in_before_recovery'))} | {r.get('code_sha256', '-')} | {r.get('train_rows', '-')} | {r.get('duplicated_step_rows', '-')} | {r.get('last_step', '-')} | {r.get('resumes', '-')} | "
             f"{'-' if r.get('gpu_h') is None else f'{r['gpu_h']:.2f}'} | {r.get('weights_sha256', '-')} | {r.get('chain', '-')} | {cd} |")
L += ["", "| manifest | entries | verified locally | missing | changed | weights not downloaded |", "|---|---|---|---|---|---|"]
L += [f"| {m['tree']}/{m['file']} | {m['entries']} | {m['verified']} | {m['missing']} | {m['changed']} | {m['weights_not_downloaded']} |" for m in manifests]
L += ["", "Learning-rate rule re-derived from the tuning runs: " + "; ".join(f"{k}: tuning {v['tuning']} -> {v['rule_choice']:g}; main runs {sorted(set(v['recorded'].values()))}" for k, v in lr_rule.items())]
L += ["", "Cross-checks: " + json.dumps({k: v for k, v in cross.items() if k != "weights"}, default=str)]
L += ["", "## Development results (strict = frozen verifier 2.0.0; %)", "",
      "| run | kind | point | lr | GSM8K strict (95% CI) | second | relaxed | format | clean | doubled </think> | tokens | MATH strict (95% CI) | second | relaxed | format | trunc | GPU-h |", "|" + "---|" * 17]
for r in dev:
    g, m = r.get("gsm8k_dev") or {}, r.get("math_dev") or {}
    ci = lambda e: f"{pc(e.get('strict'))} ({pc(e['ci95'][0])}-{pc(e['ci95'][1])})" if e else "-"
    L.append(f"| {r['id']} | {r['kind']} | {r['point']} | {r['lr'] if r['lr'] is not None else '-'} | {ci(g)} | {pc(g.get('second'))} | {pc(g.get('relaxed'))} | {pc(g.get('format'))} | {pc(g.get('clean_format'))} | {g.get('double_think', '-')} | {'-' if not g else f'{g['tokens']:.0f}'} | {ci(m)} | {pc(m.get('second'))} | {pc(m.get('relaxed'))} | {pc(m.get('format'))} | {m.get('truncated', '-')} | {'-' if r['gpu_h'] is None else f'{r['gpu_h']:.2f}'} |")
open(a.out + "/AUDIT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
