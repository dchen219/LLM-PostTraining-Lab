"""Frozen held-out evaluation: plan rules, checkpoint identity, lock, launcher gating, report statistics. CPU only; no model is loaded."""
import copy, json, os, subprocess, sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from posttrain import final_eval as F  # noqa: E402
from posttrain.common import sha_file  # noqa: E402
from posttrain.stats import holm, mcnemar_exact  # noqa: E402


@pytest.fixture
def fz():
    return F.load_freeze()[0]


def test_final_freeze_is_valid_and_follows_the_registry(fz):
    assert F.validate(fz) == []
    ids = {c["id"] for c in fz["checkpoints"]}
    assert "GRPO-A_base_s2026" not in ids and not any(i.startswith(("SFT-tune", "GRPO-tune")) for i in ids)
    assert {"SFT-exact_full_s0", "SFT-reverified_full_s0", "SFT-exact_n512_s0", "SFT-reverified_n512_s0", "GRPO-B_sft512_s2026", "BASE"} <= ids
    base = next(c for c in fz["checkpoints"] if c["id"] == "BASE")
    assert base["prompts"] == ["r1_zero", "question_only"]
    post = {c["id"] for c in fz["checkpoints"] if c["class"] == "posthoc"}
    assert post == {"GRPO-A-replay80-lr1e-5-s2026", "GRPO-A-recovery80-lr5e-6-s2026"}
    conf = fz["families"]["confirmatory"]
    assert conf["holm_m"] == 4 and any(c.get("unavailable") and "GRPO-A_base_s2026" in c["b"] for c in conf["comparisons"])
    assert fz["engine"]["prefix_caching"] is False and fz["engine"]["temperature"] == 0.0


@pytest.mark.parametrize("mut, msg", [
    (lambda z: z["checkpoints"].append(dict(id="SFT-tune_lr2e-5", dir="sft/tune_lr2e-5", stage="sft", prompts=["r1_zero"], class_="x")), "tuning"),
    (lambda z: z["checkpoints"].append(dict(id="GRPO-A_base_s2026", dir="grpo/A_base_s2026", stage="grpo", prompts=["r1_zero"], **{"class": "preregistered"})), "failed"),
    (lambda z: z["families"]["confirmatory"]["comparisons"][0].update(b="GRPO-A-replay80-lr1e-5-s2026:r1_zero"), "post-hoc"),
    (lambda z: z["families"]["sft_vs_base"].update(holm_m=6), "holm_m"),
    (lambda z: z["checkpoints"][-1].update(**{"class": "preregistered"}), "recovery branch"),
    (lambda z: z["families"]["descriptive"]["comparisons"][0].update(a="SFT-exact_full_s0:question_only"), "not evaluated with"),
    (lambda z: z.update(not_evaluated=[]), "not evaluated"),
])
def test_final_freeze_mutations_are_refused(fz, mut, msg):
    z = copy.deepcopy(fz)
    mut(z)
    for c in z["checkpoints"]:
        if "class_" in c:
            c["class"] = c.pop("class_")
            c["class"] = "preregistered"
    bad = F.validate(z)
    assert bad and any(msg in b for b in bad), bad


def make_ckpt(runs, rel, stage="sft", done=True, failed=False, tamper=False):
    d = os.path.join(runs, rel)
    c = os.path.join(d, "checkpoint")
    os.makedirs(c)
    open(os.path.join(c, "model.safetensors"), "wb").write(os.urandom(64))
    w = {"model.safetensors": sha_file(os.path.join(c, "model.safetensors"))}
    lin = dict(stage=stage, chain=["base", stage], weights_sha256=w)
    json.dump(lin, open(os.path.join(c, "lineage.json"), "w"))
    if done:
        json.dump(dict(steps=10, wall_s=3600, lineage=lin), open(os.path.join(d, "DONE.json"), "w"))
    if failed:
        json.dump(dict(error="x"), open(os.path.join(d, "FAILED.json"), "w"))
    if tamper:
        open(os.path.join(c, "model.safetensors"), "ab").write(b"x")
    return w


J = lambda i, rel, stage="sft": dict(id=i, dir=rel, stage=stage, prompts=["r1_zero"], cls="preregistered")


def test_final_check_one_identity(tmp_path):
    r = str(tmp_path)
    w = make_ckpt(r, "sft/a")
    assert F.check_one(r, J("SFT-a", "sft/a"))["weights_sha256"] == w
    make_ckpt(r, "sft/b", tamper=True)
    with pytest.raises(SystemExit, match="differ"):
        F.check_one(r, J("SFT-b", "sft/b"))
    make_ckpt(r, "sft/c", done=False)
    with pytest.raises(SystemExit, match="no DONE"):
        F.check_one(r, J("SFT-c", "sft/c"))
    make_ckpt(r, "grpo/d", stage="grpo", failed=True)
    with pytest.raises(SystemExit, match="FAILED"):
        F.check_one(r, J("GRPO-d", "grpo/d", "grpo"))
    make_ckpt(r, "sft/e")
    with pytest.raises(SystemExit, match="stage"):
        F.check_one(r, J("SFT-e", "sft/e", "grpo"))


def test_final_lock_written_once_and_never_replaced(tmp_path):
    r = str(tmp_path)
    lk = dict(freeze_sha256="f1", checkpoints={"X": dict(weights_sha256={"m": "1"})})
    F.write_lock(r, lk)
    F.write_lock(r, lk)  # identical: accepted
    with pytest.raises(SystemExit, match="refusing to replace"):
        F.write_lock(r, dict(lk, checkpoints={"X": dict(weights_sha256={"m": "2"})}))
    with pytest.raises(SystemExit, match="different freeze"):
        F.read_lock(r, "f2")


def small_fz(n=3):
    return dict(checkpoints=[dict(id=f"SFT-x{i}", dir=f"sft/x{i}", stage="sft", prompts=["r1_zero"], **{"class": "preregistered"}) for i in range(n)])


def test_final_worker_gating_and_split(tmp_path, monkeypatch):
    r = str(tmp_path)
    fz = small_fz(5)
    recs = []
    for c in fz["checkpoints"]:
        make_ckpt(r, c["dir"])
        recs.append(F.check_one(r, dict(c, cls="preregistered")))
    monkeypatch.delenv("A5_FINAL_EVAL_APPROVED", raising=False)
    monkeypatch.delenv("ALLOW_FINAL_EVAL", raising=False)
    with pytest.raises(SystemExit, match="REFUSED"):
        F.run_worker(r, fz, "f", 0, 1, call=lambda *a, **k: 0)
    monkeypatch.setenv("A5_FINAL_EVAL_APPROVED", "yes")
    with pytest.raises(SystemExit, match="no FINAL_CHECKPOINTS"):
        F.run_worker(r, fz, "f", 0, 1, call=lambda *a, **k: 0)
    F.write_lock(r, dict(freeze_sha256="f", checkpoints={x["id"]: dict(path=x["path"], stage="sft", weights_sha256=x["weights_sha256"]) for x in recs}))
    seen = []

    def call(cmd, cwd, env):
        seen.append(cmd)
        assert env["ALLOW_FINAL_EVAL"] == "1"
        return 0
    done = F.run_worker(r, fz, "f", 0, 2, call=call) + F.run_worker(r, fz, "f", 1, 2, call=call)
    assert sorted(done) == sorted(c["id"] for c in fz["checkpoints"])  # every job exactly once across the workers
    assert "ALLOW_FINAL_EVAL" not in os.environ
    for cmd in seen:
        s = cmd[cmd.index("--set") + 1:]
        assert any(x.startswith("run_dir=" + os.path.join(r, "final_eval")) for x in s)
        assert "init.expect_stage=sft" in s and any(x.startswith("init.path=") and x.endswith("checkpoint") for x in s)
        assert "--resume" not in cmd
    # a finished evaluation is skipped; weights changed after the lock are refused
    os.makedirs(os.path.join(r, "final_eval", "SFT-x0"), exist_ok=True)
    json.dump({}, open(os.path.join(r, "final_eval", "SFT-x0", "DONE.json"), "w"))
    import shutil
    shutil.rmtree(os.path.join(r, "sft/x1"))
    make_ckpt(r, "sft/x1")  # a consistent, finished, but different checkpoint in the frozen folder
    with pytest.raises(SystemExit, match="changed since the lock"):
        F.run_worker(r, fz, "f", 0, 1, call=lambda *a, **k: 0)
    open(os.path.join(r, "sft/x1/checkpoint/model.safetensors"), "ab").write(b"x")
    with pytest.raises(SystemExit, match="differ"):
        F.run_worker(r, fz, "f", 0, 1, call=lambda *a, **k: 0)


def test_final_launcher_refuses_without_authorization(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("A5_", "PT_"))}
    sh = os.path.join(ROOT, "scripts", "server_final_eval.sh")
    out = subprocess.run(["bash", sh, "run"], capture_output=True, text=True, env=env)
    assert out.returncode != 0 and "REFUSED" in out.stdout
    env.update(PT_PLATFORM="research_server", PT_HW="a6000", A5_AUTHORIZED="yes", PT_RUNS=str(tmp_path))
    out = subprocess.run(["bash", sh, "run"], capture_output=True, text=True, env=env)
    assert out.returncode != 0 and "A5_FINAL_EVAL_APPROVED" in out.stdout
    assert not os.path.exists(os.path.join(tmp_path, "final_eval"))
    txt = open(sh, encoding="utf-8").read()
    assert not any(m in txt for m in ("train_sft", "train_grpo", "train_ei", "fork_grpo", "server_tier_a", "run_experiment"))  # inference only


def write_eval(final, cid, prompt, sset, correct, weights, fmt=True):
    d = os.path.join(final, cid)
    os.makedirs(os.path.join(d, "eval"), exist_ok=True)
    rows = [dict(id=f"q{i}", question="", answer="1", response="<think>x</think> <answer>1</answer>", n_tokens=10 + i, finish="stop", correct=c, format=fmt,
                 clean_format=fmt, error=None, rule=None, category="correct" if c else "format_ok_answer_wrong") for i, c in enumerate(correct)]
    with open(os.path.join(d, "eval", f"{prompt}_{sset}.jsonl"), "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    sp = os.path.join(d, "summary.json")
    s = json.load(open(sp)) if os.path.exists(sp) else dict(results={}, parent=dict(weights_sha256=weights))
    s["results"].setdefault(prompt, {})[sset] = dict(n=len(rows), correct=sum(correct), gen_s=1.0, categories={})
    json.dump(s, open(sp, "w"))
    json.dump({}, open(os.path.join(d, "DONE.json"), "w"))


def test_final_report_statistics(tmp_path):
    import final_report as R
    fz, fsha = F.load_freeze()
    final = str(tmp_path / "final")
    n = 200
    lock = dict(freeze_sha256=fsha, checkpoints={})
    for k, c in enumerate(fz["checkpoints"]):
        w = {"model.safetensors": f"w{k}"}
        lock["checkpoints"][c["id"]] = dict(weights_sha256=w)
        for pr in c["prompts"]:
            for s in R.SETS:
                if c["id"] == "SFT-reverified_full_s0":
                    corr = [i % 10 < 8 for i in range(n)]      # 80%
                elif c["id"] == "SFT-exact_full_s0":
                    corr = [i % 10 < 6 for i in range(n)]      # 60%, nested inside the 80%: b_only = 40, a_only = 0
                else:
                    corr = [i % 2 == 0 for i in range(n)]
                write_eval(final, c["id"], pr, s, corr, w)
    json.dump(lock, open(os.path.join(final, "FINAL_CHECKPOINTS.json"), "w"))
    out = R.build(final, [], second=False)
    assert out["missing"] == [] and out["problems"] == []
    conf = [g for g in out["comparisons"] if g["family"] == "confirmatory" and g["set"] == "gsm8k_test_final"]
    c1 = next(g for g in conf if g["name"].startswith("C1"))
    assert (c1["a_only"], c1["b_only"]) == (0, 40) and abs(c1["diff"] - 0.2) < 1e-12
    p = mcnemar_exact(0, 40)
    assert c1["p"] == p
    ps = {g["name"]: g["p"] for g in conf}
    assert ps[next(k for k in ps if k.startswith("C4"))] == 1.0 and len(ps) == 4  # the failed run keeps m = 4
    assert c1["p_holm"] == holm(ps)[c1["name"]] == min(1.0, 4 * p)
    assert c1["verdict"] == "supported"
    c4 = next(g for g in conf if g["name"].startswith("C4"))
    assert c4["verdict"] == "unavailable"
    post = [g for g in out["comparisons"] if g["family"] == "posthoc"]
    assert post and all(g["verdict"].startswith("exploratory") for g in post)
    assert all(g["label"] == "post-hoc exploratory" for g in post)
    # a lock for another freeze file, or evaluated weights that differ from the lock, are refused or reported
    lock2 = dict(lock, freeze_sha256="other")
    json.dump(lock2, open(os.path.join(final, "FINAL_CHECKPOINTS.json"), "w"))
    with pytest.raises(SystemExit, match="different freeze"):
        R.build(final, [], second=False)
    lock["checkpoints"]["BASE"] = dict(weights_sha256={"model.safetensors": "zzz"})
    json.dump(lock, open(os.path.join(final, "FINAL_CHECKPOINTS.json"), "w"))
    assert any("BASE" in p for p in R.build(final, [], second=False)["problems"])


def test_final_small_difference_is_not_detected(tmp_path):
    """A GSM8K difference below the frozen 1.4 pp is 'not detected' even if p < 0.05."""
    import final_report as R
    fz, fsha = F.load_freeze()
    final = str(tmp_path / "f")
    n = 4000
    lock = dict(freeze_sha256=fsha, checkpoints={})
    for k, c in enumerate(fz["checkpoints"]):
        w = {"m": f"w{k}"}
        lock["checkpoints"][c["id"]] = dict(weights_sha256=w)
        for pr in c["prompts"]:
            for s in R.SETS:
                bump = 50 if c["id"] == "SFT-reverified_n512_s0" else 0  # +1.25 pp, b_only 50, a_only 0
                write_eval(final, c["id"], pr, s, [i < 2000 + bump for i in range(n)], w)
    json.dump(lock, open(os.path.join(final, "FINAL_CHECKPOINTS.json"), "w"))
    out = R.build(final, [], second=False)
    c2 = next(g for g in out["comparisons"] if g["name"].startswith("C2") and g["set"] == "gsm8k_test_final")
    assert c2["p_holm"] < 0.05 and c2["verdict"] == "not detected"
    c2m = next(g for g in out["comparisons"] if g["name"].startswith("C2") and g["set"] == "math_val_final")
    assert c2m["verdict"] == "supported"  # the detectable-size rule is frozen for GSM8K only
