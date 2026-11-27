"""Post-hoc recovery branches (posttrain.fork_grpo), failed-batch evidence, the incident guard, log rewind and update-size diagnostics. CPU only."""
import glob, json, math, os, shutil, subprocess, sys
import pytest, torch, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_posttrain import CFG, ROOT, CharTok, FakeVLLM, SaveTok, no_server_settings, tiny  # noqa: E402,F401  (the fixture keeps server settings out)
from posttrain.common import Run, jload, sha_file  # noqa: E402
from posttrain.model_io import make_optimizer  # noqa: E402
from posttrain.train_grpo import grpo_update  # noqa: E402
from posttrain.train_sft import sft_step  # noqa: E402


def _tree(d, skip=()):
    out = {}
    for root, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(root, f)
            if not any(s in p for s in skip):
                out[os.path.relpath(p, d)] = sha_file(p)
    return out


def _grpo_env(tmp_path, monkeypatch, fault_at=None, fault="wrong_logprobs"):
    """Tiny model, stand-in vLLM (broken from generate call `fault_at` on), rewards that vary inside each group, a recorder around grpo_update."""
    import posttrain.train_grpo as G
    base = str(tmp_path / "base")
    if not os.path.exists(base):
        tiny().save_pretrained(base)
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps(dict(id=f"p{i}", question=f"What is {i}?", answer="4")) + "\n" for i in range(6)))
    monkeypatch.setenv("PT_RUNS", str(tmp_path / "runs")); monkeypatch.setenv("PT_DATA", str(tmp_path)); monkeypatch.setenv("PT_MODEL", base)
    monkeypatch.setattr(G, "load_tokenizer", lambda p: SaveTok())
    seen = dict(gen=0, updates=0, calls=[])

    class Eng(FakeVLLM):
        def generate(self, *a, **k):
            seen["gen"] += 1
            self.fault = fault if fault_at and seen["gen"] >= fault_at else None
            return super().generate(*a, **k)
    monkeypatch.setattr(G, "make_engine", lambda cfg, path, model, tok: Eng(model))

    def upd(*a, **k):
        m, o = a[0], a[1]
        st = next(iter(o.state.values()), None)
        seen["calls"].append(dict(lr=o.param_groups[0]["lr"], sd={x: v.clone() for x, v in m.state_dict().items()},
                                  opt_step=float(st["step"]) if st else 0.0, exp_avg=st["exp_avg"].clone() if st else None))
        seen["updates"] += 1
        return grpo_update(*a, **k)
    monkeypatch.setattr(G, "grpo_update", upd)
    k = iter(range(10 ** 6))
    monkeypatch.setattr(G, "score", lambda t, g: dict(reward=float(next(k) % 2), answer_reward=0.0, format_reward=0.0, error=None, rule=None))
    return G, seen, pool


def failed_source(tmp_path, monkeypatch):
    """Shaped like GRPO-A_base_s2026: 6 planned steps, state saved at 2 and 4, 5 updates completed, step 6 rejected by the rollout gate."""
    G, seen, pool = _grpo_env(tmp_path, monkeypatch, fault_at=6)
    monkeypatch.setattr(sys, "argv", ["train_grpo", "--config", ROOT + "/configs/grpo/A_base_s2026.yaml", "--set", "device=cpu", "engine.name=hf", "eval.sets={}",
                                      "eval.at_start=false", f"data.path={pool}", "init.revision=null", "grpo.steps=6", "grpo.prompts_per_step=2", "grpo.group_size=2",
                                      "grpo.update_batch=4", "grpo.max_tokens=6", "grpo.sync_check_every=1", "train.micro_batch=2", "train.lr=1e-2", "train.save_every=2",
                                      "eval.every=null"])
    with pytest.raises(SystemExit, match="rollout log-prob check failed"):
        G.main()
    src = tmp_path / "runs" / "grpo" / "A_base_s2026"
    assert seen["updates"] == 5 and torch.load(src / "state.pt", weights_only=False)["step"] == 4
    bk = tmp_path / "backup"; bk.mkdir(); shutil.copy2(src / "state.pt", bk / "state.pt")
    return src, str(bk / "state.pt")


def _spec(tmp_path, src, state, name, changes, **over):
    s = dict(id=f"T-{name}", purpose="test", run_dir=str(tmp_path / "runs" / "recovery" / name),
             source=dict(experiment_id="GRPO-A_base_s2026", run_dir=str(src), state=state, state_sha256=sha_file(state), step=4, completed_updates=5),
             expect_source={"grpo.steps": 6, "train.micro_batch": 2}, changes=changes)
    for k, v in over.items():
        (s if k in ("run_dir", "expect_source", "changes") else s["source"])[k] = v
    p = tmp_path / f"{name}.yaml"; p.write_text(yaml.safe_dump(s))
    return str(p)


def _fork(monkeypatch, spec, *extra):
    import posttrain.fork_grpo as F
    monkeypatch.setattr(sys, "argv", ["fork_grpo", "--spec", spec, *extra])
    F.main()


def _rows(d, ev="train"):
    return [r for r in jload(os.path.join(d, "metrics.jsonl")) if r["event"] == ev]


def test_recovery_branches_start_identically_change_only_lr_and_leave_everything_else_untouched(tmp_path, monkeypatch):
    src, state = failed_source(tmp_path, monkeypatch)
    done = tmp_path / "runs" / "sft" / "finished_run"; done.mkdir(parents=True); (done / "DONE.json").write_text("{}"); (done / "metrics.jsonl").write_text("{}\n")
    before, bk = _tree(str(tmp_path / "runs"), skip=("recovery",)), sha_file(state)
    G, seen, _ = _grpo_env(tmp_path, monkeypatch)
    _fork(monkeypatch, _spec(tmp_path, src, state, "A1", {"train.save_every": 1, "grpo.save_rollouts_every": 1}))
    a1 = seen["calls"][:]; seen["calls"].clear()
    _fork(monkeypatch, _spec(tmp_path, src, state, "A2", {"train.lr": 5e-3, "train.save_every": 1, "grpo.save_rollouts_every": 1}))
    a2 = seen["calls"]
    assert _tree(str(tmp_path / "runs"), skip=("recovery",)) == before and sha_file(state) == bk     # source, its later logs and finished runs unchanged
    assert float(a1[0]["lr"]) == 1e-2 and a2[0]["lr"] == 5e-3 and all(c["lr"] == 5e-3 for c in a2)   # A2's first resumed update already uses the new rate
    assert a1[0]["opt_step"] == a2[0]["opt_step"] == 4.0 and torch.equal(a1[0]["exp_avg"], a2[0]["exp_avg"])
    assert all(torch.equal(a1[0]["sd"][k], a2[0]["sd"][k]) for k in a1[0]["sd"])                       # identical model and optimizer at the fork
    saved = torch.load(state, weights_only=False)
    assert all(torch.equal(a1[0]["sd"][k], saved["model"][k]) for k in saved["model"])
    d1, d2 = (str(tmp_path / "runs" / "recovery" / n) for n in ("A1", "A2"))
    for d in (d1, d2):
        assert [r["step"] for r in _rows(d)] == [5, 6]                                                  # only steps after the fork, once each
        assert [r["step"] for r in _rows(d, "eval")] == [4, 6]                                          # the fork state is evaluated before any update
        dj = json.load(open(d + "/DONE.json"))
        assert dj["source_check"]["source_unchanged"] is True and dj["lineage"]["extra"]["fork"]["source_step"] == 4
        assert all(r["update_l2"] > 0 and "rollout_logprob_p99" in r and "rollout_logprob_over1" in r for r in _rows(d))
    assert _rows(d1, "fork_start")[0]["state_digest"] == _rows(d2, "fork_start")[0]["state_digest"]
    diff = {x["key"]: (x["source"], x["branch"]) for x in json.load(open(d2 + "/run.json"))["config"]["fork"]["differences"]}
    assert set(diff) == {"run_dir", "train.lr", "train.save_every", "grpo.save_rollouts_every"} and diff["train.lr"][1] == 5e-3
    assert diff["grpo.save_rollouts_every"] == (10, 1)
    f1 = json.load(open(d1 + "/run.json"))["config"]["fork"]
    assert {x["key"] for x in f1["differences"]} == {"run_dir", "train.save_every", "grpo.save_rollouts_every"}
    assert f1["criteria"]["sha256"] == sha_file(ROOT + "/recovery/diagnostic_criteria.yaml") and f1["criteria"]["version"] == 1
    for d in (d1, d2):  # every step's rollouts, with token-level detail
        for s in (5, 6):
            r = jload(f"{d}/rollouts/step{s}.jsonl")
            assert len(r) == 4 and all(x["step"] == s and len(x["token_ids"]) == len(x["sampling_logprobs"]) == len(x["policy_logprobs"]) == x["n_tokens"] > 0
                                       and x["entropy"] is not None and x["gap_mean"] is not None and "truncated" in x and "format_reward" in x
                                       and x["prompt_id"] for x in r)
    with pytest.raises(SystemExit, match="already complete"):
        _fork(monkeypatch, _spec(tmp_path, src, state, "A1", {"train.save_every": 1, "grpo.save_rollouts_every": 1}), "--resume")
    for n in ("A1", "A2"):  # the report reads the folder names of the real branches
        os.rename(tmp_path / "runs" / "recovery" / n, tmp_path / "runs" / "recovery" / {"A1": "A_replay80_lr1e-5_s2026", "A2": "A_recovery80_lr5e-6_s2026"}[n])
    r = subprocess.run([sys.executable, ROOT + "/scripts/recovery_report.py", "--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "rep")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(tmp_path / "rep" / "report.json"))
    assert rep["original"]["steps"] == 5 and rep["original"]["duplicated_rows"] == 0
    assert {b["status"] for b in rep["branches"].values()} == {"finished"} and len({b["state_digest"] for b in rep["branches"].values()}) == 1
    txt = open(tmp_path / "rep" / "REPORT.md").read()
    assert "Replay fidelity" in txt and "Registered diagnostic criteria" in txt
    assert all(b["criteria"]["successful_completion"]["hit"] is False for b in rep["branches"].values())  # test branches end at step 6, not 100


def test_recovery_branch_refusals(tmp_path, monkeypatch):
    src, state = failed_source(tmp_path, monkeypatch)
    _grpo_env(tmp_path, monkeypatch)
    for kw, msg in ((dict(state_sha256="0" * 64), "sha256"), (dict(changes={"grpo.clip": 0.1}), "may not be changed"),
                    (dict(run_dir=str(tmp_path / "runs" / "grpo" / "B")), "recovery"), (dict(expect_source={"grpo.steps": 100}), "spec expects"),
                    (dict(step=3), "save point"), (dict(completed_updates=6), "updates were completed"), (dict(step=2), "at step 4, expected 2")):
        ch = kw.pop("changes", {})
        with pytest.raises(SystemExit, match=msg):
            _fork(monkeypatch, _spec(tmp_path, src, state, "R", ch, **kw))
    x = tmp_path / "runs" / "recovery" / "X"; x.mkdir(parents=True); (x / "keep.txt").write_text("x")
    with pytest.raises(SystemExit, match="exists"):
        _fork(monkeypatch, _spec(tmp_path, src, state, "X", {}))
    assert os.listdir(x) == ["keep.txt"]


def test_recovery_source_check_detects_a_changed_source(tmp_path, monkeypatch):
    import posttrain.fork_grpo as F
    src, state = failed_source(tmp_path, monkeypatch)
    _, fork = F.build(_spec(tmp_path, src, state, "C", {}))
    assert fork.check_source()["source_unchanged"] is True
    with open(src / "metrics.jsonl", "a") as f:
        f.write("{}\n")
    assert fork.check_source()["source_unchanged"] is False
    _, fork = F.build(_spec(tmp_path, src, state, "C", {}))
    with open(state, "ab") as f:
        f.write(b"x")
    assert fork.check_source()["source_unchanged"] is False


def test_recovery_gate_failure_saves_the_batch_and_makes_no_update(tmp_path, monkeypatch):
    src, state = failed_source(tmp_path, monkeypatch)
    G, seen, _ = _grpo_env(tmp_path, monkeypatch, fault_at=1)
    sp = _spec(tmp_path, src, state, "F", {"train.save_every": 1})
    with pytest.raises(SystemExit, match="rollout log-prob check failed"):
        _fork(monkeypatch, sp)
    d = tmp_path / "runs" / "recovery" / "F"
    assert seen["updates"] == 0 and _rows(str(d)) == [] and not (d / "checkpoint").exists() and not (d / "state.pt").exists()
    f = json.load(open(d / "FAILED.json"))
    assert f["completed_updates"] == 4 and f["attempted_step"] == 5 and f["optimizer_updated_on_attempt"] is False and f["source_check"]["source_unchanged"]
    m = json.load(open(d / "failed_batch_step5" / "meta.json"))
    assert m["optimizer_updated"] is False and m["mismatch"]["mean"] > 0.05 and len(m["prompt_ids"]) == 2 and float(m["lr_for_attempt"]) == 1e-2
    r = jload(str(d / "failed_batch_step5" / "rollouts.jsonl"))
    assert len(r) == 4 and all(len(x["token_ids"]) == len(x["sampling_logprobs"]) == len(x["policy_logprobs"]) and x["worst"] and "reward" in x for x in r)
    with pytest.raises(SystemExit, match="FAILED.json"):
        _fork(monkeypatch, sp, "--resume")


def test_recovery_branch_interrupted_resumes_from_its_own_state_without_duplicate_steps(tmp_path, monkeypatch):
    src, state = failed_source(tmp_path, monkeypatch)
    G, seen, _ = _grpo_env(tmp_path, monkeypatch)
    real = G.run_eval_sets

    def stop_once(eng, sets, run_dir, tag, **k):
        if tag == "final_step" and not seen.get("stopped"):
            seen["stopped"] = True
            raise KeyboardInterrupt
        return real(eng, sets, run_dir, tag, **k)
    monkeypatch.setattr(G, "run_eval_sets", stop_once)
    sp = _spec(tmp_path, src, state, "I", {"train.save_every": 1})
    with pytest.raises(KeyboardInterrupt):
        _fork(monkeypatch, sp)
    d = str(tmp_path / "runs" / "recovery" / "I")
    assert [r["step"] for r in _rows(d)] == [5, 6] and torch.load(d + "/state.pt", weights_only=False)["step"] == 5 and not os.path.exists(d + "/FAILED.json")
    _fork(monkeypatch, sp, "--resume")
    assert [r["step"] for r in _rows(d)] == [5, 6] and len(_rows(d, "fork_start")) == 1
    assert [r["step"] for r in jload(d + "/metrics.superseded.jsonl")] == [6] and os.path.exists(d + "/DONE.json")


def test_incident_failed_original_is_never_resumed_and_ordinary_config_check_remains(tmp_path, monkeypatch):
    inc = json.load(open(ROOT + "/recovery/incident_GRPO-A_base_s2026.json"))
    assert inc["completed_updates"] == 97 and inc["attempted_step"] == 98 and inc["last_saved_state"]["step"] == 80 and inc["status"] == "FAILED"
    monkeypatch.setenv("PT_RUNS", str(tmp_path))
    d = tmp_path / inc["run_dir_rel"]; d.mkdir(parents=True)
    json.dump(dict(config_sha256="x"), open(d / "run.json", "w"))
    for resume in (False, True):
        with pytest.raises(SystemExit, match="never resumed"):
            Run(dict(run_dir=str(d)), "grpo", resume)
    assert os.listdir(d) == ["run.json"]
    cfg = dict(run_dir=str(tmp_path / "grpo" / "other"), a=1); Run(cfg, "grpo").log(dict(event="train", step=1))
    with pytest.raises(SystemExit, match="config differs"):
        Run(dict(cfg, a=2), "grpo", resume=True)
    sh = open(ROOT + "/scripts/server_tier_a.sh", encoding="utf-8").read()
    assert sh.count("recovery/incident_$i.json") >= 2 and "never resumed" in sh


def test_rewind_moves_late_rows_and_files_aside(tmp_path):
    r = Run(dict(run_dir=str(tmp_path / "r")), "x")
    for s in (1, 2, 3):
        r.log(dict(event="train", step=s))
    os.makedirs(tmp_path / "r" / "rollouts"); (tmp_path / "r" / "rollouts" / "step3.jsonl").write_text("x"); (tmp_path / "r" / "rollouts" / "step2.jsonl").write_text("x")
    assert r.rewind(2) == dict(rows=1, files=1)
    assert [x["step"] for x in jload(str(tmp_path / "r" / "metrics.jsonl"))] == [1, 2] and os.path.exists(tmp_path / "r" / "rollouts" / "step2.jsonl")
    assert glob.glob(str(tmp_path / "r" / "superseded" / "*" / "rollouts" / "step3.jsonl")) and len(jload(str(tmp_path / "r" / "metrics.superseded.jsonl"))) == 1


def test_adam_update_stats_equal_the_actual_parameter_change():
    from posttrain.diagnostics import adam_update_stats
    m = tiny(); o = make_optimizer(m, dict(lr=1e-3, betas=(0.9, 0.95))); t = CharTok()
    for _ in range(3):
        before = [p.detach().clone() for p in m.parameters()]
        sft_step(m, o, t, [("ab", "cdefg"), ("abc", "xyz")], CFG, "cpu")
        u = adam_update_stats(o, chunk=100)  # small chunks: the chunked path must give the same answer
        real = sum(float((p.detach() - b).pow(2).sum()) for p, b in zip(m.parameters(), before)) ** 0.5
        assert math.isclose(u["update_l2"], real, rel_tol=1e-3) and 0 < u["update_rel"] < 1 and u["update_max_abs"] < 1e-2


def test_recovery_specs_differ_only_in_learning_rate():
    a, b = (yaml.safe_load(open(f"{ROOT}/recovery/{n}.yaml")) for n in ("A_replay80_lr1e-5_s2026", "A_recovery80_lr5e-6_s2026"))
    assert a["changes"]["train.lr"] == 1e-5 == a["expect_source"]["train.lr"] and b["changes"]["train.lr"] == 5e-6
    for x in (a, b):
        x.pop("id"); x.pop("purpose"); x.pop("run_dir"); x["changes"].pop("train.lr")
    assert a == b and a["source"]["step"] == 80 and a["source"]["completed_updates"] == 97 and a["source"]["state_sha256"].startswith("73114307")
    assert a["expect_source"]["grpo.rollout_tol"] == 0.05 and a["expect_source"]["engine.prefix_caching"] is False
    import posttrain.fork_grpo as F
    assert set(F.ALLOWED) == {"train.lr", "train.save_every", "grpo.save_rollouts_every"}
    assert a["changes"] == {"train.save_every": 5, "grpo.save_rollouts_every": 1}


def test_recovery_launcher_is_gated_and_has_no_dangerous_commands():
    s = open(ROOT + "/scripts/server_recovery.sh", encoding="utf-8").read()
    assert "A5_RECOVERY_APPROVED" in s and "A5_AUTHORIZED" in s and b"\r\n" not in open(ROOT + "/scripts/server_recovery.sh", "rb").read()
    for bad in ("rm -", "rm  ", "shutdown", "runpod", "truncate", "--force", "> \"$d", "mv "):
        assert bad not in s.lower(), bad
    r = subprocess.run(["bash", ROOT + "/scripts/server_recovery.sh", "run"], capture_output=True, text=True, env={k: v for k, v in os.environ.items() if not k.startswith(("PT_", "A5_"))})
    assert r.returncode == 1 and "REFUSED" in r.stdout


def test_rollout_saving_changes_no_update_sampling_or_output(tmp_path, monkeypatch):
    """Same run with rollouts saved every step vs never: identical prompts, rewards, losses, gradients, update sizes and final weights."""
    res = []
    for every in ("1", "null"):
        d = tmp_path / every; d.mkdir()
        G, seen, pool = _grpo_env(d, monkeypatch)
        monkeypatch.setattr(sys, "argv", ["train_grpo", "--config", ROOT + "/configs/grpo/A_base_s2026.yaml", "--set", "device=cpu", "engine.name=hf", "eval.sets={}",
                                          "eval.at_start=false", f"data.path={pool}", "init.revision=null", "grpo.steps=4", "grpo.prompts_per_step=2", "grpo.group_size=2",
                                          "grpo.update_batch=4", "grpo.max_tokens=6", "grpo.sync_check_every=1", "train.micro_batch=2", "train.lr=1e-2",
                                          "train.save_every=null", "eval.every=null", f"grpo.save_rollouts_every={every}"])
        G.main()
        import random as _r
        import numpy as _np
        rng = (_r.random(), float(_np.random.random()), float(torch.rand(1)))  # draws AFTER the run: equal only if both runs consumed the same random numbers
        run = d / "runs" / "grpo" / "A_base_s2026"
        keep = lambda r: {k: v for k, v in r.items() if not (k.endswith("_s") or k.endswith("per_s") or "peak" in k)}
        w = glob.glob(str(run / "checkpoint" / "*.safetensors"))[0]
        res.append(([keep(r) for r in _rows(str(run))], sha_file(w), [c["sd"] for c in seen["calls"]], sorted(os.listdir(run)), rng))
    (r1, w1, s1, f1, g1), (r2, w2, s2, f2, g2) = res
    assert g1 == g2                                                                     # Python, NumPy and PyTorch random streams untouched by saving
    assert len(r1) == 4 and r1 == r2 and w1 == w2                                      # every logged number (rewards, losses, gradients, update sizes, gaps), final weights
    assert len(s1) == 4 and all(torch.equal(a[k], b[k]) for a, b in zip(s1, s2) for k in a)   # model state before every update
    assert "rollouts" in f1 and "rollouts" not in f2 and len(os.listdir(tmp_path / "1" / "runs" / "grpo" / "A_base_s2026" / "rollouts")) == 4


def test_nonfinite_update_saves_the_batch_and_makes_no_update(tmp_path, monkeypatch):
    src, state = failed_source(tmp_path, monkeypatch)
    G, seen, _ = _grpo_env(tmp_path, monkeypatch)
    monkeypatch.setattr(G, "score", lambda t, g: dict(reward=float("nan"), answer_reward=0.0, format_reward=1.0, error=None, rule=None))
    with pytest.raises(RuntimeError, match="non-finite"):
        _fork(monkeypatch, _spec(tmp_path, src, state, "N", {"grpo.save_rollouts_every": 1}))
    d = tmp_path / "runs" / "recovery" / "N"
    f = json.load(open(d / "FAILED.json")); assert f["attempted_step"] == 5 and f["completed_updates"] == 4
    m = json.load(open(d / "failed_batch_step5" / "meta.json")); assert m["stage"] == "update" and m["optimizer_updated"] is False
    assert len(jload(str(d / "failed_batch_step5" / "rollouts.jsonl"))) == 4 and _rows(str(d)) == [] and not (d / "rollouts").exists()
    assert [c["opt_step"] for c in seen["calls"]] == [4.0] and not (d / "state.pt").exists()   # one attempt on the restored state, never stepped


def test_registered_criteria_are_evaluated_as_written():
    sys.path.insert(0, ROOT + "/scripts")
    from recovery_criteria import evaluate
    crit = yaml.safe_load(open(ROOT + "/recovery/diagnostic_criteria.yaml"))
    c = crit["criteria"]
    assert c["early_warning"]["window"] == 20 and c["early_warning"]["threshold"] == 3.0
    assert c["early_warning"]["metrics"] == ["rollout_logprob_mad", "rollout_logprob_p99"] and c["early_warning"]["both_required"] is True
    assert c["severe_reward_deterioration"]["below"] == 0.30 and c["sustained_reward_deterioration"]["consecutive"] == 2 and c["numerical_failure"]["above"] == 0.05
    assert c["successful_completion"]["steps"] == 100 and "retrospectively" in crit["caveat"]
    noise = [0.0035, 0.0036, 0.0034, 0.0037, 0.0033]
    row = lambda s, mad, p99, rw: dict(step=s, rollout_logprob_mad=mad, rollout_logprob_p99=p99, reward_mean=rw)
    hist = [row(s, noise[s % 5], 0.1 + noise[s % 5], 0.7) for s in range(1, 81)]
    e = evaluate(hist, [row(s, noise[s % 5], 0.1 + noise[s % 5], 0.7) for s in range(81, 101)], crit, done={"steps": 100})
    assert e["early_warning"]["steps"] == [] and e["successful_completion"]["hit"] is True and not e["severe_reward_deterioration"]["hit"]
    # step 90: only the mean gap rises (no warning); 91: both rise (warning); 92-93: reward below 0.30; then a gate stop
    br = [row(s, noise[s % 5], 0.1 + noise[s % 5], 0.7) for s in range(81, 90)] + [row(90, 0.006, 0.1 + noise[0], 0.7), row(91, 0.006, 0.16, 0.7),
                                                                                    row(92, 0.04, 0.6, 0.05), row(93, 0.03, 0.5, 0.2)]
    e = evaluate(hist, br, crit, failed={"error": "rollout log-prob check failed: 0.08 exceeds 0.05", "attempted_step": 94})
    assert e["early_warning"]["first"] == 91 and 90 not in e["early_warning"]["steps"]
    assert e["severe_reward_deterioration"]["steps"] == [92, 93] and e["sustained_reward_deterioration"]["hit"] is True
    assert e["numerical_failure"] == dict(hit=True, gate_stop=True, attempted_step=94) and e["successful_completion"]["hit"] is False
    # exact z against a hand computation: window = the 20 steps BEFORE t (61-80 for t = 81), current step excluded
    import random as _r
    g = _r.Random(7)
    h2 = [row(s, 0.003 + 0.001 * g.random(), 0.1 + 0.02 * g.random(), 0.7) for s in range(1, 81)]
    b2 = [row(81, 0.0042, 0.13, 0.7)]
    prev = [r["rollout_logprob_mad"] for r in h2[60:80]]
    med = sorted(prev)[9:11]; med = (med[0] + med[1]) / 2
    dev = sorted(abs(x - med) for x in prev)[9:11]; mad = (dev[0] + dev[1]) / 2
    want = (0.0042 - med) / (1.4826 * mad)
    assert abs(evaluate(h2, b2, crit)["early_warning"]["z"][81]["rollout_logprob_mad"] - round(want, 2)) < 1e-9
    e = evaluate(hist, br[:-1], crit, failed={"error": "rollout log-prob check failed", "attempted_step": 93})  # one low step, then the gate
    assert str(e["sustained_reward_deterioration"]["hit"]).startswith("not observable")
    sh = open(ROOT + "/scripts/server_recovery.sh", encoding="utf-8").read()
    assert "reward" not in sh.lower() and "recovery_criteria" not in sh and "reward_mean" not in open(ROOT + "/posttrain/train_grpo.py", encoding="utf-8").read().split("def train(")[1].split("while step")[0]
