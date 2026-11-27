"""Local CPU tests. Run: python -m pytest tests -q   (set PT_TOKENIZER to a local Qwen2.5 snapshot dir)"""
import copy, glob, json, math, os, random, subprocess, sys
import pytest, torch, yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from posttrain import losses as L  # noqa: E402
from posttrain.common import Run, jload, load_config, sha_file  # noqa: E402
from posttrain.diagnostics import DifficultySampler, ParamProbe, group_stats  # noqa: E402
from posttrain.batching import microbatches, padded_tokens, seq_len  # noqa: E402
from posttrain.engines import Completion, HFEngine, compare_engine_state, expected_engine_state, verified_sync  # noqa: E402
from posttrain.evaluate import category, check_lock  # noqa: E402
from posttrain.model_io import (load_policy, load_state, make_optimizer, read_lineage, resolve_parent, save_model, save_state)  # noqa: E402
from posttrain.stats import holm, mcnemar_exact, paired  # noqa: E402
from posttrain.train_ei import select  # noqa: E402
from posttrain.train_grpo import check_rollouts, chunks, grpo_update, logprobs_nograd, place, sampling_mismatch  # noqa: E402
from posttrain.train_sft import finalize, guard_training_data, sft_step  # noqa: E402
from posttrain.verify import clean_format, score  # noqa: E402

D = os.path.join(ROOT, "data", "stage0")


@pytest.fixture(autouse=True)
def no_server_settings(monkeypatch):
    """Tests build their own tiny models and gates, so settings from server_env.sh must not leak in (setup_server.sh runs the tests with them set)."""
    for k in ("PT_BASE_SHA256", "PT_HW", "PT_PLATFORM", "A5_AUTHORIZED", "ALLOW_FINAL_EVAL"):
        monkeypatch.delenv(k, raising=False)


class CharTok:
    """Minimal byte-level tokenizer so tests need no download."""
    pad_token_id, eos_token_id, padding_side = 0, 1, "right"

    def __call__(self, t, add_special_tokens=False, **k):
        return {"input_ids": [2 + (ord(c) % 60) for c in t]}


def tiny(seed=0, layers=2):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=layers, num_attention_heads=4,
                                        num_key_value_heads=2, max_position_embeddings=256, tie_word_embeddings=True))


CFG = dict(train=dict(micro_batch=2, max_grad_norm=1e9, append_eos=True, lr=1e-3, max_len=200),
           grpo=dict(loss="grpo_clip", clip=0.2, length_norm="const", norm_const=64, epochs=1, update_batch=8, group_filter="none"))


# ---------- losses ----------
def test_mask_alignment_and_eos():
    t = CharTok(); b = L.encode(t, ["abc", "a"], ["xy", "wxyz"], append_eos=True)
    for i, (p, r) in enumerate((("abc", "xy"), ("a", "wxyz"))):
        got = b["labels"][i][b["response_mask"][i].bool()].tolist()
        assert got == t(r)["input_ids"] + [1]
    assert b["response_mask"].sum(1).tolist() == [3, 5]


def test_encode_refuses_truncation():
    with pytest.raises(ValueError):
        L.encode(CharTok(), ["a" * 10], ["b" * 10], max_len=15)


def test_token_stats_match_manual_and_padding_invariant():
    m = tiny().eval(); t = CharTok()
    one = L.encode(t, ["abc"], ["xy"]); two = L.encode(t, ["abc", "abcdefgh"], ["xy", "pqrstuv"])
    lp1, e1 = L.token_stats(m, one, True); lp2, _ = L.token_stats(m, two)
    a = lp1[0][one["response_mask"][0].bool()]; b = lp2[0][two["response_mask"][0].bool()]
    assert torch.allclose(a, b, atol=1e-5)
    z = torch.log_softmax(m(one["input_ids"]).logits.float(), -1)
    assert torch.allclose(lp1, z.gather(-1, one["labels"][..., None])[..., 0], atol=1e-5)
    assert torch.allclose(e1, -(z.exp() * z).sum(-1), atol=1e-5)


def test_sft_gradient_independent_of_micro_batch():
    t = CharTok(); ex = [("ab", "cdefg"), ("abc", "x"), ("a", "bcdefghij"), ("abcd", "zz")]; gr = []
    for mb in (1, 2, 4):
        m = tiny(); c = copy.deepcopy(CFG); c["train"]["micro_batch"] = mb
        opt = torch.optim.SGD(m.parameters(), lr=1.0); before = [p.detach().clone() for p in m.parameters()]
        sft_step(m, opt, t, ex, c, "cpu"); gr.append([a - p.detach() for a, p in zip(before, m.parameters())])
    for g in gr[1:]:
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(gr[0], g))


def test_group_advantages():
    a = L.group_advantages([1, 0, 0, 0, 1, 1, 1, 1], 4)
    assert torch.allclose(a, torch.tensor([.75, -.25, -.25, -.25, 0, 0, 0, 0]))
    s = L.group_advantages([1, 0, 0, 0, 1, 1, 1, 1], 4, std_norm=True)
    assert torch.isfinite(s).all() and s[4:].abs().max() == 0
    with pytest.raises(ValueError):
        L.group_advantages([1, 0, 1], 2)


def test_clip_loss_hand_values():
    old = torch.zeros(1, 2); mask = torch.ones(1, 2); lp = torch.log(torch.tensor([[1.5, 0.5]]))
    s, m = L.policy_loss("grpo_clip", lp, mask, torch.tensor([1.0]), old_lp=old, clip=0.2, length_norm="mean")
    assert math.isclose(s.item(), -(1.2 + 0.5) / 2, rel_tol=1e-6) and m["clip_active_tokens"] == 1 and m["clip_outside_tokens"] == 2
    s, _ = L.policy_loss("grpo_clip", lp, mask, torch.tensor([-1.0]), old_lp=old, clip=0.2, length_norm="mean")
    assert math.isclose(s.item(), (1.5 + 0.8) / 2, rel_tol=1e-6)


def test_ratio_one_gradient_equals_reinforce():
    """At ratio 1 (a single on-policy update) GRPO-clip and REINFORCE-with-baseline have identical gradients."""
    lp = torch.randn(3, 5, requires_grad=True); mask = torch.ones(3, 5); adv = torch.tensor([.5, -.5, .2])
    ga = torch.autograd.grad(L.policy_loss("baseline", lp, mask, adv)[0].sum(), lp)[0]
    gb = torch.autograd.grad(L.policy_loss("grpo_clip", lp, mask, adv, old_lp=lp.detach())[0].sum(), lp)[0]
    assert torch.allclose(ga, gb)


def test_length_norm_modes():
    x = torch.ones(2, 4); m = torch.tensor([[1., 1, 0, 0], [1, 1, 1, 1]])
    assert L.reduce_seq(x, m, "mean", 8).tolist() == [1, 1] and L.reduce_seq(x, m, "const", 8).tolist() == [.25, .5]


# ---------- GRPO update ----------
def rollouts():
    P = ["ab", "ab", "ab", "ab", "cd", "cd", "cd", "cd"]; R = [1., 0, 0, 1, 0, 0, 0, 0]
    rng = random.Random(0)
    C = [Completion("", [rng.randrange(2, 60) for _ in range(rng.randrange(2, 9))], "stop") for _ in P]
    return P, C, torch.tensor(R)


def step_delta(filt, mb=2, epochs=1, ub=8, kind="grpo_clip", tb=None):
    m = tiny(); c = copy.deepcopy(CFG); c["grpo"].update(group_filter=filt, epochs=epochs, update_batch=ub, loss=kind); c["train"]["micro_batch"] = mb
    if tb: c["train"]["token_budget"] = tb
    P, C, R = rollouts(); adv = L.group_advantages(R, 4); old = logprobs_nograd(m, CharTok(), P, C, "cpu", 3, 200)
    opt = torch.optim.SGD(m.parameters(), lr=1.0); b = [p.detach().clone() for p in m.parameters()]
    d = grpo_update(m, opt, CharTok(), P, C, adv, R, old, c, "cpu", 1)
    return [x - p.detach() for x, p in zip(b, m.parameters())], d


def test_compute_only_filter_keeps_gradient_and_saves_work():
    a, da = step_delta("none"); b, db = step_delta("compute_only")
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(a, b))
    assert da["trained_rollouts"] == 8 and db["trained_rollouts"] == 4


def test_renormalize_filter_scales_gradient():
    a, _ = step_delta("none"); b, _ = step_delta("renormalize")
    assert all(torch.allclose(2 * x, y, atol=1e-5) for x, y in zip(a, b))


def test_grpo_micro_batch_invariant_and_ratio_one():
    a, da = step_delta("none", 1); b, _ = step_delta("none", 8)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(a, b))
    assert abs(da["ratio_sum"] / da["tok_sum"] - 1) < 1e-4 and da["updates"] == 1


def test_off_policy_update_count():
    _, d = step_delta("none", epochs=2, ub=4)
    assert d["updates"] == 4


def test_update_moves_probability_in_advantage_direction():
    m = tiny(); P, C, R = rollouts(); adv = L.group_advantages(R, 4); t = CharTok()
    before = logprobs_nograd(m, t, P, C, "cpu", 4, 200); c = copy.deepcopy(CFG)
    opt = torch.optim.SGD(m.parameters(), lr=0.5); grpo_update(m, opt, t, P, C, adv, R, before, c, "cpu", 0)
    after = logprobs_nograd(m, t, P, C, "cpu", 4, 200)
    assert sum(float(a) * (y.sum() - x.sum()).item() for a, x, y in zip(adv, before, after)) > 0


def test_all_zero_advantage_skips_update():
    m = tiny(); P, C, _ = rollouts(); c = copy.deepcopy(CFG); c["grpo"]["group_filter"] = "compute_only"
    z = torch.zeros(8); opt = torch.optim.SGD(m.parameters(), lr=1.0); b = [p.detach().clone() for p in m.parameters()]
    d = grpo_update(m, opt, CharTok(), P, C, z, z, logprobs_nograd(m, CharTok(), P, C, "cpu", 4, 200), c, "cpu", 0)
    assert d["updates"] == 0 and all(torch.equal(x, p) for x, p in zip(b, m.parameters()))


def test_place_roundtrip():
    mask = torch.tensor([[0., 1, 1, 0], [0, 0, 1, 1]])
    assert place([torch.tensor([1., 2]), torch.tensor([3., 4])], mask).tolist() == [[0, 1, 2, 0], [0, 0, 3, 4]]


# ---------- token-budget micro-batching ----------
def _lens(k=40, seed=0):
    r = random.Random(seed)
    return [r.randrange(3, 60) for _ in range(k)]


def test_microbatches_legacy_is_fixed_size_chunks():
    order = list(range(11)); random.Random(1).shuffle(order); L_ = _lens(11)
    assert microbatches(order, None, 4) == chunks(order, 4)


def test_microbatches_partition_and_budget():
    L_ = _lens(40); order = list(range(40))
    for tb, mb in ((150, None), (150, 4), (60, None), (999, 3)):
        parts = microbatches(order, L_, mb, tb)
        flat = [i for p in parts for i in p]
        assert sorted(flat) == order                                              # every item exactly once
        assert all(len(p) <= (mb or 10**9) for p in parts)
        pad = padded_tokens(parts, L_)
        for p, n in zip(parts, pad):
            assert n <= tb or len(p) == 1, (tb, p)                                # only a lone item may exceed the budget
        assert flat == sorted(flat, key=lambda i: (L_[i], i))                     # length-sorted


def test_microbatches_rejects_bad_settings():
    with pytest.raises(ValueError):
        microbatches([0, 1], [1, 1])
    with pytest.raises(ValueError):
        microbatches([0], [1], 2, 0)


def test_grpo_gradient_independent_of_token_budget():
    a, _ = step_delta("none", mb=2)
    for tb in (12, 20, 33, 999):
        b, d = step_delta("none", mb=4, tb=tb)
        assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(a, b)), tb
        assert d["trained_rollouts"] == 8
    c, _ = step_delta("renormalize", mb=2); e, _ = step_delta("renormalize", mb=8, tb=33)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(c, e))


def test_sft_gradient_independent_of_token_budget():
    t = CharTok(); ex = [("ab", "cdefg"), ("abc", "x"), ("a", "bcdefghij"), ("abcd", "zz"), ("ab", "q" * 30)]; gr = []
    for tb, mb in ((None, 2), (40, 8), (20, 3), (None, 8)):
        m = tiny(); c = copy.deepcopy(CFG); c["train"]["micro_batch"] = mb; c["train"]["token_budget"] = tb
        opt = torch.optim.SGD(m.parameters(), lr=1.0); b = [p.detach().clone() for p in m.parameters()]
        sft_step(m, opt, t, ex, c, "cpu"); gr.append([x - p.detach() for x, p in zip(b, m.parameters())])
    for g in gr[1:]:
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(gr[0], g))


def test_old_policy_logprobs_independent_of_token_budget():
    m = tiny().eval(); P, C, _ = rollouts(); t = CharTok()
    a = logprobs_nograd(m, t, P, C, "cpu", 3, 200); b = logprobs_nograd(m, t, P, C, "cpu", 8, 200, token_budget=25)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(a, b))


def test_seq_len_matches_encode():
    t = CharTok(); b = L.encode(t, ["abcde"], [[7, 8, 9]], append_eos=True)
    assert seq_len(t, "abcde", [7, 8, 9], append_eos=True) == int(b["attention_mask"].sum()) + 1


# ---------- weight synchronisation ----------
def test_engine_state_map_and_exact_comparison():
    m = tiny(layers=2); sd = m.state_dict(); exp = expected_engine_state(sd)
    assert "model.layers.0.self_attn.qkv_proj.weight" in exp and "model.layers.1.mlp.gate_up_proj.weight" in exp
    assert not any(k.endswith((".q_proj.weight", ".up_proj.weight", "lm_head.weight")) for k in exp)
    assert exp["model.layers.0.self_attn.qkv_proj.weight"].shape[0] == sum(
        sd[f"model.layers.0.self_attn.{x}_proj.weight"].shape[0] for x in "qkv")
    good = {k: v.clone() for k, v in exp.items()}
    assert compare_engine_state(exp, good)["params_mismatched"] == 0
    stale = {k: v.clone() for k, v in exp.items()}                                # a layer that was never updated
    n = stale["model.layers.0.self_attn.o_proj.weight"]; stale["model.layers.0.self_attn.o_proj.weight"] = n + 0.01 * n.abs().max()
    assert compare_engine_state(exp, stale)["params_mismatched"] == 1
    swapped = {k: v.clone() for k, v in exp.items()}                              # k and v rows swapped inside the merged projection
    w = swapped["model.layers.0.self_attn.qkv_proj.weight"]; q = sd["model.layers.0.self_attn.q_proj.weight"].shape[0]
    kr = sd["model.layers.0.self_attn.k_proj.weight"].shape[0]
    kv = torch.cat([w[q + kr:q + kr + kr], w[q:q + kr]], 0) if w.shape[0] - q == 2 * kr else None
    if kv is not None and not torch.equal(kv, w[q:]):
        swapped["model.layers.0.self_attn.qkv_proj.weight"] = torch.cat([w[:q], kv], 0)
        assert compare_engine_state(exp, swapped)["params_mismatched"] == 1
    missing = {k: v for k, v in exp.items() if "gate_up" not in k}
    r = compare_engine_state(exp, missing); assert r["missing"] == 2 and r["params_mismatched"] == 2


def test_engine_state_ignores_only_sub_bf16_differences():
    """A fp32 change below bf16 resolution cannot be seen by the engine; a change above it must be."""
    m = tiny(); sd = m.state_dict(); exp = expected_engine_state(sd)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(p.abs().mean() * 1e-7)
    assert compare_engine_state(expected_engine_state(m.state_dict()), exp)["params_mismatched"] == 0
    with torch.no_grad():
        for p in m.parameters():
            p.add_(p.abs().mean() * 0.1)
    assert compare_engine_state(expected_engine_state(m.state_dict()), exp)["params_mismatched"] > 0


def test_tied_lm_head_is_not_expected_in_the_engine():
    m = tiny(); sd = dict(m.state_dict())
    assert "lm_head.weight" in sd and "lm_head.weight" not in expected_engine_state(sd)
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"] * 1.5                   # an untied head must be checked
    assert "lm_head.weight" in expected_engine_state(sd)


def test_sampling_mismatch_detects_misalignment():
    c = [Completion("", [1, 2, 3], "stop", [-1.0, -2.0])]
    with pytest.raises(RuntimeError):
        sampling_mismatch(c, [torch.tensor([-1.0, -2.0, -3.0])])


# ---------- diagnostics ----------
def test_group_stats():
    s = group_stats([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0], 4)
    assert (s["all_correct"], s["all_incorrect"], s["zero_adv"], round(s["informative_rate"], 3)) == (1, 1, 2, .333)


def test_difficulty_sampler_prefers_informative():
    s = DifficultySampler(range(100), random.Random(0), True); s.update(range(50), [1.0] * 50); s.update(range(50, 100), [0.5] * 50)
    n = sum(i >= 50 for _ in range(50) for i in s.sample(10))
    assert n > 400 and len(set(s.sample(100))) == 100


# ---------- verifier ----------
def test_verifier_contract():
    w = lambda x: f" r </think> <answer>{x}</answer>"
    assert score(w("1/2"), "\\frac{1}{2}")["reward"] == 1 and score(w("2"), "\\frac{1}{2}")["reward"] == 0
    assert score(" r</think><answer>4</answer>", "4")["format_reward"] == 0
    assert score(w("4"), None)["reward"] == 0  # never raises
    assert clean_format(w("4")) and not clean_format(w("4") + w("5")) and not clean_format("</answer></think><answer>")


from posttrain import verifier_lock, verify as verify_mod  # noqa: E402


@pytest.mark.parametrize("ans,gold,want", verifier_lock.CASES)
def test_verifier_golden_cases(ans, gold, want):
    assert score(f"r </think> <answer>{ans}</answer>", gold)["reward"] == want


def test_verifier_is_frozen_and_a_change_is_refused(monkeypatch, tmp_path):
    assert verify_mod.require_frozen()["version"] == "2.0.0"
    monkeypatch.setitem(verifier_lock.SHA256, "grader_py", "0" * 64)
    with pytest.raises(SystemExit):
        verify_mod.require_frozen()
    from posttrain.common import Run
    with pytest.raises(SystemExit):
        Run(dict(run_dir=str(tmp_path / "r")), "sft")


def test_verifier_v2_only_adds_to_the_strict_grader():
    from posttrain.verifiers.drgrpo_grader import r1_zero_reward_fn
    for ans, gold, _ in verifier_lock.CASES:
        x = f"r </think> <answer>{ans}</answer>"
        assert score(x, gold)["reward"] >= r1_zero_reward_fn(x, gold)["reward"]
    s = score("Answer is 2.", "2")  # no tags: the format rule is untouched
    assert s["format_reward"] == s["reward"] == 0 and s["rule"] is None


def test_verifier_factorial_agrees_with_independent_methods():
    import math
    import sympy
    from fractions import Fraction
    from functools import reduce
    for n in range(0, 21):
        prod = reduce(lambda a, b: a * b, range(1, n + 1), Fraction(1))          # exact product, no library factorial
        assert prod == math.factorial(n) == int(sympy.factorial(n)) and (n == 0 or sympy.gamma(n + 1) == prod)
        assert abs(math.lgamma(n + 1) - math.log(float(prod))) < 1e-9 * max(1, math.lgamma(n + 1))  # numerical
        w = lambda g: score(f"r </think> <answer>${n}!$</answer>", g)
        assert w(str(prod))["reward"] == 1 and w(f"{int(prod):,}")["reward"] == 1 and w(str(prod + 1))["reward"] == 0
    assert int(sympy.factorial(8)) == 40320 and sympy.Rational(40320, 9) == 4480


def test_verifier_question_only_uses_the_same_rules():
    q = lambda b, g: score("so \\boxed{" + b + "}", g, "question_only")["reward"]
    assert q("8!", "40,\\!320") == 1 and q("Answer is 2", "2") == 1 and q("9!/9", "40320") == 0 and score("Answer is 2", "2", "question_only")["reward"] == 0


def _done(d, correct, lr, sha="x", n=1000):
    os.makedirs(d)
    json.dump(dict(eval=dict(gsm8k_dev=dict(correct=correct, n=n, accuracy=correct / n, data_sha256=sha)), lineage=dict(chain=["base", "sft"])), open(d + "/DONE.json", "w"))
    json.dump(dict(config=dict(train=dict(lr=lr)), code_sha256="c"), open(d + "/run.json", "w"))


@pytest.mark.parametrize("c,want", [((612, 619, 622), "1.0e-5"), ((600, 615, 622), "2.0e-5"), ((600, 610, 622), "5.0e-5"), ((622, 622, 622), "1.0e-5"), ((611, 600, 622), "5.0e-5")])
def test_choose_lr_rule(tmp_path, c, want):
    for r, k in zip(("1e-5", "2e-5", "5e-5"), c):
        _done(f"{tmp_path}/sft/tune_lr{r}", k, r)
    run = lambda *x: subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "choose_lr.py"), "sft", *x], env=dict(os.environ, PT_RUNS=str(tmp_path)), capture_output=True, text=True)
    assert run("--value").stdout.strip() == want
    json.dump(dict(config=dict(train=dict(lr=9e-5))), open(f"{tmp_path}/sft/tune_lr1e-5/run.json", "w"))
    assert run("--value").returncode != 0
    os.remove(f"{tmp_path}/sft/tune_lr5e-5/DONE.json")
    assert run("--value").returncode != 0


def test_readiness_separates_sft_from_grpo(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import readiness as rd
    R = str(tmp_path)
    assert rd.problems("sft", R) and rd.problems("grpo", R)
    g = {k: True for k in rd.COMMON + rd.GRPO}
    os.makedirs(R + "/validation")
    for d in ("sft", "grpo_ref", "grpo_from_sft"):
        os.makedirs(f"{R}/validation/{d}")
        json.dump(dict(code_sha256=rd.VALIDATED), open(f"{R}/validation/{d}/run.json", "w"))
    put = lambda **k: json.dump(dict(gates=dict(g, **k)), open(R + "/validation/summary.json", "w"))
    put()
    assert rd.problems("sft", R) == [] and any("Session A" in x for x in rd.problems("grpo", R))
    put(rollout_logprobs_agree=False)
    assert rd.problems("sft", R) == [] and any("rollout" in x for x in rd.problems("grpo", R))
    put(sync=False)
    assert rd.problems("sft", R)
    put()
    for i in rd.A:
        if i.startswith("SFT"):
            _done(f"{R}/sft/{i[4:]}", 600, 2e-5)
    assert rd.problems("grpo", R) == []
    json.dump(dict(code_sha256="other"), open(R + "/validation/grpo_ref/run.json", "w"))
    assert rd.problems("grpo", R) and rd.problems("sft", R) == []


def test_category_taxonomy():
    f = lambda t, fin="stop": category(t, score(t, "4"), "4", fin)
    assert f("</think> <answer>4</answer>") == "correct" and f("</think> <answer>5</answer>") == "format_ok_answer_wrong"
    assert f("so \\boxed{4}") == "format_fail_answer_right" and f("so 7", "length") == "format_fail_truncated"


def test_ei_selection_caps_dedupes_and_filters():
    good = Completion(" a </think> <answer>4</answer>", [1, 2]); bad = Completion(" a </think> <answer>5</answer>", [1])
    g2 = Completion(" b </think> <answer>4</answer>", [3]); g3 = Completion(" c </think> <answer>4</answer>", [4])
    keep, sc = select([dict(id="p", answer="4")], [[good, good, bad, g2, g3]], dict(require_clean_format=True, max_per_problem=2), random.Random(0))
    assert len(keep) == 2 and len(sc) == 5 and all("4</answer>" in c.text for _, c in keep)


# ---------- checkpoints, lineage, resume ----------
class StubTok:
    def save_pretrained(self, p):
        open(os.path.join(p, "tok.txt"), "w").write("x")


META = dict(config_sha256="c", code_sha256="k")


def test_checkpoint_lineage_and_tamper_detection(tmp_path):
    m = tiny(); base = str(tmp_path / "base"); m.save_pretrained(base)
    p, parent = resolve_parent(dict(init=dict(path=base, expect_stage="base")))
    ck = str(tmp_path / "sft"); save_model(m, StubTok(), ck, "sft", parent, META)
    lin = read_lineage(ck); assert lin["chain"] == ["base", "sft"] and lin["parent"]["weights_sha256"] == parent["weights_sha256"]
    m2 = load_policy(ck, "cpu", "fp32")
    for (k, a), (_, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert torch.allclose(a, b, atol=2e-2, rtol=1e-2), k
    with pytest.raises(SystemExit):
        save_model(m, StubTok(), ck, "sft", parent, META)
    with pytest.raises(SystemExit):
        resolve_parent(dict(init=dict(path=ck, expect_stage="base")))
    resolve_parent(dict(init=dict(path=ck, expect_stage="sft")))
    f = glob.glob(ck + "/*.safetensors")[0]
    with open(f, "r+b") as h:
        h.seek(-1, 2); x = h.read(1); h.seek(-1, 2); h.write(bytes([x[0] ^ 1]))
    with pytest.raises(SystemExit):
        resolve_parent(dict(init=dict(path=ck)))


def test_resume_is_bit_exact(tmp_path):
    t = CharTok(); ex = [("ab", "cdefg"), ("abc", "xyz"), ("a", "bcd"), ("abcd", "zz")]
    def run(m, o, k):
        for _ in range(k):
            sft_step(m, o, t, ex, CFG, "cpu")
    a = tiny(); oa = make_optimizer(a, dict(lr=1e-3)); run(a, oa, 4)
    b = tiny(); ob = make_optimizer(b, dict(lr=1e-3)); run(b, ob, 2); save_state(str(tmp_path / "s.pt"), b, ob, 2, {})
    c = tiny(5); oc = make_optimizer(c, dict(lr=1e-3)); st, _ = load_state(str(tmp_path / "s.pt"), c, oc); run(c, oc, 2)
    assert st == 2 and all(torch.equal(x, y) for x, y in zip(a.parameters(), c.parameters()))


def test_run_dir_protection(tmp_path):
    cfg = dict(run_dir=str(tmp_path / "r"), a=1); r = Run(cfg, "x"); r.log(dict(event="t"))
    with pytest.raises(SystemExit):
        Run(cfg, "x")
    Run(cfg, "x", resume=True)
    with pytest.raises(SystemExit):
        Run(dict(cfg, a=2), "x", resume=True)
    r.finish()
    with pytest.raises(SystemExit):
        Run(cfg, "x", resume=True)


def test_final_eval_is_locked(monkeypatch):
    monkeypatch.delenv("ALLOW_FINAL_EVAL", raising=False)
    with pytest.raises(SystemExit):
        check_lock({"g": "x/eval_gsm8k_test_final.jsonl"})
    check_lock({"g": "x/eval_gsm8k_dev_1000.jsonl"})


def test_param_probe_detects_movement_and_bf16_loss():
    m = tiny(); p = ParamProbe(m); p.before(); r = p.after()
    assert r["param_changed_frac"] == 0
    p.before()
    with torch.no_grad():
        for x in m.parameters():
            x.add_(1e-6)
    r = p.after()
    assert r["param_changed_frac"] > 0.9 and r["param_changed_frac_bf16"] < 0.2  # visible in fp32, mostly lost in bf16


def test_sft_step_reports_movement():
    m = tiny(); r = sft_step(m, make_optimizer(m, dict(lr=1e-3)), CharTok(), [("ab", "cdefg")], CFG, "cpu", ParamProbe(m))
    assert r["param_changed_frac"] > 0.5 and math.isfinite(r["loss"])


def test_cannot_train_on_eval_files():
    for f in ("eval_gsm8k_dev_1000.jsonl", "eval_math_val_final.jsonl"):
        with pytest.raises(SystemExit):
            guard_training_data("/x/" + f)
    guard_training_data("/x/rl_train_pool.jsonl")


def test_finalize_is_idempotent_after_crash(tmp_path):
    from posttrain.model_io import weights_hash
    m = tiny(); base = str(tmp_path / "b"); m.save_pretrained(base); _, par = resolve_parent(dict(init=dict(path=base)))
    run = type("R", (), dict(dir=str(tmp_path), meta=META))()
    a = finalize(run, m, StubTok(), "sft", par, dict(steps=3)); b = finalize(run, m, StubTok(), "sft", par, dict(steps=3))
    assert a["weights_sha256"] == b["weights_sha256"] == weights_hash(str(tmp_path / "checkpoint"))
    with pytest.raises(SystemExit):
        finalize(run, m, StubTok(), "sft", par, dict(steps=4))


def test_sampling_mismatch():
    c = [Completion("", [1, 2], "stop", [-1.0, -2.0]), Completion("", [3], "stop", [])]
    r = sampling_mismatch(c, [torch.tensor([-1.1, -2.0]), torch.tensor([-5.0])])
    assert math.isclose(r["rollout_logprob_mad"], 0.05, rel_tol=1e-4) and sampling_mismatch(c[1:], [torch.tensor([-5.0])]) == {}


def test_placement_matches_vllm_072(monkeypatch):
    """vLLM 0.7.2 in-process always uses cuda:0; every GPU config must put the policy on cuda:1."""
    monkeypatch.setenv("PT_DATA", D); monkeypatch.setenv("PT_RUNS", "/tmp/pt_runs")
    for f in glob.glob(ROOT + "/configs/**/*.yaml", recursive=True):
        c = load_config(f)
        assert c["engine"]["device"] == "cuda:0" and c.get("device", "cuda:1") == "cuda:1", f


# ---------- statistics ----------
def test_stats():
    assert math.isclose(mcnemar_exact(18, 27), 0.232693, abs_tol=1e-6)
    r = paired({1: True, 2: True, 3: False}, {1: True, 2: False, 3: True}); assert r["diff"] == 0 and r["a_only"] == r["b_only"] == 1
    with pytest.raises(ValueError):
        paired({1: True}, {2: True})
    h = holm(dict(a=.01, b=.04, c=.03)); assert math.isclose(h["a"], .03) and math.isclose(h["c"], .06) and math.isclose(h["b"], .06)


# ---------- data ----------
need = pytest.mark.skipif(not os.path.exists(D + "/sft_all_labeled_4836.jsonl"), reason="stage0 data not built (the public export ships MANIFEST.json and id lists only; see docs/PUBLIC_EXPORT.md)")


@need
def test_manifest_hashes_and_counts():
    m = json.load(open(D + "/MANIFEST.json"))
    for f, h in m["output_sha256"].items():
        assert sha_file(f"{D}/{f}") == h, f
    assert m["counts"]["reverified"] == 4545 and m["counts"]["exact"] == 3496


@need
def test_no_train_eval_leak_and_nesting():
    ev = {r["question"].strip() for f in glob.glob(D + "/eval_*.jsonl") for r in jload(f)}
    rows = {r["id"]: r for r in jload(D + "/sft_all_labeled_4836.jsonl")}
    assert not ev & {r["question"].strip() for r in jload(D + "/rl_train_pool.jsonl")}
    S = {os.path.basename(f)[:-5]: json.load(open(f))["ids"] for f in glob.glob(D + "/subsets/*.json")}
    for n, ids in S.items():
        assert len(ids) == len(set(ids)), n
        assert not ev & {rows[i]["problem"].strip() for i in ids}, n
        if n.startswith("reverified"):
            assert all(rows[i]["reverified"] for i in ids)
        if n.startswith("exact"):
            assert all(rows[i]["exact_match"] for i in ids)
    for s in (0, 1, 2):
        for arm in ("exact", "reverified"):
            for a, b in zip((128, 256, 512, 1024), (256, 512, 1024, 2048)):
                assert S[f"{arm}_n{b}_s{s}"][:a] == S[f"{arm}_n{a}_s{s}"]


@need
def test_all_configs_resolve(monkeypatch):
    monkeypatch.setenv("PT_DATA", D); monkeypatch.setenv("PT_RUNS", "/tmp/pt_runs")
    fs = [f for f in glob.glob(ROOT + "/configs/**/*.yaml", recursive=True) if not os.path.basename(f).startswith("_")]
    assert len(fs) >= 10
    for f in fs:
        c = load_config(f); assert c["run_dir"] and c["seed"] is not None, f
        d = c.get("data") or {}
        for p in [d.get("path"), d.get("subset")] + list(((c.get("eval") or {}).get("sets") or c.get("sets") or {}).values()):
            if p:
                assert os.path.exists(p), (f, p)


@need
def test_registry_matches_configs():
    import csv
    rows = list(csv.DictReader(open(ROOT + "/EXPERIMENT_REGISTRY.csv", encoding="utf-8")))
    ids = [r["experiment_id"] for r in rows]
    assert len(ids) == len(set(ids)) and all(os.path.exists(os.path.join(ROOT, r["config"])) for r in rows if r["config"])
    assert all(r["status"] in ("PLANNED", "GATED", "HOLD_B", "DEFERRED", "HISTORICAL", "DONE_LOCAL", "DONE_PILOT", "BLOCKED") for r in rows)


# ---------- research server ----------
OLD = "/srv/research/legacy-workdir"


def test_a6000_settings_touch_memory_only(monkeypatch):
    from posttrain.hw import HW, hw_overrides
    monkeypatch.setenv("PT_DATA", D); monkeypatch.setenv("PT_RUNS", "/tmp/pt_runs")
    for mod, ov in HW["a6000"].items():
        assert all(o.split("=")[0] in ("train.micro_batch", "train.token_budget") for o in ov), mod
    for f, mod in (("sft/reverified_full_s0", "posttrain.train_sft"), ("grpo/B_sft512_s2026", "posttrain.train_grpo"), ("ei/ei_base", "posttrain.train_ei")):
        a = load_config(f"{ROOT}/configs/{f}.yaml"); b = load_config(f"{ROOT}/configs/{f}.yaml", hw_overrides(mod, "a6000"))
        assert a["train"]["micro_batch"] == 4 and b["train"]["micro_batch"] == 2
        for c in (a, b):
            c["train"].pop("micro_batch"); c["train"].pop("token_budget")
        assert a == b, f
    assert hw_overrides("posttrain.train_sft", None) == []
    with pytest.raises(SystemExit):
        hw_overrides("posttrain.train_sft", "a100_typo")


def test_a6000_micro_batch_2_keeps_grpo_gradient_loss_and_update_count():
    """Same 8 rollouts, same divisor: micro-batch 2 (with and without a token cap) equals 4 and the whole batch."""
    ref, d0 = step_delta("none", mb=4)
    for mb, tb in ((2, None), (2, 20), (2, 33), (8, None)):
        g, d = step_delta("none", mb=mb, tb=tb)
        assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(ref, g)), (mb, tb)
        assert abs(d["loss"] - d0["loss"]) < 1e-5 and d["updates"] == d0["updates"] == 1 and d["trained_rollouts"] == 8
        assert abs(d["grad_norms"][0] - d0["grad_norms"][0]) < 1e-4 * max(1.0, d0["grad_norms"][0]) and d["tok_sum"] == d0["tok_sum"]


def test_a6000_micro_batch_2_keeps_sft_gradient_and_loss():
    t = CharTok(); ex = [("ab", "cdefg"), ("abc", "x"), ("a", "bcdefghij"), ("abcd", "zz"), ("ab", "q" * 30), ("a", "b")]; out = []
    for mb in (4, 2, 6):
        m = tiny(); c = copy.deepcopy(CFG); c["train"]["micro_batch"] = mb
        opt = torch.optim.SGD(m.parameters(), lr=1.0); b = [p.detach().clone() for p in m.parameters()]
        r = sft_step(m, opt, t, ex, c, "cpu"); out.append(([x - p.detach() for x, p in zip(b, m.parameters())], r))
    for g, r in out[1:]:
        assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(out[0][0], g))
        assert abs(r["loss"] - out[0][1]["loss"]) < 1e-5 and r["tokens"] == out[0][1]["tokens"]


def test_a6000_token_cap_bounds_predicted_memory():
    """Worst padded micro-batch under the cap must fit 48 GB by the pilot's fit; plain micro-batch 2 with two 2,048-token rollouts would not leave 2 GiB."""
    peak = lambda pad: 25.43 + 5.10 * pad / 1000
    rng = random.Random(0); L_ = [rng.randrange(60, 2049) for _ in range(2000)] + [2048, 2048]
    parts = microbatches(list(range(len(L_))), L_, 2, 3300)
    assert sorted(i for p in parts for i in p) == list(range(len(L_))) and max(len(p) for p in parts) <= 2
    assert peak(max(padded_tokens(parts, L_))) < 42.5 and peak(2 * 2048) > 46


def test_unidentified_weights_are_refused(tmp_path, monkeypatch):
    m = tiny(); base = str(tmp_path / "b"); m.save_pretrained(base)
    real = sha_file(base + "/model.safetensors")
    monkeypatch.setenv("PT_BASE_SHA256", "0" * 64)
    with pytest.raises(SystemExit):
        resolve_parent(dict(init=dict(path=base, expect_stage="base")))  # e.g. an old SFT checkpoint passed as the base model
    monkeypatch.setenv("PT_BASE_SHA256", real)
    resolve_parent(dict(init=dict(path=base, expect_stage="base")))
    with pytest.raises(SystemExit):
        resolve_parent(dict(init=dict(path=base, expect_sha256="", declared_stage="historical")))
    _, p = resolve_parent(dict(init=dict(path=base, expect_sha256=real, declared_stage="historical", expect_stage="historical")))
    assert p["chain"] == ["historical"]
    with pytest.raises(SystemExit):
        resolve_parent(dict(init=dict(path=base, expect_sha256=real, declared_stage="historical", expect_stage="base")))


def env_sh(tmp_path, **kw):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("A5_", "PT_", "CUDA_"))}
    e.update({k: str(v) for k, v in kw.items()})
    return subprocess.run(["bash", "-c", f"source {ROOT}/scripts/server_env.sh && env"], env=e, capture_output=True, text=True, cwd=str(tmp_path))


def test_server_env_refuses_unsafe_setups(tmp_path):
    ok = dict(A5_AUTHORIZED="yes", A5_GPUS="2,3", A5_ROOT=tmp_path / "new")
    for bad in (dict(A5_AUTHORIZED=""), dict(A5_GPUS="0"), dict(A5_GPUS="1,1"), dict(A5_GPUS="0,1,2,3"), dict(A5_ROOT="relative"), dict(A5_ROOT=OLD), dict(A5_ROOT=OLD + "/x"),
                dict(A5_ROOT="/srv/research"), dict(A5_ROOT=OLD + "/../storage/y")):
        r = env_sh(tmp_path, **{**ok, **bad})
        assert r.returncode != 0 and "REFUSED" in r.stdout, bad
    assert not (tmp_path / "new").exists()
    r = env_sh(tmp_path, **ok, GPU_USD_PER_HOUR="4.41", RUNPOD_API_KEY="x", ALLOW_FINAL_EVAL="1", POD_START_EPOCH="1")
    v = dict(x.split("=", 1) for x in r.stdout.splitlines() if "=" in x)
    assert r.returncode == 0 and v["CUDA_VISIBLE_DEVICES"] == "2,3" and v["PT_HW"] == "a6000" and v["PT_PLATFORM"] == "research_server"
    assert not {"GPU_USD_PER_HOUR", "RUNPOD_API_KEY", "ALLOW_FINAL_EVAL", "POD_START_EPOCH"} & set(v)
    assert all(v[k].startswith(str(tmp_path)) for k in ("PT_RUNS", "HF_HOME", "XDG_CACHE_HOME", "UV_CACHE_DIR", "TMPDIR", "TORCH_HOME", "PT_DATA"))


def test_runpod_scripts_are_disabled_and_server_scripts_have_no_dangerous_commands():
    import re
    pod = ("scripts/pod_env.sh", "scripts/pod_watchdog.sh", "scripts/check_volume.sh", "scripts/remote_pilot.sh", "scripts/pilot.sh", "scripts/pull_backup.sh", "setup_cloud.sh")
    e = {k: v for k, v in os.environ.items() if k != "A5_PLATFORM_RUNPOD"}
    for f in pod:
        if os.path.exists(os.path.join(ROOT, f)):  # they are not shipped to the server
            r = subprocess.run(["bash", os.path.join(ROOT, f), "start"], env=e, capture_output=True, text=True)
            assert r.returncode == 1 and "REFUSED: RunPod-only" in r.stdout, f
    for f in ("scripts/server_env.sh", "scripts/server_preflight.sh", "scripts/server_validation.sh", "setup_server.sh", "scripts/grpo_resume_test.py"):
        code = "\n".join(x.split("#")[0] for x in open(os.path.join(ROOT, f), encoding="utf-8"))
        assert not re.search(r"runpodctl|\bshutdown\b(?!.*off)|poweroff|reboot|pkill|killall|\bsudo\b|rm -r|/workspace", code), f


def test_run_experiment_gates():
    def go(i, *args, **env):
        e = {k: v for k, v in os.environ.items() if not k.startswith(("A5_", "PT_"))}
        return subprocess.run([sys.executable, f"{ROOT}/scripts/run_experiment.py", "--id", i, *args], env={**e, **env}, capture_output=True, text=True)
    assert "DEFERRED" in go("EI-base", "--go").stderr and "HOLD_B" in go("SFT-exact_full_s1", "--go").stderr and go("HIST-SFT512", "--go").returncode != 0
    assert "PT_HW" in go("SFT-tune_lr1e-5", "--go", PT_PLATFORM="research_server").stderr
    assert "not been authorized" in go("SFT-tune_lr1e-5", "--go", PT_PLATFORM="research_server", PT_HW="a6000").stderr
    r = go("GRPO-A_base_s2026", PT_PLATFORM="research_server", PT_HW="a6000")  # prints only
    assert r.returncode == 0 and "train.micro_batch=2 train.token_budget=3300" in r.stdout
    assert "--set" not in go("GRPO-A_base_s2026").stdout


class FakeVLLM:
    """Stand-in with vLLM's interface: its own bf16 copy of the weights and merged parameter names. `fault` breaks it on purpose."""

    shares_policy = False
    probe = 0.03

    def __init__(self, model, fault=None):
        self.m, self.fault = copy.deepcopy(model).to(torch.bfloat16), fault

    def _model(self):
        return self

    def load_weights(self, stream):
        if self.fault == "ignores_updates":
            return
        sd = dict(self.m.named_parameters())
        for k, v in stream:
            if k in sd and not (self.fault == "drops_one_tensor" and k == "model.layers.0.mlp.up_proj.weight"):
                with torch.no_grad():
                    sd[k].copy_(v)

    def named_parameters(self):
        return expected_engine_state(self.m.state_dict()).items()

    def sync(self, model):
        self.load_weights(model.state_dict().items())
        return {}

    def check_params(self, model):
        return compare_engine_state(expected_engine_state(model.state_dict()), dict(self.named_parameters()))

    def check_sync(self, model, prompts):
        return dict(sync_mean_abs_logprob_diff=self.probe, sync_max_abs_logprob_diff=self.probe)

    def generate(self, prompts, n, temperature, max_tokens, stop, seed, want_logprobs=False):
        g, tok = torch.Generator().manual_seed(seed), CharTok()
        out = []
        for p in prompts:
            row = []
            for _ in range(n):
                ids, lps = list(tok(p)["input_ids"]), []
                k = len(ids)
                for _ in range(6):
                    lp = torch.log_softmax(self.m(torch.tensor([ids])).logits[0, -1].float(), -1)
                    i = torch.multinomial(lp.exp(), 1, generator=g).item()
                    ids.append(i); lps.append(lp[i].item())
                if self.fault == "drops_a_logprob":
                    lps = lps[:-1]
                if self.fault == "wrong_logprobs":
                    lps = [x - 0.2 for x in lps]
                if self.fault == "no_logprobs":
                    lps = []
                row.append(Completion("", ids[k:], "stop", lps))
            out.append(row)
        return out


def run_fake_sync(fault=None):
    sys.path.insert(0, ROOT + "/scripts")
    import gpu_validation as V
    m = tiny()
    return V.run_sync(FakeVLLM(m, fault), m, CharTok(), ["ab", "cd", "ef", "gh"], [("ab", "cdefg"), ("abc", "xyz"), ("a", "bcdef"), ("abcd", "zz")], "cpu", n=2, lr=1e-3)


def test_gpu_validation_sync_logic_passes_a_correct_engine():
    r = run_fake_sync()
    assert r["pass"], r["gates"]
    assert r["3_stale_by_one_step"]["params_mismatched"] > 0 and r["4_one_tensor_not_synced"]["mismatch_names"] == ["model.layers.1.self_attn.qkv_proj.weight"]


@pytest.mark.parametrize("fault", ["ignores_updates", "drops_one_tensor"])
def test_gpu_validation_sync_logic_fails_broken_engines(fault):
    r = run_fake_sync(fault)
    assert not r["pass"] and not r["gates"]["exact_after_sync"]


def test_gpu_validation_sync_logic_rejects_misaligned_logprobs():
    with pytest.raises(RuntimeError):
        run_fake_sync("drops_a_logprob")


def test_gpu_validation_policy_logic(monkeypatch):
    sys.path.insert(0, ROOT + "/scripts")
    import gpu_validation as V
    monkeypatch.setattr(V, "prompt_of", lambda tok, ids, n: "a" * n)
    r = V.run_policy(tiny(), CharTok(), [2 + i % 60 for i in range(500)], "cpu", scale=0.1, total_gib=48)
    assert r["pass"], r["gates"]
    assert r["gates"]["gradient_check_detects_a_lost_rollout"] and r["partition"]["largest_padded"] <= 330


def test_final_eval_config_runs_both_base_prompts():
    assert yaml.safe_load(open(ROOT + "/configs/eval/final.yaml"))["prompts"] == ["r1_zero", "question_only"]


@need
def test_historical_rl_pool_contains_math_dev_but_not_gsm8k_dev():
    """Why the historical GRPO checkpoint may only ever be compared on gsm8k_dev (RESEARCH_SERVER_AUDIT.md)."""
    f = ROOT + "/results/recovered_audit/historical_audit.json"
    if not os.path.exists(f):
        pytest.skip("audit output not shipped")
    c = json.load(open(f))["contamination"]["grpo_pool"]
    assert c["math_dev"]["exact"] == 500 and c["gsm8k_dev"]["exact"] == c["gsm8k_test"]["exact"] == 0


# ---------- approved sync gates: exact weights (primary), rollout log-probs (separate gate), probe (diagnostic) ----------
def test_exact_check_rejects_wrong_dtype_extra_missing_and_tiny_changes():
    e = expected_engine_state(tiny().state_dict()); good = {k: v.clone() for k, v in e.items()}
    assert compare_engine_state(e, good)["params_mismatched"] == 0
    k = next(iter(e))
    assert compare_engine_state(e, {**good, k: good[k].float()})["mismatch_names"] == [k]           # same values, another dtype
    x = compare_engine_state(e, {**good, "extra.weight": torch.zeros(1)})                            # engine parameter nobody verified
    assert x["unexpected"] == x["params_mismatched"] == 1 and x["mismatch_names"] == ["extra.weight"]
    assert compare_engine_state(e, {x: v for x, v in good.items() if x != k})["missing"] == 1
    t = {**good, k: good[k].clone()}; t[k].view(-1)[0] = t[k].view(-1)[0] * 1.0079
    assert compare_engine_state(e, t)["mismatch_names"] == [k]                                      # one element, one bf16 step
    n = {**good, k: good[k].clone()}; n[k].view(-1)[0] = float("nan")
    assert compare_engine_state(e, n)["params_mismatched"] == 1


def test_verified_sync_is_the_gate_and_cannot_pass_vacuously():
    m = tiny(); eng = FakeVLLM(m)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.01)
    r = verified_sync(eng, m)
    assert r["params_checked"] > 0 and r["params_mismatched"] == 0
    for fault in ("ignores_updates", "drops_one_tensor"):
        m = tiny(); eng = FakeVLLM(m, fault)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01)
        with pytest.raises(SystemExit, match="weight-sync check failed"):
            verified_sync(eng, m)

    class Empty(FakeVLLM):
        def check_params(self, model):
            return dict(params_checked=0, params_mismatched=0, mismatch_names=[])
    with pytest.raises(SystemExit):
        verified_sync(Empty(tiny()), tiny())
    assert verified_sync(HFEngine(tiny(), CharTok()), tiny())["params_checked"] == 0  # shares the policy object: nothing to copy


def test_rollout_gate():
    c = [Completion("", [3, 4], "stop", [-1.0, -2.0]), Completion("", [5], "stop", [-0.5])]
    assert check_rollouts(c, [torch.tensor([-1.01, -2.0]), torch.tensor([-0.5])], 0.05)["rollout_logprob_tokens"] == 3
    with pytest.raises(SystemExit, match="exceeds"):
        check_rollouts(c, [torch.tensor([-1.3, -2.0]), torch.tensor([-0.5])], 0.05)
    with pytest.raises(SystemExit, match="exceeds"):
        check_rollouts(c, [torch.tensor([float("nan"), -2.0]), torch.tensor([-0.5])], 0.05)
    with pytest.raises(SystemExit, match="every rollout token"):
        check_rollouts([c[0], Completion("", [5], "stop", [])], [torch.tensor([-1.0, -2.0]), torch.tensor([-0.5])], 0.05)
    with pytest.raises(RuntimeError):
        check_rollouts(c, [torch.tensor([-1.0, -2.0, -3.0]), torch.tensor([-0.5])], 0.05)
    assert check_rollouts(c, None, 0.05, shares_policy=True) == {}


class SaveTok(CharTok):
    def save_pretrained(self, p):
        open(os.path.join(p, "tok.txt"), "w").write("x")


def grpo_main(tmp_path, monkeypatch, fault=None, probe=0.03, extra=()):
    """Runs the REAL train_grpo.main() for two steps on a tiny model with a stand-in vLLM. Returns the logged training rows."""
    import posttrain.train_grpo as G
    base = str(tmp_path / "base"); tiny().save_pretrained(base)
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps(dict(id=f"p{i}", question=f"What is {i}?", answer="4")) + "\n" for i in range(6)))
    monkeypatch.setenv("PT_RUNS", str(tmp_path / "runs")); monkeypatch.setenv("PT_DATA", str(tmp_path)); monkeypatch.setenv("PT_MODEL", base)
    monkeypatch.setattr(G, "load_tokenizer", lambda p: SaveTok())
    seen = {}

    def engine(cfg, path, model, tok):
        eng = FakeVLLM(model, fault); eng.probe = probe; seen["eng"] = eng
        return eng
    monkeypatch.setattr(G, "make_engine", engine)
    monkeypatch.setattr(G, "grpo_update", lambda *a, **k: (seen.__setitem__("updates", seen.get("updates", 0) + 1), grpo_update(*a, **k))[1])
    # rewards that differ inside each group, so every step really updates the weights
    k = iter(range(10 ** 6))
    monkeypatch.setattr(G, "score", lambda t, g: dict(reward=float(next(k) % 2), answer_reward=0.0, format_reward=0.0, error=None, rule=None))
    monkeypatch.setattr(sys, "argv", ["train_grpo", "--config", ROOT + "/configs/grpo/A_base_s2026.yaml", "--set", "device=cpu", "engine.name=hf", "eval.sets={}", "eval.at_start=false",
                                      f"data.path={pool}", "init.revision=null", "grpo.steps=2", "grpo.prompts_per_step=2", "grpo.group_size=2", "grpo.update_batch=4", "grpo.max_tokens=6",
                                      "grpo.sync_check_every=1", "train.micro_batch=2", "train.lr=1e-2", "train.save_every=null", *extra])
    G.main()
    rows = [json.loads(x) for x in open(tmp_path / "runs" / "grpo" / "A_base_s2026" / "metrics.jsonl")]
    return [r for r in rows if r["event"] == "train"], seen


def test_grpo_main_probe_above_0_05_is_recorded_and_does_not_stop(tmp_path, monkeypatch):
    rows, seen = grpo_main(tmp_path, monkeypatch, probe=0.9)
    assert len(rows) == 2 and seen["updates"] == 2
    assert all(r["sync_mean_abs_logprob_diff"] == 0.9 and r["sync_probe_over_tol"] is True for r in rows)
    assert all(r["params_checked"] > 0 and r["params_mismatched"] == 0 and r["rollout_logprob_mad"] <= 0.05 and r["rollout_logprob_tokens"] > 0 for r in rows)


def test_grpo_main_exact_check_runs_every_step_even_if_the_probe_does_not(tmp_path, monkeypatch):
    rows, _ = grpo_main(tmp_path, monkeypatch, extra=("grpo.sync_check_every=10",))
    assert all(r["params_checked"] > 0 for r in rows) and "sync_mean_abs_logprob_diff" not in rows[1]


@pytest.mark.parametrize("fault,updates", [("ignores_updates", 1), ("drops_one_tensor", 1), ("wrong_logprobs", 0), ("no_logprobs", 0)])
def test_grpo_main_stops_before_training_on_bad_weights_or_rollouts(tmp_path, monkeypatch, fault, updates):
    """Stale weights are caught at step 2 (step 1 starts in sync) although the probe reads a comfortable 0.03; bad log-probs at step 1, before any update."""
    with pytest.raises(SystemExit, match="check failed"):
        grpo_main(tmp_path, monkeypatch, fault=fault, probe=0.03)
    rows = [json.loads(x) for x in open(tmp_path / "runs" / "grpo" / "A_base_s2026" / "metrics.jsonl")] if os.path.exists(tmp_path / "runs" / "grpo" / "A_base_s2026" / "metrics.jsonl") else []
    assert len([r for r in rows if r["event"] == "train"]) == updates
    assert not os.path.exists(tmp_path / "runs" / "grpo" / "A_base_s2026" / "checkpoint")


def test_grpo_main_refuses_to_disable_the_rollout_gate(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="correctness gate"):
        grpo_main(tmp_path, monkeypatch, extra=("grpo.check_rollout_logprobs=false",))


def test_every_engine_sync_in_training_code_is_verified():
    import re
    for f in ("train_sft", "train_grpo", "train_ei"):
        code = open(f"{ROOT}/posttrain/{f}.py", encoding="utf-8").read()
        assert not re.search(r"\beng\.sync\(", code) and "verified_sync(eng, model)" in code, f


@need
def test_all_grpo_configs_keep_both_gates_and_the_diagnostic(monkeypatch):
    monkeypatch.setenv("PT_DATA", D); monkeypatch.setenv("PT_RUNS", "/tmp/pt_runs")
    for f in glob.glob(ROOT + "/configs/grpo/*.yaml") + glob.glob(ROOT + "/configs/pilot/*.yaml"):
        g = load_config(f)["grpo"]
        assert g["check_rollout_logprobs"] is True and g["rollout_tol"] == 0.05 and g["sync_tol"] == 0.05 and g["sync_check_every"], f

