"""GPU validation for the research server. Trains nothing that is kept; every weight change is thrown away.

python scripts/gpu_validation.py --part sync   --model $PT_MODEL --out <new.json>     two GPUs, vLLM on cuda:0, policy on cuda:1
python scripts/gpu_validation.py --part policy --model $PT_MODEL --out <new.json>     one GPU (cuda:1), no vLLM

sync   : exact weight equality after a sync; deliberately stale and partly stale engines must be DETECTED; sampled log-probs must agree after a sync.
policy : micro-batch 1 / 2 / 2 with a 3,300-token cap must give the same gradient; long-sequence memory stress with Adam state allocated.

--layers / --scale / --device shrink the policy part for a local dry run on a small GPU. A dry run proves the script works, not that 48 GB is enough.
"""
import argparse, json, math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from posttrain import losses as L
from posttrain import train_grpo
from posttrain.batching import microbatches, padded_tokens
from posttrain.common import ROOT, jdump, jload
from posttrain.engines import Completion
from posttrain.evaluate import template
from posttrain.model_io import load_policy, load_tokenizer, make_optimizer, resolve_parent
from posttrain.train_grpo import grpo_update, logprobs_nograd, sampling_mismatch
from posttrain.train_sft import sft_step

ROLLOUT_TOL = 0.05   # the pilot's existing gate on mean |sampling log-prob - policy log-prob|; unchanged
PROBE_TOL = 0.05     # the original probe tolerance: a diagnostic, never a gate
SKIP = "model.layers.1.self_attn.k_proj.weight"
GIB = 2 ** 30


# ------------------------------------------------------------------ sync ------------------------------------------------------------------
def load_except(eng, policy, skip=()):
    """Sync every tensor except `skip`: a deliberately partial update."""
    eng._model().load_weights((k, v) for k, v in policy.state_dict().items() if k not in skip)


def rollout_gap(eng, policy, tok, prompts, dev, n, max_tokens, seed):
    g = eng.generate(prompts, n, 1.0, max_tokens, ["</answer>"], seed, want_logprobs=True)
    comps, flat = [c for x in g for c in x], [p for p in prompts for _ in range(n)]
    d = sampling_mismatch(comps, logprobs_nograd(policy, tok, flat, comps, dev, 2, 2064))  # raises if the token counts do not line up
    return dict(d, rollouts=len(comps), tokens=sum(len(c.token_ids) for c in comps))


def run_sync(eng, policy, tok, prompts, examples, dev, n=4, max_tokens=512, lr=1e-5, skip=SKIP):
    rep, P = {}, lambda: eng.check_params(policy)
    cfg = dict(train=dict(lr=lr, betas=[0.9, 0.95], weight_decay=0.0, micro_batch=2, max_grad_norm=1.0, append_eos=True, max_len=2064, log_entropy=False))
    opt = make_optimizer(policy, cfg["train"])

    def steps(k, rate):
        for g in opt.param_groups:
            g["lr"] = rate
        for _ in range(k):
            sft_step(policy, opt, tok, examples, cfg, dev)
    eng.sync(policy)
    rep["1_after_initial_sync"] = dict(P(), **eng.check_sync(policy, prompts[:4]))
    rep["2_rollouts_after_sync"] = rollout_gap(eng, policy, tok, prompts, dev, n, max_tokens, 1)
    steps(1, lr)  # ONE real optimizer step at a training-sized learning rate, engine not synced
    rep["3_stale_by_one_step"] = dict(P(), **eng.check_sync(policy, prompts[:4]))
    load_except(eng, policy, (skip,))
    rep["4_one_tensor_not_synced"] = P()
    eng.sync(policy)
    steps(1, lr)
    load_except(eng, policy, ("model.embed_tokens.weight", "lm_head.weight"))
    rep["5_embeddings_not_synced"] = dict(P(), **eng.check_sync(policy, prompts[:4]))
    eng.sync(policy)
    rep["6_after_resync"] = dict(P(), **eng.check_sync(policy, prompts[:4]))
    rep["7_rollouts_after_resync"] = rollout_gap(eng, policy, tok, prompts, dev, n, max_tokens, 2)
    steps(10, 5e-5)  # a large drift, so the rollout comparison itself has a positive control
    rep["8_rollouts_from_stale_engine"] = rollout_gap(eng, policy, tok, prompts, dev, n, max_tokens, 3)
    rep["8_params"] = P()
    eng.sync(policy)
    rep["9_after_final_sync"] = P()
    bad = lambda k: rep[k]["params_mismatched"]
    parts = {"q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj", "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj"}
    want = skip
    for a, b in parts.items():
        want = want.replace("." + a + ".", "." + b + ".")
    mad = lambda k: rep[k].get("rollout_logprob_mad", math.inf)
    rep["gates"] = dict(
        exact_after_sync=bad("1_after_initial_sync") == 0 and bad("6_after_resync") == 0 and bad("9_after_final_sync") == 0 and rep["1_after_initial_sync"]["params_checked"] > 0 and len(dict(eng._model().named_parameters())) == rep["1_after_initial_sync"]["params_checked"],
        one_step_stale_detected=bad("3_stale_by_one_step") > 0,
        single_stale_tensor_named_exactly=rep["4_one_tensor_not_synced"]["mismatch_names"] == [want],
        stale_embeddings_named_exactly=rep["5_embeddings_not_synced"]["mismatch_names"] == ["model.embed_tokens.weight"],
        rollout_logprobs_agree=mad("2_rollouts_after_sync") < ROLLOUT_TOL and mad("7_rollouts_after_resync") < ROLLOUT_TOL,
        rollout_check_sees_large_staleness=mad("8_rollouts_from_stale_engine") > 2 * mad("7_rollouts_after_resync") and bad("8_params") > 0)
    k = "sync_mean_abs_logprob_diff"
    rep["probe_reported_only"] = {s: dict(value=rep[s][k], over_0_05=rep[s][k] > PROBE_TOL) for s in rep if isinstance(rep[s], dict) and k in rep[s]}
    rep["pass"] = all(rep["gates"].values())
    return rep


# ----------------------------------------------------------------- policy ------------------------------------------------------------------
class GradCatcher:
    """Stands in for the optimizer: keeps the gradient of one update on the CPU and changes nothing."""

    def __init__(self, model):
        self.model, self.param_groups, self.grads = model, [dict(lr=0.0)], None

    def step(self):
        self.grads = [p.grad.detach().float().cpu().clone() for p in self.model.parameters() if p.grad is not None]

    def zero_grad(self, set_to_none=True):
        for p in self.model.parameters():
            p.grad = None


def compare(a, b):
    dot = sum(torch.dot(x.flatten().double(), y.flatten().double()).item() for x, y in zip(a, b))
    na, nb = (math.sqrt(sum((x.double() ** 2).sum().item() for x in v)) for v in (a, b))
    return dict(cosine=dot / max(na * nb, 1e-30), norm_ratio=nb / max(na, 1e-30), bitwise_equal=all(torch.equal(x, y) for x, y in zip(a, b)))


def synth(tok, text_ids, total, prompt_len, rng):
    """(prompt string, Completion) whose combined length is exactly `total` tokens."""
    s = rng.randrange(0, max(1, len(text_ids) - 8))
    ids = [text_ids[(s + i) % len(text_ids)] for i in range(total - prompt_len)]
    return Completion("", ids, "stop")


def prompt_of(tok, text_ids, n):
    """A prompt string that tokenises to exactly n tokens (a repeated single-token word)."""
    p = " the" * n
    got = len(tok(p, add_special_tokens=False)["input_ids"])
    assert got == n, (got, n)
    return p


def batch(tok, text_ids, lengths, rng, plen):
    p = prompt_of(tok, text_ids, plen)
    return [p] * len(lengths), [synth(tok, text_ids, n, plen, rng) for n in lengths]


def grpo_cfg(mb, tb, ub):
    return dict(train=dict(micro_batch=mb, token_budget=tb, max_grad_norm=1e9, max_len=2064),
                grpo=dict(loss="grpo_clip", clip=0.2, length_norm="const", norm_const=1024, epochs=1, update_batch=ub, group_filter="none"))


class no_autocast:
    """Runs the update in fp32. Padding changes bf16 rounding, so only fp32 separates a partitioning error from rounding noise."""

    def __enter__(self):
        self.old = train_grpo.autocast
        train_grpo.autocast = lambda device, on=True: self.old(device, False)

    def __exit__(self, *a):
        train_grpo.autocast = self.old


def grads_of(model, tok, dev, prompts, comps, adv, rw, mb, tb):
    old = logprobs_nograd(model, tok, prompts, comps, dev, 2, 2064, tb)
    c = GradCatcher(model)
    d = grpo_update(model, c, tok, prompts, comps, adv, rw, old, grpo_cfg(mb, tb, len(comps)), dev, 1)
    g = c.grads
    c.zero_grad()
    return g, d


def equivalence(model, tok, dev, prompts, comps, adv, rw, cap, control=False):
    (g0, d0), out = grads_of(model, tok, dev, prompts, comps, adv, rw, 2, None), {}
    for name, mb, tb in (("mb2_again", 2, None), ("mb1", 1, None), ("mb2_cap", 2, cap)):
        g, d = grads_of(model, tok, dev, prompts, comps, adv, rw, mb, tb)
        out[name] = dict(compare(g0, g), loss=d["loss"], loss_ref=d0["loss"], same_update=d["trained_rollouts"] == d0["trained_rollouts"] == len(comps) and d["updates"] == d0["updates"] == 1 and d["tok_sum"] == d0["tok_sum"])
        del g
    if control:  # what a real accounting error looks like: one rollout with a non-zero advantage does not contribute
        a = adv.clone()
        i = max((j for j in range(len(comps)) if adv[j] != 0), key=lambda j: abs(adv[j].item()))
        a[i] = 0
        g, _ = grads_of(model, tok, dev, prompts, comps, a, rw, 2, None)
        out["control_one_rollout_lost"] = compare(g0, g)
    return out


def run_policy(model, tok, text_ids, dev, scale=1.0, total_gib=None, n_grad=32):
    S = lambda x: max(24, int(x * scale))
    cap, plen, rng, rep = S(3300), 16, random.Random(0), {}
    # A. gradient equivalence on one batch with realistic, very unequal lengths; advantages of both signs
    lens = [S(min(2048, max(120, int(rng.lognormvariate(5.8, 0.6))))) for _ in range(n_grad - 2)] + [S(1650), S(2048)]
    prompts, comps = batch(tok, text_ids, lens, rng, plen)
    rw = torch.tensor([float(rng.random() < 0.3) for _ in lens])
    adv = L.group_advantages(rw, 8)
    T = dict(fp32=dict(cosine=0.9999, norm=0.001), bf16=dict(cosine=0.99, norm=0.02))
    half = [max(24, n // 2) for n in lens]  # fp32 activations are twice the size, so the fp32 check uses half the lengths
    p2, c2 = batch(tok, text_ids, half, rng, plen)
    with no_autocast():
        rep["gradients_fp32"] = equivalence(model, tok, dev, p2, c2, adv, rw, cap // 2, control=True)
    rep["gradients_bf16"] = equivalence(model, tok, dev, prompts, comps, adv, rw, cap)
    parts = microbatches(sorted(range(len(lens)), key=lambda i: lens[i]), lens, 2, cap)
    rep["partition"] = dict(longest=max(lens), cap=cap, microbatches=len(parts), largest_padded=max(padded_tokens(parts, lens)))
    ok = lambda v, k: v["cosine"] >= T[k]["cosine"] and abs(v["norm_ratio"] - 1) <= T[k]["norm"]
    ctl = rep["gradients_fp32"].pop("control_one_rollout_lost")
    rep["gradients_fp32_control_one_rollout_lost"] = ctl
    same = all(v["same_update"] for k in ("gradients_fp32", "gradients_bf16") for v in rep[k].values())
    close32, close16 = all(ok(v, "fp32") for v in rep["gradients_fp32"].values()), all(ok(v, "bf16") for v in rep["gradients_bf16"].values())
    # B. allocate Adam state with one tiny real step, as in training
    opt = make_optimizer(model, dict(lr=1e-8, betas=[0.9, 0.95]))
    p8, c8 = batch(tok, text_ids, [S(300)] * 8, rng, plen)
    a8 = torch.tensor([1.0, -1.0] * 4)
    grpo_update(model, opt, tok, p8, c8, a8, a8, logprobs_nograd(model, tok, p8, c8, dev, 2, 2064), grpo_cfg(2, None, 8), dev, 1)
    # C. stress cases, expected to fit; then two that may not
    cuda = str(dev).startswith("cuda")
    cases = [("cap_worst_two_1650", [S(1650)] * 8, 2, cap, True), ("longest_single_2048", [S(2048)] * 4, 2, cap, True),
             ("mixed_64_with_long_tail", [S(min(2048, max(80, int(rng.lognormvariate(5.8, 0.8))))) for _ in range(54)] + [S(2048)] * 10, 2, cap, True),
             ("sft_worst_two_1267", [S(1267)] * 4, 2, None, True), ("plain_mb2_two_1651 (pilot's longest)", [S(1651)] * 4, 2, None, True),
             ("plain_mb2_two_2048 (may not fit)", [S(2048)] * 4, 2, None, False)]
    rep["stress"] = {}
    for name, ls, mb, tb, must in cases:
        r = dict(must_fit=must, micro_batch=mb, token_cap=tb, longest=max(ls), n=len(ls))
        ps, cs = batch(tok, text_ids, ls, rng, plen)
        av = torch.tensor([1.0, -1.0] * (len(ls) // 2))
        order = sorted(range(len(ls)), key=lambda i: ls[i])
        pad = max(padded_tokens(microbatches(order, ls, mb, tb), ls))
        r.update(padded=pad, predicted_gib=round(25.43 + 5.10 * pad / 1000, 2) if scale == 1.0 else None)
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
        try:
            o = logprobs_nograd(model, tok, ps, cs, dev, mb, 2064, tb)
            d = grpo_update(model, opt, tok, ps, cs, av, av, o, grpo_cfg(mb, tb, len(ls)), dev, 1)
            r.update(fits=True, updates=d["updates"], finite=math.isfinite(d["grad_norms"][0]))
        except torch.OutOfMemoryError as e:
            opt.zero_grad(set_to_none=True)
            r.update(fits=False, error=str(e)[:160])
            e = None
        if cuda:
            r.update(peak_allocated_gib=round(torch.cuda.max_memory_allocated(dev) / GIB, 2), peak_reserved_gib=round(torch.cuda.max_memory_reserved(dev) / GIB, 2))
        rep["stress"][name] = r
        print(name, r, flush=True)
    need = [v for v in rep["stress"].values() if v["must_fit"]]
    limit = (total_gib or (torch.cuda.get_device_properties(dev).total_memory / GIB if cuda else math.inf)) - 2.0
    rep["memory_limit_gib"] = limit
    rep["gates"] = dict(same_update=same, gradients_equal_fp32=close32, gradients_close_bf16=close16, gradient_check_detects_a_lost_rollout=not ok(ctl, 'fp32'), token_cap_respected=rep["partition"]["largest_padded"] <= max(cap, max(lens)),
                        required_cases_fit=all(v.get("fits") and v.get("finite") for v in need),
                        two_gib_headroom=all(v.get("peak_allocated_gib", 0) <= limit for v in need))
    rep["thresholds"] = dict(T, headroom_gib=2.0)
    rep["pass"] = all(rep["gates"].values())
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("sync", "policy"), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    assert not os.path.exists(a.out), "refusing to overwrite"
    path, parent = resolve_parent(dict(init=dict(path=a.model, revision=a.revision, expect_stage="base")))
    tok = load_tokenizer(path)
    data = os.environ.get("PT_DATA", os.path.join(ROOT, "data", "stage0"))
    sft = jload(data + "/sft_exactmatch_3496.jsonl")[:64]
    if a.part == "sync":
        from posttrain.engines import VLLMEngine
        assert torch.cuda.device_count() >= 2, "needs two GPUs"
        pol = load_policy(path, a.device, "fp32")
        eng = VLLMEngine(path, "cuda:0", prefix_caching=False)
        tpl = template("r1_zero")
        prompts = [tpl.format(question=r["question"].strip()) for r in jload(data + "/rl_train_pool.jsonl")[:16]]
        ex = [(tpl.format(question=r["problem"].strip()), r["reasoning_trace"]) for r in sft[:8]]
        rep = run_sync(eng, pol, tok, prompts, ex, a.device)
    else:
        pol = load_policy(path, "cpu", "fp32")
        if a.layers:
            pol.model.layers = pol.model.layers[:a.layers]
        pol = pol.to(a.device)
        ids = [t for r in sft for t in tok(r["reasoning_trace"], add_special_tokens=False)["input_ids"]]
        rep = run_policy(pol, tok, ids, a.device, a.scale)
        rep.update(dry_run=bool(a.layers or a.scale != 1.0), layers=a.layers, scale=a.scale)
    rep.update(part=a.part, gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], torch=torch.__version__, weights=parent["weights_sha256"])
    jdump(a.out, rep)
    print(json.dumps(rep, indent=1, default=str))
    sys.exit(0 if rep["pass"] else 1)


if __name__ == "__main__":
    main()
