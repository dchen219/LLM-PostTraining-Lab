"""Generation back ends with one interface: generate() -> list[list[Completion]], sync(model), check_sync(model, prompts)."""
import time
from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class Completion:
    text: str
    token_ids: List[int] = field(default_factory=list)
    finish: str = "length"  # "stop" or "length"
    logprobs: List[float] = field(default_factory=list)  # sampling-time log-probs, if requested


class HFEngine:
    """Reference engine for CPU tests and small GPUs. Shares the policy object, so sync is the identity."""
    name = "hf"
    shares_policy = True  # generates from the policy object itself, so there is no copy that could be stale

    def __init__(self, model, tok, batch_size=16):
        self.model, self.tok, self.bs = model, tok, batch_size

    def sync(self, model):
        self.model = model
        return dict(sync_s=0.0)

    def check_sync(self, model, prompts):
        return dict(sync_max_abs_logprob_diff=0.0, sync_mean_abs_logprob_diff=0.0)

    def check_params(self, model):
        return dict(params_checked=0, params_mismatched=0, mismatch_names=[], missing=0)  # shares the policy object

    @torch.no_grad()
    def generate(self, prompts, n=1, temperature=0.0, max_tokens=256, stop=None, seed=0, want_logprobs=False):
        was = self.model.training
        self.model.eval()
        dev = next(self.model.parameters()).device
        flat = [p for p in prompts for _ in range(n)]
        out = []
        rng = torch.get_rng_state()
        torch.manual_seed(seed)
        side, self.tok.padding_side = self.tok.padding_side, "left"
        for s in range(0, len(flat), self.bs):
            enc = self.tok(flat[s:s + self.bs], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 else \
                dict(do_sample=False, temperature=None, top_p=None, top_k=None)
            if stop:
                kw.update(stop_strings=list(stop), tokenizer=self.tok)
            g = self.model.generate(**enc, max_new_tokens=max_tokens, pad_token_id=self.tok.pad_token_id, use_cache=True, **kw)
            for ids in g[:, enc["input_ids"].shape[1]:].tolist():
                out.append(self._finish(ids, stop))
        self.tok.padding_side = side
        torch.set_rng_state(rng)
        if was:
            self.model.train()
        return [out[i * n:(i + 1) * n] for i in range(len(prompts))]

    def _finish(self, ids, stop):
        fin, eos = "length", self.tok.eos_token_id
        if eos in ids:
            ids, fin = ids[:ids.index(eos) + 1], "stop"  # keep EOS: it is a sampled action, as in vLLM's token_ids
        dec = lambda x: self.tok.decode(x, skip_special_tokens=True)
        txt = dec(ids)
        for st in stop or []:
            if st in txt:
                txt, fin = txt[:txt.index(st) + len(st)], "stop"
                ids = ids[:next((j for j in range(1, len(ids) + 1) if st in dec(ids[:j])), len(ids))]
        return Completion(txt, ids, fin)


class VLLMEngine:
    """vLLM 0.7.2 in process. NOT runnable on the author's laptop: validated by scripts/pilot_sync_check.py.

    Placement: UniProcExecutor in 0.7.2 hard-codes local_rank = 0, so the engine ALWAYS lands on cuda:0 and any other
    `device` index is ignored. The policy therefore lives on cuda:1; placement is asserted after loading.

    Prefix caching is off by default. After in-place weight updates, cached KV blocks of the shared prompt prefix would
    otherwise come from the previous policy. If enabled, sync() resets the cache and fails loudly if it cannot.
    """
    name = "vllm"
    shares_policy = False

    def __init__(self, path, device="cuda:0", gpu_mem=0.8, max_model_len=2048, prefix_caching=False, seed=2026):
        from unittest.mock import patch
        from vllm import LLM
        if device != "cuda:0":
            raise ValueError("vLLM 0.7.2 in-process always uses cuda:0; put the policy on cuda:1")
        self.prefix_caching = prefix_caching
        with patch("torch.distributed.get_world_size", return_value=1), \
                patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None):
            self.llm = LLM(model=path, device="cuda", dtype="bfloat16", enable_prefix_caching=prefix_caching, seed=seed,
                           gpu_memory_utilization=gpu_mem, max_model_len=max_model_len, tensor_parallel_size=1)
        where = {str(p.device) for p in self._model().parameters()}
        if where != {device}:
            raise RuntimeError(f"vLLM weights are on {where}, expected {device}")

    def _model(self):
        return self.llm.llm_engine.model_executor.driver_worker.model_runner.model

    def sync(self, model):
        t = time.time()
        self._model().load_weights((k, v) for k, v in model.state_dict().items())
        if self.prefix_caching and not self.llm.reset_prefix_cache():
            raise RuntimeError("prefix cache reset failed after weight sync")
        return dict(sync_s=time.time() - t)

    @torch.no_grad()
    def check_sync(self, model, prompts):
        """Prompt log-probs from vLLM vs PyTorch on identical tokens. Catches stale or partially loaded weights."""
        from vllm import SamplingParams
        outs = self.llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)
        worst, means = 0.0, []
        was = model.training
        model.eval()
        dev = next(model.parameters()).device
        for o in outs:
            ids = list(o.prompt_token_ids)
            with torch.autocast("cuda", torch.bfloat16):
                lg = model(input_ids=torch.tensor([ids], device=dev)).logits[0, :-1].float()
            lp = torch.log_softmax(lg, -1)
            ref = lp[torch.arange(len(ids) - 1, device=dev), torch.tensor(ids[1:], device=dev)].cpu()
            got = torch.tensor([o.prompt_logprobs[i][ids[i]].logprob for i in range(1, len(ids))])
            d = (ref - got).abs()
            worst = max(worst, d.max().item())
            means.append(d.mean().item())
        if was:
            model.train()
        return dict(sync_max_abs_logprob_diff=worst, sync_mean_abs_logprob_diff=sum(means) / len(means))

    def check_params(self, model):
        """Exact weight check against the engine's own tensors (the primary synchronization gate; see verified_sync)."""
        return compare_engine_state(expected_engine_state(model.state_dict()), dict(self._model().named_parameters()))

    def generate(self, prompts, n=1, temperature=0.0, max_tokens=256, stop=None, seed=0, want_logprobs=False):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=max_tokens, stop=list(stop) if stop else None,
                            include_stop_str_in_output=bool(stop), seed=seed if temperature > 0 else None, logprobs=1 if want_logprobs else None)
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        return [[Completion(c.text, list(c.token_ids), "stop" if c.finish_reason == "stop" else "length",
                            [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)] if want_logprobs else []) for c in o.outputs] for o in outs]


def expected_engine_state(state):
    """The bf16 tensors the engine should hold after load_weights, keyed by vLLM 0.7.2's Qwen2 parameter names.

    vLLM merges q/k/v into qkv_proj and gate/up into gate_up_proj (concatenation along dim 0) and has no lm_head when embeddings are tied.
    Assumes tensor_parallel_size=1 and a vocab size divisible by 64 (Qwen2.5-Math: 151936), so there are no padding rows.
    """
    emb, head = state.get("model.embed_tokens.weight"), state.get("lm_head.weight")
    tied = emb is not None and head is not None and (head is emb or torch.equal(head, emb))
    bf = {k: v.detach().to(torch.bfloat16).cpu() for k, v in state.items() if v.is_floating_point() and not (tied and k == "lm_head.weight")}
    out, used = {}, set()
    for k in bf:
        for merged, parts in (("self_attn.qkv_proj", ("q_proj", "k_proj", "v_proj")), ("mlp.gate_up_proj", ("gate_proj", "up_proj"))):
            for leaf in ("weight", "bias"):
                tail = f".{parts[0]}.{leaf}"
                if k.endswith(tail):
                    pre = k[:-len(tail)]
                    names = [f"{pre}.{x}.{leaf}" for x in parts]
                    if all(n in bf for n in names):
                        out[f"{pre}.{merged.split('.')[-1]}.{leaf}"] = torch.cat([bf[n] for n in names], 0)
                        used.update(names)
    out.update({k: v for k, v in bf.items() if k not in used})
    return out


def compare_engine_state(expected, actual):
    """Exact, tolerance-free check that the engine holds the policy's weights (rounded to bf16).

    A log-prob probe cannot replace this: it mixes weight errors with bf16 and kernel noise, and on the pilot that noise (about 0.03) is
    as large as several optimizer steps.
    """
    bad, missing, extra = [], [k for k in expected if k not in actual], [k for k in actual if k not in expected]
    for k, v in expected.items():
        if k in missing:
            continue
        a = actual[k].detach().cpu()  # not converted: an engine holding another dtype is itself a mismatch
        if a.dtype != v.dtype or a.shape != v.shape or not torch.equal(a, v):
            bad.append(k)
    return dict(params_checked=len(expected), params_mismatched=len(bad) + len(missing) + len(extra), mismatch_names=(bad + missing + extra)[:10],
                missing=len(missing), unexpected=len(extra))


def verified_sync(eng, model):
    """Sync, then prove it: the PRIMARY gate. Every engine parameter must equal the policy's parameter converted to the engine dtype, bit for bit.

    Called before every rollout batch and every evaluation. No tolerance, and a check that compared nothing is a failure.
    """
    s = eng.sync(model)
    t = time.time()
    c = eng.check_params(model)
    if c["params_mismatched"] or (not c["params_checked"] and not getattr(eng, "shares_policy", False)):
        raise SystemExit(f"weight-sync check failed: {c['params_mismatched']} of {c['params_checked']} engine parameters do not match the policy: {c['mismatch_names']}")
    return dict(s, params_checked=c["params_checked"], params_mismatched=0, sync_verify_s=time.time() - t)


def make_engine(cfg, model_path, model, tok):
    e = cfg["engine"]
    if e["name"] == "hf":
        return HFEngine(model, tok, e.get("batch_size", 16))
    if e["name"] == "vllm":
        return VLLMEngine(model_path, e.get("device", "cuda:0"), e.get("gpu_mem", 0.8), e.get("max_model_len", 2048),
                          e.get("prefix_caching", False), cfg.get("seed", 2026))
    raise ValueError(e["name"])
