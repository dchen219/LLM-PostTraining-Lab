import statistics as st
import torch


def group_stats(rewards, group_size):
    g = [rewards[i:i + group_size] for i in range(0, len(rewards), group_size)]
    var = [st.pvariance(x) for x in g]
    n = len(g)
    zero = sum(v == 0 for v in var)
    return dict(groups=n, all_correct=sum(min(x) >= 1 for x in g), all_incorrect=sum(max(x) <= 0 for x in g), zero_adv=zero,
                informative_rate=1 - zero / n, reward_mean=sum(rewards) / len(rewards), reward_var=st.pvariance(rewards),
                within_group_var=sum(var) / n)


def rollout_stats(comps, scores):
    L = [len(c.token_ids) for c in comps]
    ok = [s["answer_reward"] == 1 for s in scores]
    mean = lambda x: sum(x) / len(x) if x else None
    return dict(len_mean=mean(L), len_correct=mean([x for x, o in zip(L, ok) if o]), len_incorrect=mean([x for x, o in zip(L, ok) if not o]),
                len_max=max(L), truncated_rate=mean([c.finish == "length" for c in comps]),
                format_rate=mean([s["format_reward"] for s in scores]), answer_rate=mean([s["answer_reward"] for s in scores]),
                verifier_errors=sum(bool(s.get("error")) for s in scores))


def gpu_mem():
    return {f"gpu{i}_peak_gib": round(torch.cuda.max_memory_allocated(i) / 2**30, 2) for i in range(torch.cuda.device_count())}


def reset_peaks():
    for i in range(torch.cuda.device_count()):  # reset_peak_memory_stats() alone only touches the current device
        torch.cuda.reset_peak_memory_stats(i)


class ParamProbe:
    """Proves that optimizer steps move parameters. Samples up to 6 tensors; also reports how much of the change survives
    bf16 rounding, i.e. what the bf16 vLLM copy can see after a sync."""

    def __init__(self, model, tensors=6, elems=20000):
        ps = [p for p in model.parameters() if p.requires_grad and p.dim() > 1]
        self.ps = ps[::max(1, len(ps) // tensors)][:tensors]
        self.elems, self.old = elems, None

    def _take(self):
        return [p.detach().flatten()[:self.elems].clone() for p in self.ps]

    def before(self):
        self.old = self._take()

    def after(self):
        n = ch = bf = 0
        delta = 0.0
        for a, b in zip(self.old, self._take()):
            n += a.numel()
            ch += int((a != b).sum())
            bf += int((a.to(torch.bfloat16) != b.to(torch.bfloat16)).sum())
            delta += (a - b).abs().sum().item()
        return dict(param_changed_frac=ch / n, param_changed_frac_bf16=bf / n, param_mean_abs_delta=delta / n)


@torch.no_grad()
def adam_update_stats(opt, chunk=1 << 24):
    """Size of the AdamW step just taken, over ALL parameters, from the optimizer state (no weight copy needed).

    PyTorch AdamW: p <- p - lr*wd*p - lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps). Exact up to rounding (tests/test_posttrain.py).
    Adam normalises the gradient, so the step size is set by lr and by m/sqrt(v), not by the gradient norm.
    """
    u2 = p2 = m2 = mx = sq = 0.0
    n = 0
    for g in opt.param_groups:
        if g.get("amsgrad") or g.get("maximize"):
            raise ValueError("adam_update_stats assumes plain AdamW")
        if "betas" not in g:  # not Adam (tests use SGD): nothing to report
            return {}
        b1, b2 = g["betas"]
        for p in g["params"]:
            s = opt.state.get(p)
            if not s:
                continue
            t = float(s["step"])
            c1, c2 = 1 - b1 ** t, (1 - b2 ** t) ** 0.5
            for pc, mc, vc in zip(p.detach().view(-1).split(chunk), s["exp_avg"].view(-1).split(chunk), s["exp_avg_sq"].view(-1).split(chunk)):
                den = vc.float().sqrt() / c2 + g["eps"]
                u = g["lr"] * (mc.float() / c1) / den + g["lr"] * g["weight_decay"] * pc.float()
                u2 += float(u.pow(2).sum())
                mx = max(mx, float(u.abs().max()))
                p2 += float(pc.float().pow(2).sum())
                m2 += float(mc.float().pow(2).sum())
                sq += float((den - g["eps"]).sum())
                n += pc.numel()
    if not n:
        return {}
    return dict(update_l2=u2 ** 0.5, update_rms=(u2 / n) ** 0.5, update_max_abs=mx, param_l2=p2 ** 0.5, update_rel=(u2 / max(p2, 1e-30)) ** 0.5,
                adam_m_l2=m2 ** 0.5, adam_sqrt_v_mean=sq / n)


class DifficultySampler:
    """Gated extension: sample prompts in proportion to p(1-p) of an EMA pass rate (binary-reward variance) plus a floor."""

    def __init__(self, ids, rng, enabled=False, floor=0.1, ema=0.5):
        self.ids, self.rng, self.on, self.floor, self.ema = list(ids), rng, enabled, floor, ema
        self.p = {}

    def weight(self, i):
        return self.p[i] * (1 - self.p[i]) + self.floor * 0.25 if i in self.p else 0.25

    def sample(self, k):
        if not self.on:
            return self.rng.sample(self.ids, k)
        return sorted(self.ids, key=lambda i: self.rng.random() ** (1 / self.weight(i)), reverse=True)[:k]

    def update(self, ids, rates):
        for i, r in zip(ids, rates):
            self.p[i] = r if i not in self.p else self.ema * r + (1 - self.ema) * self.p[i]
