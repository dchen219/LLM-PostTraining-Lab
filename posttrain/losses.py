import torch
import torch.nn.functional as F


def encode(tok, prompts, responses, device="cpu", append_eos=False, max_len=None):
    """Right-padded batch; response_mask is aligned to labels. Raises instead of silently truncating.

    responses may be strings or lists of token ids (use sampled ids for RL so re-tokenisation cannot change the action).
    """
    seqs = []
    for p, r in zip(prompts, responses):
        a = tok(p, add_special_tokens=False)["input_ids"]
        b = list(r) if isinstance(r, (list, tuple)) else tok(r, add_special_tokens=False)["input_ids"]
        if append_eos and (not b or b[-1] != tok.eos_token_id):
            b = b + [tok.eos_token_id]
        if max_len and len(a) + len(b) > max_len:
            raise ValueError(f"sequence of {len(a) + len(b)} tokens exceeds max_len={max_len}; filter the data explicitly")
        seqs.append((a, b))
    T = max(len(a) + len(b) for a, b in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    att = torch.zeros_like(ids)
    mask = torch.zeros((len(seqs), T), dtype=torch.float32)
    for i, (a, b) in enumerate(seqs):
        n = len(a) + len(b)
        ids[i, :n] = torch.tensor(a + b)
        att[i, :n] = 1
        mask[i, len(a):n] = 1
    return {"input_ids": ids[:, :-1].to(device), "attention_mask": att[:, :-1].to(device),
            "labels": ids[:, 1:].to(device), "response_mask": mask[:, 1:].to(device)}


def token_stats(model, b, entropy=False):
    """log p(label) per position in fp32; entropy is diagnostic only (no gradient)."""
    logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], use_cache=False).logits.float()
    lp = -F.cross_entropy(logits.transpose(1, 2), b["labels"], reduction="none")
    ent = None
    if entropy:
        with torch.no_grad():
            rows = []
            for row in logits.detach():  # one sequence at a time: the full-batch version needs ~3 extra logits-sized buffers
                z = F.log_softmax(row, -1)
                rows.append(-(z.exp() * z).sum(-1))
            ent = torch.stack(rows)
    return lp, ent


def sft_loss_sum(lp, mask):
    """Summed NLL and token count. Dividing by tokens in the whole optimizer batch makes micro-batching exact."""
    return -(lp * mask).sum(), mask.sum()


def group_advantages(rewards, group_size, std_norm=False, eps=1e-6):
    r = torch.as_tensor(rewards, dtype=torch.float32)
    if r.numel() % group_size:
        raise ValueError("rollouts not divisible by group_size")
    g = r.view(-1, group_size)
    a = g - g.mean(1, keepdim=True)
    if std_norm:
        a = a / (g.std(1, keepdim=True, unbiased=True) + eps)
    return a.reshape(-1)


def reduce_seq(x, mask, how, const):
    s = (x * mask).sum(1)
    if how == "mean":
        return s / mask.sum(1).clamp(min=1)
    if how == "const":
        return s / float(const)
    raise ValueError(how)


def policy_loss(kind, lp, mask, adv, rewards=None, old_lp=None, clip=0.2, length_norm="const", const=1024):
    """Per-sequence loss (B,) plus raw diagnostic sums. kind: no_baseline | baseline | grpo_noclip | grpo_clip."""
    m = {}
    if kind == "no_baseline":
        tok = -rewards[:, None] * lp
    elif kind == "baseline":
        tok = -adv[:, None] * lp
    elif kind in ("grpo_clip", "grpo_noclip"):
        ratio = ((lp - old_lp) * mask).exp()
        un = ratio * adv[:, None]
        outside = (((ratio - 1).abs() > clip).float() * mask).sum()
        m["clip_outside_tokens"] = outside.item()
        if kind == "grpo_clip":
            cl = ratio.clamp(1 - clip, 1 + clip) * adv[:, None]
            tok = -torch.minimum(un, cl)
            m["clip_active_tokens"] = ((cl < un).float() * mask).sum().item()
        else:
            tok = -un
        r = ratio.detach()[mask.bool()]
        if r.numel():
            m.update(ratio_sum=r.sum().item(), ratio_min=r.min().item(), ratio_max=r.max().item(),
                     kl_sum=((r - 1) - r.log()).sum().item())
    else:
        raise ValueError(kind)
    return reduce_seq(tok, mask, length_norm, const), m
