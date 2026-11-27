"""Micro-batch partitioning shared by SFT, GRPO updates and old-policy log-probs.

Only the partition changes; the loss is always divided by a whole-update denominator (tokens for SFT, rollouts for GRPO), so any partition
gives the same gradient up to floating-point rounding. Memory, not correctness, is what the partition controls.

Why it exists: in the pilot, policy-GPU peak memory was an almost exact linear function of the largest padded micro-batch
(results/pilot_analysis/memory_analysis.json). Fixed sequence counts ignore length, so a few long rollouts set the peak.
A padded-token budget bounds exactly the quantity that drives memory.
"""


def microbatches(order, length, micro_batch=None, token_budget=None):
    """Split indices into consecutive micro-batches. `length[i]` is the padded length of item i if it is alone in a batch.

    token_budget None: legacy fixed-size chunks of `micro_batch`, in the given order (identical to the pilot's behaviour).
    token_budget set:  items are sorted by length; each micro-batch has at most `token_budget` padded tokens
                       (size x longest member) and at most `micro_batch` items (if given). A single item longer than the budget
                       gets its own micro-batch: splitting it would need a different loss.
    """
    if token_budget is None:
        if not micro_batch:
            raise ValueError("micro_batch is required when token_budget is not set")
        return [list(order[i:i + micro_batch]) for i in range(0, len(order), micro_batch)]
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    out, cur, longest = [], [], 0
    for i in sorted(order, key=lambda j: (length[j], j)):
        n, t = len(cur) + 1, max(longest, length[i])
        if cur and (n * t > token_budget or (micro_batch and n > micro_batch)):
            out.append(cur)
            cur, longest = [], 0
            n, t = 1, length[i]
        cur.append(i)
        longest = max(longest, length[i])
    if cur:
        out.append(cur)
    return out


def padded_tokens(parts, length):
    """Padded tokens of each micro-batch; the quantity that predicts policy-GPU memory."""
    return [len(p) * max(length[i] for i in p) for p in parts]


def seq_len(tok, prompt, response, append_eos=False):
    """Token length of prompt + response exactly as posttrain.losses.encode builds it. `response` may be text or sampled token ids."""
    n = len(tok(prompt, add_special_tokens=False)["input_ids"])
    n += len(response) if isinstance(response, (list, tuple)) else len(tok(response, add_special_tokens=False)["input_ids"])
    return n + 1 if append_eos else n
