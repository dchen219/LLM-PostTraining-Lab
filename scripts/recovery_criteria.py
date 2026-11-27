"""The registered diagnostic criteria (recovery/diagnostic_criteria.yaml), evaluated for one branch. Diagnostic only; stops nothing.

evaluate(history, branch, crit, done, failed) -> dict
  history: the original run's train rows up to the fork step (steps 1..80); branch: the branch's train rows (81..)
  done / failed: the branch's DONE.json / FAILED.json content, or None
"""
import statistics as st


def robust_z(series, t_index, window):
    prev = series[t_index - window:t_index]
    if len(prev) < window or any(x is None for x in prev) or series[t_index] is None:
        return None
    m = st.median(prev)
    s = 1.4826 * st.median([abs(x - m) for x in prev])
    if s == 0:
        return float("inf") if series[t_index] > m else (float("-inf") if series[t_index] < m else 0.0)
    return (series[t_index] - m) / s


def evaluate(history, branch, crit, done=None, failed=None):
    c = crit["criteria"]
    rows = sorted(history, key=lambda r: r["step"]) + sorted(branch, key=lambda r: r["step"])
    steps = [r["step"] for r in rows]
    assert steps == list(range(steps[0], steps[0] + len(steps))), "steps must be contiguous (history then branch)"
    first_branch = branch[0]["step"] if branch else None
    ew = c["early_warning"]
    ser = {k: [r.get(k) for r in rows] for k in ew["metrics"]}
    warn, z = [], {}
    for i, r in enumerate(rows):
        if first_branch is None or r["step"] < first_branch:
            continue
        zz = {k: robust_z(ser[k], i, ew["window"]) for k in ew["metrics"]}
        z[r["step"]] = zz
        if all(v is not None and v >= ew["threshold"] for v in zz.values()):
            warn.append(r["step"])
    below = [r["step"] for r in branch if r["reward_mean"] < c["severe_reward_deterioration"]["below"]]
    run_ = best = 0
    prev = None
    for s in below:
        run_ = run_ + 1 if prev is not None and s == prev + 1 else 1
        best, prev = max(best, run_), s
    gate_stop = bool(failed) and "rollout log-prob check failed" in str(failed.get("error", ""))
    sustained = best >= c["sustained_reward_deterioration"]["consecutive"]
    if not sustained and below and gate_stop and below[-1] == branch[-1]["step"]:
        sustained = "not observable (rollout-gate stop right after a low step)"
    last = branch[-1]["step"] if branch else None
    gap_fail = [r["step"] for r in branch if r.get("rollout_logprob_mad") is not None and r["rollout_logprob_mad"] > c["numerical_failure"]["above"]]
    return dict(early_warning=dict(first=warn[0] if warn else None, steps=warn, z={s: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()} for s, d in z.items()}),
                severe_reward_deterioration=dict(hit=bool(below), first=below[0] if below else None, steps=below),
                sustained_reward_deterioration=dict(hit=sustained, longest_run=best),
                numerical_failure=dict(hit=gate_stop or bool(gap_fail), gate_stop=gate_stop, attempted_step=(failed or {}).get("attempted_step") if gate_stop else None),
                successful_completion=dict(hit=bool(done) and not failed and last == c["successful_completion"]["steps"], last_step=last),
                diagnostic_only=True, caveat=crit["caveat"])
