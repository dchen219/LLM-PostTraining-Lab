from math import comb, sqrt


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / 2 / n) / d
    h = z * sqrt(p * (1 - p) / n + z * z / 4 / n / n) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    m = b + c
    return 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, k) for k in range(min(b, c) + 1)) / 2 ** m)


def paired(a, b):
    """a, b: {item id: bool} on the SAME items. 2x2 table, b - a difference with CI, exact McNemar p."""
    if a.keys() != b.keys():
        raise ValueError("paired comparison needs identical item ids")
    both = sum(a[i] and b[i] for i in a)
    ao = sum(a[i] and not b[i] for i in a)
    bo = sum(b[i] and not a[i] for i in a)
    n = len(a)
    d = (bo - ao) / n
    se = sqrt(max(0.0, (ao + bo) - (bo - ao) ** 2 / n)) / n
    return dict(n=n, both=both, a_only=ao, b_only=bo, neither=n - both - ao - bo, diff=d,
                ci95=(d - 1.96 * se, d + 1.96 * se), p=mcnemar_exact(ao, bo))


def holm(ps):
    """Holm-Bonferroni adjusted p-values for {name: p}."""
    out, run = {}, 0.0
    for i, (k, p) in enumerate(sorted(ps.items(), key=lambda t: t[1])):
        run = max(run, min(1.0, (len(ps) - i) * p))
        out[k] = run
    return out
