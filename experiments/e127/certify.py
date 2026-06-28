"""Equivalence-to-real certificate: a Clopper-Pearson lower bound on held-out next-frame accuracy,
plus per-level coverage. This is the acceptance gate -- NOT two-model agreement. A certificate that
fails still returns its measured numbers (a bound, never a binary 'unified')."""
import math
from experiments.e127 import engine as _engine


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta (Numerical Recipes, modified Lentz)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betai(a, b, x):
    """Regularized incomplete beta I_x(a,b) via the Lentz continued fraction (Numerical Recipes).

    The continued fraction converges fast only for x < (a+1)/(a+b+2); beyond that pivot we use the
    symmetry I_x(a,b) = 1 - I_{1-x}(b,a), evaluating the CF with (a,b) and x both swapped. This is
    the standard branch (the naive 1 - front*betacf(a,b,x) is wrong)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    # bt is symmetric in the log terms; same value for the (a,b,x) and (b,a,1-x) framings.
    bt = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def clopper_pearson_lower(k, n, delta):
    """Lower (1-delta) confidence bound on a binomial proportion with k successes in n trials.
    Defined by P[Bin(n,L) >= k] = delta  <=>  I_L(k, n-k+1) = delta. Solve by bisection in L."""
    if n == 0:
        return 0.0
    if k == 0:
        return 0.0
    if k == n:
        return delta ** (1.0 / n)
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if betai(k, n - k + 1, mid) > delta:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def certify_engine(factory, holdout, n_levels, eps=0.01, delta=0.05, coverage_target=0.8):
    """Score `factory` on a disjoint held-out set of Episodes; emit the certificate dict."""
    n = exact = lv_match = lv_tot = 0
    errored = False
    seen_levels = set()
    for ep in holdout:
        for s in ep:
            seen_levels.add(int(s["levels"]))
        sc = _engine.score_rollout(factory, ep)
        if sc["errored"]:
            errored = True
        n += sc["transitions"]
        exact += sc["exact"]
        lv_match += sc["levelup_match"]
        lv_tot += sc["levelup_total"]
    acc = (exact / n) if n else 0.0
    acc_lower = clopper_pearson_lower(exact, n, delta) if n else 0.0
    coverage = len(seen_levels & set(range(n_levels + 1))) / max(1, n_levels + 1)
    passed = (not errored) and (acc_lower >= 1.0 - eps) and (coverage >= coverage_target)
    return {"pass": bool(passed), "acc": acc, "acc_lower": acc_lower, "n": n, "exact": exact,
            "eps": eps, "delta": delta, "coverage": coverage, "coverage_target": coverage_target,
            "levelup_match": lv_match, "levelup_total": lv_tot, "errored": errored}
