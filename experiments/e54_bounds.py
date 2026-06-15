"""E54 - Abstract interpretation: provable bounds on world outcomes.

A world relaxes toward a target by an exponential moving average with an UNCERTAIN
smoothing factor a (the same a every step) plus per-step shocks:

    x_{t+1} = a*x_t + (1 - a)*c + s_t ,  a in [a_lo, a_hi],  s_t in [-d, d]

We want a GUARANTEE on x_T. Three ways:

  Monte Carlo - sample a and the shocks, simulate, take the observed [min, max].
                UNSOUND: the worst case needs a extreme AND all shocks extreme at
                once (a rare joint corner), so sampling under-covers it.
  Interval    - run over [a_lo, a_hi]. Sound, but it decorrelates: the SAME a
                appears as both `a` and `1-a`, and interval treats them
                independently, so bounds inflate (the wrapping effect).
  Affine      - track a as ONE correlated symbol, so `a` and `1-a` stay coupled;
                sound AND much tighter.

The point: abstract interpretation gives a VERIFIED envelope that always contains
the truth (where Monte Carlo cannot), and the affine domain keeps it tight where
intervals wrap. Deterministic/offline.
"""

import numpy as np

from openworld import Affine, Interval

from common import save_results

A_LO, A_HI = 0.70, 0.95      # uncertain smoothing factor (recurs every step)
C = 20.0                      # target the world relaxes toward
D = 0.5                       # per-step shock magnitude
X0 = 5.0
SEED = 54


def true_range(T, n=4001):
    """Dense ground truth: monotone in a, extremized by all shocks at +/- d."""
    a = np.linspace(A_LO, A_HI, n)
    xmax = np.full(n, X0)
    xmin = np.full(n, X0)
    for _ in range(T):
        xmax = a * xmax + (1 - a) * C + D
        xmin = a * xmin + (1 - a) * C - D
    return float(xmin.min()), float(xmax.max())


def interval_bound(T):
    x = Interval(X0, X0)
    a = Interval(A_LO, A_HI)
    for _ in range(T):
        x = a * x + (Interval(1, 1) + (-1.0) * a) * C + Interval(-D, D)
    return x.lo, x.hi


def affine_bound(T):
    Affine._counter[0] = 0
    x = Affine(X0)
    a = Affine.from_interval(A_LO, A_HI)        # ONE symbol, reused as a and 1-a
    for _ in range(T):
        x = a * x + (Affine(1.0) + (-1.0) * a) * C + Affine.from_interval(-D, D)
    iv = x.to_interval()
    return iv.lo, iv.hi


def monte_carlo(T, samples=4000, seed=SEED):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(samples):
        a = rng.uniform(A_LO, A_HI)
        x = X0
        for _ in range(T):
            x = a * x + (1 - a) * C + rng.uniform(-D, D)
        out.append(x)
    return float(min(out)), float(max(out))


def main():
    horizons = [6, 12, 24, 36]
    rows = []
    for T in horizons:
        tlo, thi = true_range(T)
        ilo, ihi = interval_bound(T)
        alo, ahi = affine_bound(T)
        mlo, mhi = monte_carlo(T)
        rows.append({
            "T": T,
            "true": [round(tlo, 3), round(thi, 3)],
            "interval": [round(ilo, 3), round(ihi, 3)], "interval_width": round(ihi - ilo, 3),
            "affine": [round(alo, 3), round(ahi, 3)], "affine_width": round(ahi - alo, 3),
            "mc": [round(mlo, 3), round(mhi, 3)], "mc_width": round(mhi - mlo, 3),
            "true_width": round(thi - tlo, 3),
            # soundness: does the abstract domain contain the true range?
            "affine_sound": alo <= tlo + 1e-6 and ahi >= thi - 1e-6,
            "interval_sound": ilo <= tlo + 1e-6 and ihi >= thi - 1e-6,
            # MC under-coverage: how much of the true worst case it misses
            "mc_misses_hi": round(thi - mhi, 3),
        })

    save_results("e54_bounds", {
        "a_lo": A_LO, "a_hi": A_HI, "x0": X0, "horizons": horizons, "rows": rows})

    print("E54 - abstract interpretation: provable bounds vs Monte Carlo\n")
    print(f"  {'T':>3}{'true width':>12}{'affine':>10}{'interval':>10}{'MC':>8}"
          f"{'affine snd':>11}{'MC misses hi':>14}")
    for r in rows:
        print(f"  {r['T']:>3}{r['true_width']:>12.2f}{r['affine_width']:>10.2f}"
              f"{r['interval_width']:>10.2f}{r['mc_width']:>8.2f}"
              f"{str(r['affine_sound']):>11}{r['mc_misses_hi']:>14.2f}")

    # --- self-checks ---
    assert all(r["affine_sound"] for r in rows), "affine bounds must contain the truth (sound)"
    assert all(r["interval_sound"] for r in rows), "interval bounds must be sound"
    # MC under-covers the worst case (its max is below the true max)
    assert all(r["mc_misses_hi"] > 0 for r in rows), "Monte Carlo should miss the true worst case"
    # affine is much tighter than interval at long horizons (tracks correlation)
    assert rows[-1]["affine_width"] < 0.6 * rows[-1]["interval_width"], \
        "affine should be far tighter than interval over a long horizon"
    # affine stays close to the true range (a useful, not vacuous, envelope)
    assert rows[-1]["affine_width"] < 2.0 * rows[-1]["true_width"], \
        "affine envelope should be reasonably tight to the truth"
    print("\nchecks pass: sound, tight, verified envelopes; MC silently under-covers.")


if __name__ == "__main__":
    main()
