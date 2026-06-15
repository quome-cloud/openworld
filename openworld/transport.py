"""Optimal transport (Wasserstein) for world calibration and drift detection.

KL divergence is the usual way to compare a world's output distribution to data,
but it breaks when supports barely overlap (it diverges to infinity) - exactly
the situation in a regime shift, where the new distribution has moved off the old
one. The Wasserstein (earth-mover) distance instead measures how far mass has to
move, so it stays finite and is proportional to the size of the shift: it detects
and localizes drift gracefully where KL is undefined, and gives a smooth objective
for calibrating a world to data even when the initial guess misses the support.

1-D Wasserstein has a closed form (the L1 distance between quantile functions);
that is all we need for scalar world outputs. Numpy-only, deterministic.
"""

from __future__ import annotations

import numpy as np


def wasserstein1(a, b, n: int = 200) -> float:
    """1-Wasserstein distance between two empirical samples = mean absolute
    difference of their quantile functions."""
    q = np.linspace(0, 1, n)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def kl_hist(a, b, bins: int = 40, span=(-4.0, 14.0), alpha: float = 1e-3) -> float:
    """KL(a || b) via shared-bin histograms with Laplace smoothing (so it is
    finite within overlapping regimes). It SATURATES to a constant when the
    supports are disjoint - the failure mode: across a regime shift KL gives no
    gradient/signal, only a saturated value, where Wasserstein scales with the
    actual distance moved."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    edges = np.linspace(span[0], span[1], bins + 1)
    pa = np.histogram(a, edges)[0] + alpha
    pb = np.histogram(b, edges)[0] + alpha
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(np.sum(pa * np.log(pa / pb)))
