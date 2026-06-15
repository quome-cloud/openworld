"""Tests for optimal-transport (Wasserstein) world calibration / drift."""

import numpy as np

from openworld import kl_hist, wasserstein1


def test_wasserstein_zero_on_identical():
    a = np.linspace(0, 1, 100)
    assert wasserstein1(a, a) < 1e-9


def test_wasserstein_proportional_to_shift():
    rng = np.random.RandomState(0)
    base = rng.normal(0, 1, 500)
    d2 = wasserstein1(base, base + 2.0)
    d5 = wasserstein1(base, base + 5.0)
    assert abs(d2 - 2.0) < 0.2          # 1-W of a pure translation is the shift
    assert d5 > d2                      # and it grows with the shift


def test_wasserstein_finite_when_kl_saturates():
    rng = np.random.RandomState(1)
    a = rng.normal(0, 0.5, 400)
    b = rng.normal(8.0, 0.5, 400)       # disjoint support
    w = wasserstein1(a, b)
    assert np.isfinite(w) and w > 5.0   # Wasserstein still finite + large
    # KL with shared bins saturates: the far distribution sees the smoothing floor,
    # so moving it further changes KL negligibly (no usable gradient).
    far = kl_hist(a, rng.normal(8.0, 0.5, 400))
    farther = kl_hist(a, rng.normal(12.0, 0.5, 400))
    assert abs(far - farther) < 0.05 * far
