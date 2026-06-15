"""Tests for interval / affine abstract interpretation."""

import random

from openworld import Affine, Interval


def test_interval_arithmetic_sound():
    x = Interval(1, 3)
    y = Interval(-2, 4)
    rng = random.Random(0)
    for _ in range(2000):
        a = rng.uniform(1, 3); b = rng.uniform(-2, 4)
        assert (x + y).contains(a + b)
        assert (x * y).contains(a * b)


def test_affine_tracks_correlation():
    # a and (1-a) are perfectly anticorrelated; affine keeps that, interval loses it
    a_iv = Interval(0.2, 0.8)
    one_minus_a_iv = Interval(1, 1) + (-1.0) * a_iv
    expr_iv = a_iv + one_minus_a_iv            # should be exactly 1, interval says wide
    a_af = Affine.from_interval(0.2, 0.8)
    expr_af = a_af + (Affine(1.0) + (-1.0) * a_af)
    assert expr_iv.width > 0.5                  # interval wrongly inflates
    assert expr_af.to_interval().width < 1e-9   # affine knows it is exactly 1


def test_affine_sound_contains_samples():
    a = Affine.from_interval(0.5, 0.9)
    expr = a * a + (-1.0) * a                    # a^2 - a
    iv = expr.to_interval()
    rng = random.Random(1)
    for _ in range(2000):
        v = rng.uniform(0.5, 0.9)
        assert iv.contains(v * v - v)
