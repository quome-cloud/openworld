"""Verified wavelet transform + denoising (the signal-perception boundary).

A wavelet denoiser is a clean instance of the framework's perception loop: a
noisy continuous signal is transformed into a SPARSE symbolic state (the few
significant wavelet coefficients - real structure separated from dense noise),
a declared, auditable rule shrinks the noise coefficients, and the inverse
transform emits the cleaned signal. No training, signal-agnostic, exact.

This module is numpy-only (no PyWavelets) and uses the orthonormal Haar wavelet,
which gives bit-exact perfect reconstruction (a built-in invariant). The
multi-resolution decomposition is a composite of scale sub-bands. (A smoother
wavelet - db4/symN - would improve perceptual audio quality; Haar is used here
for a verified, dependency-free, exactly-reconstructing core.)

Denoising rule: soft-threshold the detail coefficients at the universal threshold
lambda = sigma * sqrt(2 ln N), with sigma = MAD(finest details)/0.6745 (the
robust noise estimate of Donoho & Johnstone). Declared and readable.

Additive module: new functions only.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

SQRT2 = math.sqrt(2.0)


def _dwt1(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One Haar level: returns (approx, detail). Pads odd length by edge repeat."""
    if len(x) % 2:
        x = np.append(x, x[-1])
    even, odd = x[0::2], x[1::2]
    return (even + odd) / SQRT2, (even - odd) / SQRT2


def _idwt1(a: np.ndarray, d: np.ndarray, n: int) -> np.ndarray:
    """Invert one Haar level back to length n."""
    even = (a + d) / SQRT2
    odd = (a - d) / SQRT2
    x = np.empty(2 * len(a))
    x[0::2], x[1::2] = even, odd
    return x[:n]


def dwt(x: np.ndarray, levels: int) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
    """Multi-level Haar DWT. Returns (final approx, [detail_1..detail_L], lengths)
    where lengths records each level's pre-pad length for exact inversion."""
    x = np.asarray(x, dtype=float)
    details, lengths = [], []
    a = x
    for _ in range(levels):
        lengths.append(len(a))
        a, d = _dwt1(a)
        details.append(d)
    return a, details, lengths


def idwt(a: np.ndarray, details: List[np.ndarray], lengths: List[int]) -> np.ndarray:
    """Invert a multi-level Haar DWT (exact)."""
    for d, n in zip(reversed(details), reversed(lengths)):
        a = _idwt1(a, d, n)
    return a


def estimate_sigma(detail: np.ndarray) -> float:
    """Robust noise std from the finest detail band (MAD / 0.6745)."""
    return float(np.median(np.abs(detail)) / 0.6745)


def soft_threshold(c: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(c) * np.maximum(np.abs(c) - lam, 0.0)


def denoise(x: np.ndarray, levels: int = 6) -> np.ndarray:
    """Wavelet shrinkage denoise: decompose, soft-threshold details at the
    universal threshold (noise estimated from the finest band), reconstruct."""
    x = np.asarray(x, dtype=float)
    a, details, lengths = dwt(x, levels)
    sigma = estimate_sigma(details[0])
    lam = sigma * math.sqrt(2.0 * math.log(max(2, len(x))))
    cleaned = [soft_threshold(d, lam) for d in details]
    return idwt(a, cleaned, lengths)


def sparsity(x: np.ndarray, levels: int = 6) -> float:
    """Fraction of detail coefficients that are effectively zero after shrinkage
    (the compactness of the symbolic state)."""
    _, details, lengths = dwt(np.asarray(x, float), levels)
    sigma = estimate_sigma(details[0])
    lam = sigma * math.sqrt(2.0 * math.log(max(2, len(x))))
    flat = np.concatenate([soft_threshold(d, lam) for d in details])
    return float(np.mean(flat == 0.0))
