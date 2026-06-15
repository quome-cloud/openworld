"""Tests for the verified wavelet transform + denoiser."""

import math

import numpy as np

from openworld import dwt, idwt, wavelet_denoise


def _snr(clean, est):
    return 10 * math.log10(np.sum(clean ** 2) / np.sum((clean - est) ** 2))


def test_perfect_reconstruction_exact():
    rng = np.random.RandomState(0)
    for n in (256, 257, 1000, 4096):
        x = rng.randn(n)
        a, det, lengths = dwt(x, 6)
        xr = idwt(a, det, lengths)
        assert np.allclose(xr, x, atol=1e-9), f"PR failed at n={n}"


def test_denoise_improves_snr():
    rng = np.random.RandomState(1)
    t = np.linspace(0, 1, 4096, endpoint=False)
    clean = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    noise = rng.normal(0, 0.5, len(t))
    noisy = clean + noise
    est = wavelet_denoise(noisy, levels=6)
    assert _snr(clean, est) > _snr(clean, noisy) + 2.0, "denoising should raise SNR"


def test_denoise_noop_on_clean_is_gentle():
    # on an already-clean smooth signal, denoising shouldn't destroy it
    t = np.linspace(0, 1, 2048, endpoint=False)
    clean = np.sin(2 * np.pi * 3 * t)
    est = wavelet_denoise(clean, levels=5)
    assert _snr(clean, est) > 20.0, "clean signal should survive denoising"
