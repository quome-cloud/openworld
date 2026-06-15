# E52 - Wavelet denoising as a perception->world->emit loop

**Date:** 2026-06-14
**Status:** approved (design)

## Goal

Show a wavelet denoiser as a literal instance of the perception boundary: noisy
signal -> sparse symbolic state (significant wavelet coefficients) -> declared
shrinkage rule -> inverse transform -> clean signal emitted. Verified, exact,
training-free, auditable - and honest about where the basis matches the signal.

## Module

`openworld/wavelets.py` (numpy-only): orthonormal Haar multi-level DWT/IDWT with
bit-exact perfect reconstruction (a built-in invariant), universal soft-threshold
denoise (sigma = MAD(finest)/0.6745, lambda = sigma sqrt(2 ln N)), and a sparsity
measure. Multi-resolution = a composite of scale sub-bands. (Haar is the verified,
dependency-free core; a smoother db/sym wavelet would improve smooth/harmonic
signals - noted, not hidden.)

## Experiment (`experiments/e52_denoise.py`)

Signals: the Donoho-Johnstone benchmarks (blocks, bumps, doppler, heavisine) +
real speech via macOS `say`. Add calibrated AWGN across SNRs. Methods: wavelet
(universal threshold, parameter-free), naive low-pass, and a low-pass with its
cutoff TUNED to one signal/SNR. Metrics: SNR gain (dB) vs clean. Emits
before/after `.wav` for listening.

## Results (honest, basis-match)

1. Perfect reconstruction is exact (invariant); the symbolic state is ~97% zero.
2. On EDGE/discontinuous signals (blocks, heavisine) the parameter-free wavelet
   beats even an optimally tuned low-pass (it preserves jumps low-pass blurs).
3. Parameter-free advantage: the low-pass needs tuning to be competitive; the
   wavelet's threshold is automatic.
4. On SMOOTH/oscillatory signals (bumps, doppler) and HARMONIC speech the Haar
   basis is the wrong match (Fourier wins; speech gain is negative) - reported,
   not hidden.

## Deliverables

`openworld/wavelets.py` + exports + `tests/test_wavelets.py`;
`experiments/e52_denoise.py` (+ results); `datasets/openworld-audio/`
(make_speech.py, speech_clean.wav, emitted out/*.wav); figure + table + paper
subsection `sec:denoise`; `\NumExperiments` -> 51. PR targets `main`.

## Honest boundaries

Haar is a crude wavelet (good for piecewise/discontinuous, poor for harmonic);
the claim is about the perception->world->emit framing, exact/auditable/
parameter-free denoising, and basis-structure match - not state-of-the-art audio
denoising (modern speech denoisers are learned/spectral). Not the same as a DNN
denoiser; complementary.
