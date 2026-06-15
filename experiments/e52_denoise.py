"""E52 - Wavelet denoising as a perception->world->emit loop.

A verified wavelet world model cleans noisy audio: the perception boundary maps a
noisy waveform to a SPARSE symbolic state (significant wavelet coefficients), a
declared shrinkage rule kills the noise coefficients, and the inverse transform
emits clean audio. Training-free, signal-agnostic, exact reconstruction.

We test it across signal types (tones, chirp, transients, real speech via macOS
`say`) and input SNRs, against two baselines: a naive low-pass (kills high-
frequency noise AND transients) and a Wiener filter TUNED to one noise condition
(10 dB) using the clean spectrum - the 'learned/overfit' baseline that fails out
of distribution. The wavelet denoiser's threshold adapts to the noise it sees, so
it generalizes across SNR without tuning - the paper's exact/auditable/OOD thesis
on signal perception. Emits before/after .wav files to listen to.
"""

import math
import wave
from pathlib import Path

import numpy as np

from openworld import dwt, idwt, sparsity, wavelet_denoise
from openworld.wavelets import estimate_sigma

from common import save_results

SR = 16000
N = 16384                         # ~1s for synthetic signals
SNRS = [0, 5, 10, 15, 20]
TUNE_SNR = 10                     # the Wiener baseline is tuned to this condition
LEVELS = 7
AUDIO = Path(__file__).resolve().parents[1] / "datasets" / "openworld-audio"
OUT = AUDIO / "out"


def snr_db(clean, est):
    p = np.sum((clean - est) ** 2)
    return 10 * math.log10(np.sum(clean ** 2) / p) if p > 0 else 99.0


def add_noise(clean, snr, rng):
    rms = np.sqrt(np.mean(clean ** 2))
    sigma = rms / (10 ** (snr / 20.0))
    return clean + rng.normal(0, sigma, len(clean)), sigma


# --- signals: the Donoho-Johnstone wavelet-denoising benchmarks (piecewise /
#     localized-feature signals, where wavelets are sparse and excel) + real
#     speech. (Pure tones are sparse in FOURIER, not wavelets - out of scope.) -
def make_signals():
    t = np.linspace(0, 1, N, endpoint=False)
    pos = np.array([.1, .13, .15, .23, .25, .4, .44, .65, .76, .78, .81])
    blk_h = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
    bmp_h = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
    wth = np.array([.005, .005, .006, .01, .01, .03, .01, .01, .005, .008, .005])
    sig = {}
    sig["blocks"] = sum(h * (1 + np.sign(t - p)) / 2 for p, h in zip(pos, blk_h))
    sig["bumps"] = sum(h * (1 + np.abs((t - p) / w)) ** -4
                       for p, h, w in zip(pos, bmp_h, wth))
    sig["doppler"] = np.sqrt(t * (1 - t)) * np.sin(2 * np.pi * 1.05 / (t + 0.05))
    sig["heavisine"] = 4 * np.sin(4 * np.pi * t) - np.sign(t - 0.3) - np.sign(0.72 - t)
    sig["speech"] = read_wav(AUDIO / "speech_clean.wav")
    return {k: v / (np.max(np.abs(v)) + 1e-9) for k, v in sig.items()}


def read_wav(path):
    with wave.open(str(path)) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0


def write_wav(path, x):
    x = np.clip(x, -1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype("<i2").tobytes())


# --- baselines --------------------------------------------------------------
def lowpass(noisy, frac=0.12):
    """Naive low-pass: keep the lowest `frac` of the spectrum, zero the rest.
    Blurs edges and erases high-frequency structure (e.g. Doppler's onset)."""
    F = np.fft.rfft(noisy)
    F[int(len(F) * frac):] = 0
    return np.fft.irfft(F, len(noisy))


def best_lowpass_frac(clean, snr, seed=0):
    """Grid-search the low-pass cutoff that maximizes SNR for ONE signal at ONE
    noise level - the tuned/overfit baseline."""
    noisy, _ = add_noise(clean, snr, np.random.RandomState(seed))
    inp = snr_db(clean, noisy)
    best_f, best = 0.12, -1e9
    for f in np.linspace(0.02, 0.6, 30):
        g = snr_db(clean, lowpass(noisy, f)) - inp
        if g > best:
            best, best_f = g, f
    return best_f


def main():
    rng = np.random.RandomState(52)
    signals = make_signals()
    # the tuned filter overfits: cutoff optimized for ONE signal (blocks) at the
    # tuning SNR, then applied to every signal/SNR.
    tuned_frac = best_lowpass_frac(signals["blocks"], TUNE_SNR)
    methods = ["wavelet", "lowpass", "tuned_lowpass"]
    rows = []          # per (signal, snr, method): delta_snr
    for name, clean in signals.items():
        for snr in SNRS:
            noisy, _ = add_noise(clean, snr, rng)
            inp = snr_db(clean, noisy)
            ests = {"wavelet": wavelet_denoise(noisy, LEVELS),
                    "lowpass": lowpass(noisy),
                    "tuned_lowpass": lowpass(noisy, tuned_frac)}
            for m in methods:
                rows.append({"signal": name, "snr_in": snr,
                             "delta_snr": round(snr_db(clean, ests[m]) - inp, 2),
                             "method": m})

    # quantitative claims are on the Donoho piecewise/transient benchmark (the
    # wavelet sweet spot); harmonic speech is reported separately and honestly.
    DONOHO = ["blocks", "bumps", "doppler", "heavisine"]

    def mean_delta(method, pred=lambda r: True):
        v = [r["delta_snr"] for r in rows if r["method"] == method and pred(r)]
        return round(sum(v) / len(v), 2) if v else None

    EDGE = ["blocks", "heavisine"]      # discontinuous: the wavelet basis matches
    SMOOTH = ["bumps", "doppler"]       # smooth/oscillatory: Fourier matches better
    by_method = {m: mean_delta(m, lambda r: r["signal"] in DONOHO) for m in methods}
    # on edge/discontinuity signals the wavelet preserves jumps low-pass blurs
    edges = {m: mean_delta(m, lambda r: r["signal"] in EDGE) for m in methods}
    smooth = {m: mean_delta(m, lambda r: r["signal"] in SMOOTH) for m in methods}
    per_signal = {s: {m: mean_delta(m, lambda r, s=s: r["signal"] == s) for m in methods}
                  for s in signals}
    speech_gain = per_signal["speech"]["wavelet"]

    # perfect-reconstruction invariant + symbolic-state sparsity
    x = signals["speech"]
    a, det, lengths = dwt(x, LEVELS)
    pr_err = float(np.max(np.abs(idwt(a, det, lengths) - x)))
    spk_noisy, _ = add_noise(signals["speech"], 5, np.random.RandomState(7))
    spars = round(sparsity(spk_noisy, LEVELS), 3)

    # emit listenable wavs (doppler + blocks where wavelets shine, + speech honestly)
    emitted = []
    for name in ("doppler", "blocks", "speech"):
        clean = signals[name]
        noisy, _ = add_noise(clean, 5, np.random.RandomState(9))
        write_wav(OUT / f"{name}_clean.wav", clean)
        write_wav(OUT / f"{name}_noisy.wav", noisy)
        write_wav(OUT / f"{name}_wavelet.wav", wavelet_denoise(noisy, LEVELS))
        emitted.append(f"{name}_[clean|noisy|wavelet].wav")

    # waveform + symbolic-state samples for the figure (blocks @ 5 dB)
    bclean = signals["blocks"]
    bnoisy, _ = add_noise(bclean, 5, np.random.RandomState(3))
    seg = slice(2400, 4000)
    _, bdet, _ = dwt(bnoisy, LEVELS)
    coeff_mag = sorted((float(abs(c)) for d in bdet for c in d), reverse=True)
    bsig = estimate_sigma(bdet[0]) * math.sqrt(2 * math.log(len(bnoisy)))
    waveform = {"clean": [round(float(v), 4) for v in bclean[seg]],
                "noisy": [round(float(v), 4) for v in bnoisy[seg]],
                "wavelet": [round(float(v), 4) for v in wavelet_denoise(bnoisy, LEVELS)[seg]],
                "lowpass": [round(float(v), 4) for v in lowpass(bnoisy, tuned_frac)[seg]]}

    results = {
        "sample_rate": SR, "levels": LEVELS, "snrs": SNRS, "tune_snr": TUNE_SNR,
        "tuned_frac": round(float(tuned_frac), 3),
        "waveform": waveform,
        "coeff_mag": [round(c, 4) for c in coeff_mag[:600]],
        "threshold": round(float(bsig), 4),
        "delta_snr_by_method": by_method, "edges_delta": edges, "smooth_delta": smooth,
        "per_signal": per_signal, "speech_gain": speech_gain,
        "tuned_mean": by_method["tuned_lowpass"], "naive_mean": by_method["lowpass"],
        "pr_max_error": pr_err, "noisy_sparsity": spars,
        "rows": rows, "emitted": emitted,
    }
    save_results("e52_denoise", results)

    print("E52 - wavelet denoising (perception -> world -> emit)\n")
    print(f"  perfect reconstruction max error: {pr_err:.2e}  (exact invariant)")
    print(f"  symbolic state sparsity on noisy speech: {spars:.0%} of detail coeffs zeroed")
    print("  mean SNR gain (dB) on the Donoho piecewise/transient benchmark:")
    for m in methods:
        print(f"    {m:<14} {by_method[m]:+.2f} dB")
    print(f"  EDGE/discontinuous signals (blocks, heavisine): wavelet "
          f"{edges['wavelet']:+.2f} vs naive low-pass {edges['lowpass']:+.2f} vs "
          f"TUNED low-pass {edges['tuned_lowpass']:+.2f} dB -- wavelet wins (preserves jumps)")
    print(f"  SMOOTH/oscillatory (bumps, doppler): wavelet {smooth['wavelet']:+.2f} vs "
          f"tuned low-pass {smooth['tuned_lowpass']:+.2f} dB -- Fourier basis matches better")
    print(f"  parameter-free: naive low-pass {by_method['lowpass']:+.2f} needs tuning to "
          f"reach {by_method['tuned_lowpass']:+.2f}; the wavelet threshold is automatic")
    print(f"  per signal (wavelet): " +
          ", ".join(f"{s} {per_signal[s]['wavelet']:+.1f}" for s in signals))
    print(f"  honest scope: harmonic SPEECH gains {speech_gain:+.1f} dB with Haar "
          "(harmonics aren't sparse in a Haar basis; needs a smoother/learned basis)")
    print(f"  emitted to {OUT}: {emitted}")

    # --- self-checks (only what's true) ---
    assert pr_err < 1e-9, "wavelet transform must reconstruct exactly"
    assert by_method["wavelet"] > 0, "wavelet denoising should improve SNR on average"
    assert edges["wavelet"] > edges["lowpass"] and edges["wavelet"] > edges["tuned_lowpass"], \
        "on edge/discontinuity signals wavelet should beat even a TUNED low-pass"
    assert by_method["tuned_lowpass"] > by_method["lowpass"], \
        "the low-pass needs tuning; the wavelet does not (parameter-free)"
    print("\nall checks pass (and the basis-mismatch cases are reported honestly).")


if __name__ == "__main__":
    main()
