"""Generate the real speech sample for E52 using macOS `say` (no network/deps).

Produces speech_clean.wav (16 kHz mono PCM), committed so the experiment runs
offline. `say` synthesizes a genuine speech waveform (a complex real signal, not
synthetic tones), and we keep the clean version as ground truth so denoising is
still scorable.

  python datasets/openworld-audio/make_speech.py
"""
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent / "speech_clean.wav"
TEXT = ("The quick brown fox jumps over the lazy dog. "
        "World models perceive, denoise, and emit clean signals.")

if __name__ == "__main__":
    subprocess.run(["say", "-o", str(OUT), "--data-format=LEI16@16000", TEXT], check=True)
    print(f"wrote {OUT}")
