"""
World Model 06: GDELT Market Sentiment Tone
Hypothesis: aggregate news tone over financial/economic articles predicts
            next-day equity direction (positive tone → long; negative → short).
Assumptions:
  - GDELT GKG tone averaged over ECON_STOCK_MARKET themed articles is a
    meaningful daily sentiment signal for US equity markets
  - Tone signal has 1-day lead on price (today's news → tomorrow's signal)
  - Z-score normalization stabilizes cross-period comparisons
  - Threshold: |z| > 0.5 to avoid noise trading on weak signals
Source: GDELT GKG v1 (gdeltproject.org) — deterministic regex-theme filter
RESEARCH ONLY — NOT INVESTMENT ADVICE.
"""
import pandas as pd
import numpy as np
from pathlib import Path

NAME = "gdelt_tone_sentiment"
ASSUMPTIONS = [
    "GDELT ECON_STOCK_MARKET theme articles capture market-relevant news",
    "Positive tone (z > 0.5) → long; Negative tone (z < -0.5) → short",
    "1-day lag applied to avoid lookahead bias",
    "Monthly sampled, forward-filled — coarse signal",
]

_SENTIMENT_PATH = Path(__file__).parent.parent / "data" / "sentiment" / "daily_sentiment.csv"


def _load_sentiment() -> pd.DataFrame | None:
    """Load pre-downloaded GDELT sentiment data. Returns None if not yet downloaded."""
    if not _SENTIMENT_PATH.exists():
        return None
    df = pd.read_csv(_SENTIMENT_PATH, parse_dates=["date"], index_col="date")
    return df


def get_signals(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Generate {-1, 0, 1} signals from GDELT market tone.
    Falls back to all-zero (flat/cash) if sentiment data not downloaded.
    """
    sentiment = _load_sentiment()

    if sentiment is None:
        print(f"  WARNING: Sentiment data not found at {_SENTIMENT_PATH}")
        print("  Run: python download_sentiment.py")
        print("  Returning flat signal (0 everywhere) — not a real backtest")
        return pd.Series(0, index=ohlcv.index)

    # Align sentiment to OHLCV dates
    tone = sentiment["gdelt_tone_z"].reindex(ohlcv.index).ffill().bfill()

    # 1-day lag (today's news → tomorrow's position)
    tone_lag = tone.shift(1)

    # Threshold-based signal
    signals = pd.Series(0, index=ohlcv.index)
    signals[tone_lag > 0.5] = 1
    signals[tone_lag < -0.5] = -1

    return signals
