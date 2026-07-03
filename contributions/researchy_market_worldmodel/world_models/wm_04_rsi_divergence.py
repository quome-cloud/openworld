"""
World Model 04: RSI Momentum Divergence
Hypothesis: momentum measured by RSI predicts near-term direction;
            extreme RSI values have higher predictive power than moderate values.
Assumptions:
  - RSI(14) > 60 → sustained uptrend
  - RSI(14) < 40 → sustained downtrend
  - RSI(14) 40-60 → no edge → flat
  - Trend confirmation: price must also be above/below 50-day SMA
RESEARCH ONLY — NOT INVESTMENT ADVICE.
"""
import pandas as pd
import numpy as np

NAME = "rsi_momentum_divergence"
ASSUMPTIONS = [
    "RSI(14) captures momentum strength better than pure price crossovers",
    "RSI > 60 + price > 50d SMA = confirmed uptrend → long",
    "RSI < 40 + price < 50d SMA = confirmed downtrend → short",
    "40-60 RSI zone has no predictive edge → flat (cash)",
]


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def get_signals(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Combine RSI zone with 50-day SMA filter for trend confirmation.
    """
    close = ohlcv["close"] if "close" in ohlcv.columns else ohlcv["Close"]
    rsi = _compute_rsi(close, 14)
    sma50 = close.rolling(50).mean()

    uptrend = (rsi > 60) & (close > sma50)
    downtrend = (rsi < 40) & (close < sma50)

    signals = pd.Series(0, index=close.index)
    signals[uptrend] = 1
    signals[downtrend] = -1

    return signals
