"""
World Model 02: Bollinger Band Mean Reversion
Hypothesis: short-term price deviations from trend revert; extremes predict reversal.
Assumptions:
  - Returns have mild negative autocorrelation at 1-5 day horizon
  - 2-std Bollinger bands mark statistically extreme deviations
  - Mean reversion signal is stronger for liquid large-caps than micro-caps
RESEARCH ONLY — NOT INVESTMENT ADVICE.
"""
import pandas as pd
import numpy as np

NAME = "mean_reversion_bb"
ASSUMPTIONS = [
    "Short-term return autocorrelation is negative (mean reversion)",
    "Bollinger Band z-score > 2 marks overextension",
    "20-day window + 2-sigma bands are regime-neutral",
]


def get_signals(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Long when price < lower BB (oversold); short when price > upper BB (overbought).
    Flat within bands.
    """
    close = ohlcv["close"] if "close" in ohlcv.columns else ohlcv["Close"]
    window = 20
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std

    signals = pd.Series(0, index=close.index)
    signals[close < lower] = 1   # oversold → expect bounce → long
    signals[close > upper] = -1  # overbought → expect reversal → short

    # Hold for up to 5 days or until price crosses mid
    result = signals.copy()
    held_signal = 0
    held_days = 0
    for i in range(len(signals)):
        if signals.iloc[i] != 0:
            held_signal = signals.iloc[i]
            held_days = 0
        elif held_signal != 0:
            held_days += 1
            price = close.iloc[i]
            crossed_mid = (held_signal == 1 and price >= mid.iloc[i]) or \
                          (held_signal == -1 and price <= mid.iloc[i])
            if crossed_mid or held_days >= 5:
                held_signal = 0
                held_days = 0
        result.iloc[i] = held_signal

    return result
