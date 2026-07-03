"""
World Model 01: Dual Moving Average Momentum
Hypothesis: price trends persist over medium-term horizons; fast-MA > slow-MA signals uptrend.
Assumptions:
  - Returns have positive autocorrelation at 20-50 day horizon
  - Signal from SPY MA crossover generalizes across universe
  - No regime conditioning (the simplest momentum model)
RESEARCH ONLY — NOT INVESTMENT ADVICE.
"""
import pandas as pd
import numpy as np

NAME = "momentum_dma"
ASSUMPTIONS = [
    "Medium-term return autocorrelation is positive (trend persistence)",
    "20-day EMA vs 60-day EMA crossover identifies regimes",
    "SPY-based signal applied universe-wide (simplicity over tuning)",
]


def get_signals(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Given daily OHLCV for SPY, return {-1, 0, 1} signals.
    1 = long when fast EMA > slow EMA, 0 = flat otherwise.
    """
    close = ohlcv["close"] if "close" in ohlcv.columns else ohlcv["Close"]
    fast = close.ewm(span=20, adjust=False).mean()
    slow = close.ewm(span=60, adjust=False).mean()

    signals = pd.Series(0, index=close.index)
    signals[fast > slow] = 1
    signals[fast < slow] = -1

    # Require signal to persist 3 days before flipping (reduce churn)
    smoothed = signals.copy()
    for i in range(3, len(signals)):
        window = signals.iloc[i-2:i+1]
        if (window == 1).all():
            smoothed.iloc[i] = 1
        elif (window == -1).all():
            smoothed.iloc[i] = -1
        else:
            smoothed.iloc[i] = smoothed.iloc[i-1]

    return smoothed
