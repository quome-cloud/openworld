"""
World Model 03: Volatility-Regime Conditioned Momentum
Hypothesis: momentum works in low-vol regimes; mean reversion (or cash) in high-vol regimes.
Assumptions:
  - Realized vol (20-day) is a reliable regime proxy
  - Trending behavior dominates when vol < 1.5% daily (annualized ~24%)
  - High-vol regimes are driven by uncertainty → trend-following breaks down
  - Regime-switched strategy reduces drawdown without proportional return loss
RESEARCH ONLY — NOT INVESTMENT ADVICE.
"""
import pandas as pd
import numpy as np

NAME = "regime_vol_conditioned"
ASSUMPTIONS = [
    "Low-vol regime (realized vol < 1.5%/day) supports momentum",
    "High-vol regime (realized vol >= 1.5%/day) → cash (no edge)",
    "20-day realized vol is the regime signal",
    "Momentum signal: 10-day EMA > 30-day EMA",
]

VOL_THRESHOLD = 0.015  # 1.5% daily = ~24% annualized


def get_signals(ohlcv: pd.DataFrame) -> pd.Series:
    """
    In low-vol regime: trade momentum (10/30 EMA crossover).
    In high-vol regime: sit in cash (signal = 0).
    """
    close = ohlcv["close"] if "close" in ohlcv.columns else ohlcv["Close"]
    returns = close.pct_change()

    # Regime detection
    realized_vol = returns.rolling(20).std()
    low_vol_regime = realized_vol < VOL_THRESHOLD

    # Momentum signal
    fast = close.ewm(span=10, adjust=False).mean()
    slow = close.ewm(span=30, adjust=False).mean()
    momentum = pd.Series(0, index=close.index)
    momentum[fast > slow] = 1
    momentum[fast < slow] = -1

    # Combine: only trade in low-vol regime
    signals = pd.Series(0, index=close.index)
    signals[low_vol_regime & (momentum == 1)] = 1
    signals[low_vol_regime & (momentum == -1)] = -1

    return signals
