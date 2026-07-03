"""
World Model 05: Market Breadth + Trend Composite
Hypothesis: SPY/QQQ divergence signals sector rotation; combined with trend,
            predicts near-term direction of SPY.
Assumptions:
  - When QQQ > SPY (tech leading): broad risk-on → long SPY
  - When SPY > QQQ (defensives leading): risk-off → short or flat
  - Signal requires 10-day window to smooth noise
  - Breadth divergence + trend alignment = strongest signal
RESEARCH ONLY — NOT INVESTMENT ADVICE.
"""
import pandas as pd
import numpy as np

NAME = "macro_breadth_composite"
ASSUMPTIONS = [
    "QQQ/SPY relative strength proxies growth vs. value rotation",
    "QQQ outperforming (10d) = risk-on regime → long SPY",
    "SPY underperforming QQQ (10d) = risk-off → short SPY",
    "Require 200-day SMA filter: only long above 200d SMA (bull mkt structural filter)",
]


def get_signals(ohlcv: pd.DataFrame, qqq_ohlcv: pd.DataFrame = None) -> pd.Series:
    """
    ohlcv = SPY data (primary)
    qqq_ohlcv = QQQ data (optional — if not provided, falls back to pure SPY trend)
    """
    spy = ohlcv["close"] if "close" in ohlcv.columns else ohlcv["Close"]
    sma200 = spy.rolling(200).mean()
    bull_mkt = spy > sma200

    if qqq_ohlcv is not None:
        qqq = qqq_ohlcv["close"] if "close" in qqq_ohlcv.columns else qqq_ohlcv["Close"]
        # Align indices
        qqq = qqq.reindex(spy.index).ffill()
        # 10-day relative return: QQQ vs SPY
        spy_ret_10 = spy.pct_change(10)
        qqq_ret_10 = qqq.pct_change(10)
        qqq_leading = qqq_ret_10 > spy_ret_10

        signals = pd.Series(0, index=spy.index)
        signals[bull_mkt & qqq_leading] = 1       # risk-on + QQQ leading → long
        signals[~bull_mkt] = -1                    # below 200d SMA → short
        # QQQ lagging in bull mkt → flat (wait for rotation to resolve)
    else:
        # Fallback: pure 20/200 trend
        sma20 = spy.rolling(20).mean()
        signals = pd.Series(0, index=spy.index)
        signals[bull_mkt & (spy > sma20)] = 1
        signals[~bull_mkt] = -1

    return signals
