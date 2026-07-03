"""
World Model Interface — all world models implement this.

RESEARCH ONLY — NOT INVESTMENT ADVICE.

The LLM (Prism, A004) writes the body of get_signals().
This file is a stub showing the required interface.
"""

import pandas as pd


def get_signals(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Given OHLCV dataframe, return a signal series.

    Args:
        ohlcv: pd.DataFrame with columns [open, high, low, close, volume],
               indexed by date (pd.DatetimeIndex or date-comparable index).

    Returns:
        pd.Series of {-1, 0, 1} signals indexed by date.
            -1 = short or move to cash (bearish)
             0 = flat / cash (neutral)
             1 = long (bullish)

    Each signal applies to the *next* day's return (signal[t] × return[t+1]).
    The backtester handles alignment.
    """
    raise NotImplementedError("Implement world model logic here")


# ---------------------------------------------------------------------------
# World model metadata (for ledger)
# ---------------------------------------------------------------------------

NAME = "placeholder"
DESCRIPTION = "Stub interface — replace get_signals() with actual world model logic"
ASSUMPTIONS = [
    "Returns are predictable from OHLCV features",
    "No regime shifts during the evaluation window",
]
UNIVERSE_SCOPE = "single symbol"  # or "multi-symbol"
