#!/usr/bin/env python3
"""panel.py — Panel container + DEV / POST-CUTOFF split constants.

RESEARCH ONLY — paper trading / analysis. No live trading.

Contamination discipline (E-market v2 rule):
  The synthesis model's knowledge cutoff is January 2026. Everything through
  2026-01-31 is DEV — usable for code development, walk-forward *fitting*, and
  hyperparameter-free sanity checks only. The scored evaluation window is
  exclusively POST-CUTOFF: 2026-02-01 onward, data the synthesis model has
  never seen. Every reported table must label which window it comes from.

Conventions:
  - Wide frames are date x symbol, trading-day index.
  - ret[t] = close[t]/close[t-1] - 1  (auto-adjusted closes).
  - A feature value at date t may use information up to and including the
    close of t. Positions formed from features(t) earn ret(t+1) onward.
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DEV_END = pd.Timestamp("2026-01-31")       # last DEV date (inclusive)
POST_START = pd.Timestamp("2026-02-01")    # first scored date
COST_BPS_PER_SIDE = 10.0
HORIZON = 5                                # holding horizon, trading days
N_DECILES = 10


@dataclasses.dataclass
class Panel:
    close: pd.DataFrame     # date x symbol, adjusted
    volume: pd.DataFrame    # date x symbol, shares
    ret: pd.DataFrame       # date x symbol, daily close-to-close
    sector: pd.Series       # symbol -> GICS sector

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> pd.Index:
        return self.close.columns


def load_panel(min_coverage: float = 0.98) -> Panel:
    """Load cached OHLCV, pivot wide, keep symbols with near-complete data.

    min_coverage: fraction of trading days a symbol must have a close for.
    Filtering on *full-window* coverage uses the whole sample including
    post-cutoff dates; this is a survivorship convenience, not lookahead into
    returns — documented in the report limitations.
    """
    raw = pd.read_parquet(DATA / "ohlcv_panel.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    close = raw.pivot(index="date", columns="symbol", values="close").sort_index()
    volume = raw.pivot(index="date", columns="symbol", values="volume").sort_index()
    keep = close.columns[close.notna().mean() >= min_coverage]
    close, volume = close[keep], volume[keep]
    ret = close.pct_change(fill_method=None)

    uni = pd.read_csv(DATA / "universe_sp500.csv")
    uni["ysym"] = uni["symbol"].str.replace(".", "-", regex=False)
    sector = uni.set_index("ysym")["sector"].reindex(keep)
    return Panel(close=close, volume=volume, ret=ret, sector=sector)


def forward_return(panel: Panel, h: int = HORIZON) -> pd.DataFrame:
    """h-day forward return starting the day AFTER the signal date:
    fwd[t] = close[t+h]/close[t] - 1, i.e. earned over t+1..t+h.
    Used as the model target (cross-sectionally demeaned at fit time)."""
    c = panel.close
    return c.shift(-h) / c - 1.0


def xs_demean(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally demean each date's row (removes the market factor)."""
    return df.sub(df.mean(axis=1), axis=0)


def xs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date. NaNs stay NaN (undefined feature)."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)
