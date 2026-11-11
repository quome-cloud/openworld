#!/usr/bin/env python3
"""FEATURE FAMILY: SLL — Sector lead-lag graph.

RESEARCH ONLY — paper trading / analysis. No live trading.

Economic rationale (causal story, written before fitting):
  Information diffuses gradually along economic linkages. A shock to sector X
  (e.g. Energy) repricing input costs or demand for sector Y (e.g. Industrials)
  shows up in X's price today but in Y's price only over subsequent days,
  because investors specialize by sector and attention is limited
  (Hong & Stein 1999; Menzly & Ozbas 2010 cross-industry momentum;
  Cohen & Frazzini 2008 economic links). If true, a linear map L from lagged
  sector returns to next-period sector relative returns is estimable from data
  and stable enough to carry out-of-sample at daily/weekly horizons.

Features (both walk-forward, refit at each month start, trailing window):
  sll_daily : s_hat_k(t+1) = sum_j L1[j,k] * r_sec_j(t),
              L1 = ridge coefficients of sector return at t on ALL 11 sector
              returns at t-1, trailing 504 trading days (min 252), lambda=1.0
              on standardized sector returns. Fixed a priori.
  sll_weekly: same structure at weekly granularity:
              regress sector return over (t+1..t+5) on sector returns over
              (t-4..t), overlapping daily observations, trailing 504d.
  Stock-level value = predicted return of the stock's GICS sector, demeaned
  across the 11 sectors (units: expected relative return per period,
  dimensionless). All member stocks of a sector share the value; the model
  layer z-scores cross-sectionally.

Data dependencies: panel.ret (daily close-to-close), panel.sector (GICS from
  Wikipedia constituents table). No external data.

Anti-lookahead: the coefficient matrix used for dates in month m is fit only
  on data through the last trading day of month m-1; predictions at t use
  sector returns up to and including t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAMILY = "SLL"
FEATURES = ["sll_daily", "sll_weekly"]

WINDOW = 504
MIN_WINDOW = 252
RIDGE_LAMBDA = 1.0


def _sector_returns(panel) -> pd.DataFrame:
    """Equal-weight daily return per GICS sector (date x sector)."""
    ret = panel.ret
    out = {}
    for sec in sorted(panel.sector.dropna().unique()):
        cols = panel.sector.index[panel.sector == sec]
        out[sec] = ret[cols].mean(axis=1)
    return pd.DataFrame(out)


def _ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Solve (X'X + lam*I) B = X'Y for multi-output Y. X standardized."""
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    k = Xs.shape[1]
    B = np.linalg.solve(Xs.T @ Xs + lam * np.eye(k), Xs.T @ (Y - Y.mean(0)))
    return B  # maps standardized lagged sector returns -> sector returns


def _walk_forward_pred(sec_ret: pd.DataFrame, lag: int, horizon: int) -> pd.DataFrame:
    """Predict each sector's next-`horizon` return from trailing-`lag` returns.

    Refit at each month start on trailing WINDOW days ending before the month.
    Returns date x sector predictions (aligned so row t uses info <= t).
    """
    X_all = sec_ret.rolling(lag).sum()               # info up to t
    Y_all = sec_ret.rolling(horizon).sum().shift(-horizon)  # t+1..t+horizon
    dates = sec_ret.index
    months = pd.Series(dates, index=dates).dt.to_period("M")
    preds = pd.DataFrame(np.nan, index=dates, columns=sec_ret.columns)
    B, mu, sd = None, None, None
    for m in months.unique():
        m_dates = dates[months == m]
        train_end = m_dates[0]
        # Fit strictly before the month, with an `horizon`-day embargo so no
        # training target overlaps month-m returns.
        Xtr = X_all.loc[:train_end].iloc[lag:]
        if len(Xtr) <= horizon + 1:
            continue
        Xtr = Xtr.iloc[:-(horizon + 1)]
        Ytr = Y_all.loc[Xtr.index]
        ok = Xtr.notna().all(axis=1) & Ytr.notna().all(axis=1)
        Xtr, Ytr = Xtr[ok].tail(WINDOW), Ytr[ok].tail(WINDOW)
        if len(Xtr) >= MIN_WINDOW:
            mu, sd = Xtr.values.mean(0), Xtr.values.std(0) + 1e-12
            B = _ridge_fit(Xtr.values, Ytr.values, RIDGE_LAMBDA)
        if B is None:
            continue
        Xm = X_all.loc[m_dates]
        preds.loc[m_dates] = ((Xm.values - mu) / sd) @ B
    return preds


def compute(panel) -> dict[str, pd.DataFrame]:
    sec_ret = _sector_returns(panel)
    out = {}
    for name, (lag, hor) in {"sll_daily": (1, 1), "sll_weekly": (5, 5)}.items():
        sec_pred = _walk_forward_pred(sec_ret, lag, hor)
        sec_pred = sec_pred.sub(sec_pred.mean(axis=1), axis=0)  # demean across sectors
        stock = pd.DataFrame(np.nan, index=panel.dates, columns=panel.symbols)
        for sec in sec_pred.columns:
            cols = panel.sector.index[panel.sector == sec]
            stock.loc[:, cols] = np.repeat(
                sec_pred[sec].values[:, None], len(cols), axis=1)
        out[name] = stock.where(panel.close.notna())
    return out
