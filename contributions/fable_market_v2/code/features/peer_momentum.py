#!/usr/bin/env python3
"""FEATURE FAMILY: PEER — Correlation-graph peer momentum gap.

RESEARCH ONLY — paper trading / analysis. No live trading.

Economic rationale (causal story, written before fitting):
  Firms are linked through customers, suppliers, shared factor exposures and
  competitive position. Investors watching stock i do not fully track news
  arriving via i's economic neighbors, so neighbor returns diffuse into i
  with a lag (Cohen & Frazzini 2008 customer-supplier momentum; Ali &
  Hirshleifer 2020 connected-firm momentum). Explicit supply-chain graphs are
  paywalled; the return-correlation graph is a measurable proxy: stocks that
  comoved historically share economic exposures. Prediction: if i's peers
  rallied over the past month but i did not, i catches up over the next days.

Feature:
  peer_mom_gap_i(t) = mean_{j in P_i(t)} m_j(t) - m_i(t)
    m_j(t)  = 21-day return of j through t, cross-sectionally demeaned
              (units: fractional return over 21 trading days).
    P_i(t)  = the K=10 stocks with highest correlation to i, estimated on
              trailing 126 trading days of daily returns, excluding i itself.
              Peer graph refit at each month start (fit window ends the prior
              month) — no future comovement information enters P_i(t).
  Parameters K=10, corr window 126d, momentum window 21d fixed a priori.

Data dependencies: panel.ret only.
Anti-lookahead: peer graph for month m uses returns through end of m-1;
  m_j(t) uses returns through t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAMILY = "PEER"
FEATURES = ["peer_mom_gap"]

K_PEERS = 10
CORR_WINDOW = 126
MOM_WINDOW = 21


def compute(panel) -> dict[str, pd.DataFrame]:
    ret = panel.ret
    dates = ret.index
    months = pd.Series(dates, index=dates).dt.to_period("M")
    mom = (1.0 + ret).rolling(MOM_WINDOW).apply(np.prod, raw=True) - 1.0
    mom = mom.sub(mom.mean(axis=1), axis=0)  # cross-sectional demean

    gap = pd.DataFrame(np.nan, index=dates, columns=ret.columns)
    peer_idx = None  # symbol -> list of peer symbols
    for m in months.unique():
        m_dates = dates[months == m]
        hist = ret.loc[: m_dates[0]].iloc[:-1].tail(CORR_WINDOW)
        if len(hist) >= CORR_WINDOW // 2:
            valid = hist.columns[hist.notna().mean() > 0.9]
            C = hist[valid].corr().to_numpy().copy()
            np.fill_diagonal(C, -np.inf)
            order = np.argsort(-C, axis=1)[:, :K_PEERS]
            peer_idx = {valid[i]: [valid[j] for j in order[i]]
                        for i in range(len(valid))}
        if peer_idx is None:
            continue
        mom_m = mom.loc[m_dates]
        for sym, peers in peer_idx.items():
            if sym in gap.columns:
                gap.loc[m_dates, sym] = mom_m[peers].mean(axis=1) - mom_m[sym]
    return {"peer_mom_gap": gap.where(panel.close.notna())}
