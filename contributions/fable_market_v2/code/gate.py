#!/usr/bin/env python3
"""gate.py — Long-short decile portfolio + the v2 statistical gate.

RESEARCH ONLY — paper trading / analysis. No live trading.

Portfolio construction:
  Each date t with predictions: rank valid symbols; long the top decile
  equal-weight (+1 gross), short the bottom decile equal-weight (-1 gross).
  Positions are held HORIZON days via overlapping tranches (Jegadeesh-Titman):
  today's weights = mean of the last HORIZON daily tranche portfolios, so
  ~1/HORIZON of the book turns over daily.
  pnl(t+1) = sum_i w_i(t) * ret_i(t+1);  costs = 10 bps per side on turnover:
  cost(t+1) = 0.0010 * sum_i |w_i(t) - w_i(t-1)*drift|, drift ignored
  (second-order at daily horizon; noted in limitations).

Gate (per strategy/model variant, POST-CUTOFF window only):
  (ii)  net long-short annualized Sharpe > 0
  (iii) cross-sectional permutation p < 0.05 (shuffle predictions across
        valid symbols WITHIN each date; preserves the market factor and the
        prediction distribution; >= 200 shuffles, we use 500; full pipeline
        incl. overlapping tranches + costs re-run per shuffle)
  (iv)  demeaned/benchmark variant: long-only top decile minus equal-weight
        universe benchmark, net Sharpe reported alongside
  (v)   all numbers are net of costs.
  Plus: circular block bootstrap 95% CI on the net LS Sharpe (block=10,
  2000 resamples) and the permutation-null-implied minimum detectable effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from panel import COST_BPS_PER_SIDE, HORIZON, N_DECILES

ANN = 252.0


def decile_weights(pred_row: np.ndarray) -> np.ndarray:
    """Single-date tranche: +1/n_top on top decile, -1/n_bot on bottom."""
    w = np.zeros_like(pred_row, dtype=float)
    ok = np.isfinite(pred_row)
    n = int(ok.sum())
    if n < N_DECILES * 2:
        return w
    k = max(n // N_DECILES, 1)
    idx = np.where(ok)[0]
    order = idx[np.argsort(pred_row[idx])]
    w[order[-k:]] = 1.0 / k
    w[order[:k]] = -1.0 / k
    return w


def ls_returns(preds: pd.DataFrame, ret: pd.DataFrame,
               cost_bps: float = COST_BPS_PER_SIDE,
               horizon: int = HORIZON) -> pd.DataFrame:
    """Daily net/gross LS returns + the long-leg and benchmark series.

    Returns DataFrame indexed by pnl date with columns:
      gross, net, long_gross, bench (EW universe), turnover.
    """
    dates = preds.index
    ret = ret.reindex(index=ret.index)  # no-op; clarity
    P = preds.to_numpy(dtype=float)
    tranches = np.zeros((len(dates), preds.shape[1]))
    for i in range(len(dates)):
        tranches[i] = decile_weights(P[i])
    # overlapping holding: weights(t) = mean of tranches t-horizon+1..t
    W = pd.DataFrame(tranches, index=dates, columns=preds.columns) \
        .rolling(horizon, min_periods=1).mean().to_numpy()

    rows = []
    prev_w = np.zeros(preds.shape[1])
    for i, d in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        r_next = ret.loc[nxt].reindex(preds.columns).to_numpy(dtype=float)
        r_next = np.nan_to_num(r_next, nan=0.0)
        w = W[i]
        gross = float(w @ r_next)
        turnover = float(np.abs(w - prev_w).sum())
        cost = cost_bps / 1e4 * turnover
        long_w = np.clip(w, 0, None)
        long_gross = float(long_w @ r_next) / max(long_w.sum(), 1e-12)
        valid = np.isfinite(P[i])
        bench = float(np.nanmean(np.where(valid, r_next, np.nan))) if valid.any() else 0.0
        rows.append((nxt, gross, gross - cost, long_gross, bench, turnover))
        prev_w = w
    out = pd.DataFrame(rows, columns=["date", "gross", "net", "long_gross",
                                      "bench", "turnover"]).set_index("date")
    # long-leg net: charge half the book's costs to the long side
    return out


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 20 or x.std() == 0:
        return float("nan")
    return float(x.mean() / x.std() * np.sqrt(ANN))


def permutation_test(preds: pd.DataFrame, ret: pd.DataFrame,
                     n_perm: int = 500, seed: int = 20260706) -> dict:
    """Two permutation nulls (design corrected at DEV calibration, before
    pre-registration — see report section 5):

    PRIMARY (mandate): shuffle predictions across valid symbols WITHIN each
    date, compare GROSS Sharpe. Within-date shuffling destroys signal
    persistence, so a shuffled book turns over ~fully daily; comparing NET
    numbers against that null would hand ~+3 Sharpe of cost savings to any
    persistent-but-useless signal (measured on random preds: null net Sharpe
    approx -5.5 at 0.75 daily turnover). Gross-vs-gross removes the
    asymmetry; costs are enforced by the separate net_sharpe>0 condition.

    ROBUSTNESS: static ticker relabeling — permute the prediction COLUMNS
    once per draw. Turnover, costs and the weight time-series are exactly
    preserved; only the alignment of predictions to the right tickers is
    destroyed. Compared on NET Sharpe.
    """
    rng = np.random.default_rng(seed)
    lr = ls_returns(preds, ret)
    obs_gross, obs_net = sharpe(lr["gross"]), sharpe(lr["net"])
    P = preds.to_numpy(dtype=float)
    null_g = np.empty(n_perm)
    null_n = np.empty(n_perm)
    for b in range(n_perm):
        Q = P.copy()
        for i in range(P.shape[0]):
            ok = np.isfinite(P[i])
            vals = Q[i][ok]
            rng.shuffle(vals)
            Q[i][ok] = vals
        qdf = pd.DataFrame(Q, index=preds.index, columns=preds.columns)
        null_g[b] = sharpe(ls_returns(qdf, ret)["gross"])
        # static relabel null
        perm = rng.permutation(P.shape[1])
        rdf = pd.DataFrame(P[:, perm], index=preds.index, columns=preds.columns)
        null_n[b] = sharpe(ls_returns(rdf, ret)["net"])
    p_gross = float((1 + np.sum(null_g >= obs_gross)) / (1 + n_perm))
    p_net_relabel = float((1 + np.sum(null_n >= obs_net)) / (1 + n_perm))
    return {"observed_gross_sharpe": obs_gross, "observed_net_sharpe": obs_net,
            "p_value": p_gross, "p_net_static_relabel": p_net_relabel,
            "null_gross_mean": float(np.nanmean(null_g)),
            "null_gross_std": float(np.nanstd(null_g)),
            "null_gross_q95": float(np.nanquantile(null_g, 0.95)),
            "null_relabel_net_mean": float(np.nanmean(null_n)),
            "null_relabel_net_std": float(np.nanstd(null_n)),
            "n_perm": n_perm}


def block_bootstrap_ci(net: pd.Series, n_boot: int = 2000, block: int = 10,
                       seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    x = net.dropna().to_numpy()
    n = len(x)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n)
            idx.extend(((s + np.arange(block)) % n).tolist())
        xi = x[np.array(idx[:n])]
        stats[b] = xi.mean() / (xi.std() + 1e-12) * np.sqrt(ANN)
    return {"ci_lo": float(np.quantile(stats, 0.025)),
            "ci_hi": float(np.quantile(stats, 0.975)),
            "n_boot": n_boot, "block": block}


def evaluate(preds: pd.DataFrame, ret: pd.DataFrame, label: str,
             n_perm: int = 500) -> dict:
    """Full gate evaluation on the window covered by `preds`."""
    lr = ls_returns(preds, ret)
    net_sh = sharpe(lr["net"])
    perm = permutation_test(preds, ret, n_perm=n_perm)
    boot = block_bootstrap_ci(lr["net"])
    long_excess = lr["long_gross"] - lr["bench"] - (COST_BPS_PER_SIDE / 1e4) * lr["turnover"] / 2
    res = {
        "label": label,
        "window": {"start": str(lr.index.min().date()), "end": str(lr.index.max().date()),
                   "n_days": int(len(lr))},
        "gross_sharpe": sharpe(lr["gross"]),
        "net_sharpe": net_sh,
        "net_ann_return": float(lr["net"].mean() * ANN),
        "net_ann_vol": float(lr["net"].std() * np.sqrt(ANN)),
        "avg_daily_turnover": float(lr["turnover"].mean()),
        "long_minus_bench_net_sharpe": sharpe(long_excess),
        "bench_ann_return": float(lr["bench"].mean() * ANN),
        "permutation": perm,
        "bootstrap_ci_net_sharpe": boot,
        "PASS": bool(net_sh > 0 and perm["p_value"] < 0.05),
    }
    return res
