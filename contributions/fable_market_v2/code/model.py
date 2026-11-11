#!/usr/bin/env python3
"""model.py — Transparent model layer: walk-forward cross-sectional ridge.

RESEARCH ONLY — paper trading / analysis. No live trading.

Design (operator restructure, 2026-07-06):
  The model is a REGULARIZED LINEAR map on the feature matrix — auditable
  because every input feature carries an explicit economic meaning (see
  features/). No deep nets, no feature interactions beyond those written as
  named features.

Target:   y_i(t) = h-day forward return of stock i (t+1..t+h close-to-close),
          cross-sectionally demeaned per date. Units: fractional return.
Inputs:   features, each cross-sectionally z-scored per date; missing -> 0
          (neutral). Only symbols with a close on date t enter that date.
Fitting:  walk-forward, refit at each month start. Training set = all panel
          days whose forward-return window closes strictly before the month
          being predicted (h-day embargo). Ridge lambda chosen per refit by
          time-ordered 5-fold cross-validation on the training window from a
          fixed grid — causal, no post-cutoff information enters any fit that
          predicts post-cutoff dates beyond the walk-forward boundary.
Output:   predictions DataFrame (date x symbol) plus per-refit standardized
          coefficients = feature-importance readout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from panel import HORIZON, forward_return, xs_demean, xs_zscore

LAMBDA_GRID = [1e2, 1e3, 1e4, 1e5]
N_CV_FOLDS = 5


def _stack(feat_dfs: dict[str, pd.DataFrame], valid: pd.DataFrame):
    """Z-score each feature per date, align, return dict of arrays + names."""
    names = sorted(feat_dfs)
    zs = {n: xs_zscore(feat_dfs[n]).where(valid) for n in names}
    return names, zs


def _fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    k = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(k), X.T @ y)


def _cv_lambda(X: np.ndarray, y: np.ndarray, day_id: np.ndarray) -> float:
    """Time-ordered CV: split by day blocks, score MSE on the later fold."""
    uniq = np.unique(day_id)
    folds = np.array_split(uniq, N_CV_FOLDS)
    scores = {lam: [] for lam in LAMBDA_GRID}
    for i in range(1, N_CV_FOLDS):
        tr_days = np.concatenate(folds[:i])
        te_days = folds[i]
        tr = np.isin(day_id, tr_days)
        te = np.isin(day_id, te_days)
        if tr.sum() < 100 or te.sum() < 50:
            continue
        for lam in LAMBDA_GRID:
            b = _fit_ridge(X[tr], y[tr], lam)
            scores[lam].append(float(np.mean((y[te] - X[te] @ b) ** 2)))
    mean_scores = {lam: np.mean(v) if v else np.inf for lam, v in scores.items()}
    return min(mean_scores, key=mean_scores.get)


def walk_forward_predict(panel, feat_dfs: dict[str, pd.DataFrame],
                         pred_start: str, pred_end: str | None = None):
    """Predict xs-demeaned h-day forward returns for dates in [pred_start, pred_end].

    Returns (predictions DataFrame, importance DataFrame [refit x feature],
    chosen_lambdas dict).
    """
    valid = panel.close.notna()
    names, zs = _stack(feat_dfs, valid)
    y_full = xs_demean(forward_return(panel, HORIZON)).where(valid)

    dates = panel.dates
    pred_dates = dates[(dates >= pd.Timestamp(pred_start))
                       & (dates <= pd.Timestamp(pred_end or dates[-1]))]
    months = pd.Series(pred_dates, index=pred_dates).dt.to_period("M")

    preds = pd.DataFrame(np.nan, index=pred_dates, columns=panel.symbols)
    importance, lambdas = [], {}

    for m in months.unique():
        m_dates = pred_dates[months == m]
        # training days: forward window closes before the month starts
        cutoff_pos = dates.searchsorted(m_dates[0]) - HORIZON - 1
        train_days = dates[:max(cutoff_pos, 0)]
        rows_X, rows_y, rows_day = [], [], []
        for di, d in enumerate(train_days):
            yv = y_full.loc[d]
            ok = yv.notna()
            if ok.sum() < 50:
                continue
            X_d = np.column_stack([zs[n].loc[d].fillna(0.0).values for n in names])
            rows_X.append(X_d[ok.values])
            rows_y.append(yv[ok].values)
            rows_day.append(np.full(int(ok.sum()), di))
        if not rows_X:
            continue
        X = np.vstack(rows_X)
        y = np.concatenate(rows_y)
        day_id = np.concatenate(rows_day)
        lam = _cv_lambda(X, y, day_id)
        beta = _fit_ridge(X, y, lam)
        lambdas[str(m)] = lam
        importance.append(pd.Series(beta, index=names, name=str(m)))
        for d in m_dates:
            X_d = np.column_stack([zs[n].loc[d].fillna(0.0).values for n in names])
            row = X_d @ beta
            row[~valid.loc[d].values] = np.nan
            preds.loc[d] = row

    imp = pd.DataFrame(importance)
    return preds, imp, lambdas
