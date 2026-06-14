# E50 - Same-day trading world model: ranking tickers and timing the entry

**Date:** 2026-06-14
**Status:** approved (design); pending spec review

## Goal

A verified, auditable world model that, across a universe of tickers, ranks the
one most likely to rise on a given day from its history, and recommends the
*exact entry* for a same-day trade (buy at the open or at the close). Honestly
backtested out-of-sample with transaction costs. The point is NOT a money-
printer: daily direction is near a coin flip and markets are near-efficient. The
point is the framework's value on a real, hard problem - an exact, auditable
model, rigorous out-of-sample evaluation that makes overfitting visible, and the
one documented effect here (the overnight-vs-intraday decomposition).

## The decision: open vs close (same-day)

Each trading day decomposes into two segments:
- **overnight**: previous close -> open (`r_on = O/PC - 1`)
- **intraday**: open -> close (`r_id = C/O - 1`)

"Exact time to purchase" = buy at the **open** (capture intraday, sell at close)
or buy at the **close** (capture overnight, sell next open). A clean, testable
open/close choice.

## World model (verified, auditable - no black box)

- **Each ticker is a world** (cf. E46): from history, verified estimators produce
  P(up) and expected return for each segment, plus momentum / mean-reversion /
  realised-volatility features. All readable numpy, no learned weights.
- **Selection = many-worlds posterior**: each day, rank the universe by predicted
  same-day expected return / P(up) and pick the top (traverse to the highest-
  probability-up ticker).
- **Timing**: for the pick, choose the segment (open- or close-entry) with the
  better historical edge.
- **Concrete output**: for the latest day, emit {ticker, segment (open/close),
  predicted P(up)} with a disclaimer.

## Data

- **Real**: daily OHLC via **yfinance** for ~30 large-cap liquid tickers + SPY,
  ~5 years; committed to `datasets/openworld-market/prices.csv` so the backtest
  reruns offline (yfinance only needed to refresh). yfinance is an OPTIONAL
  dependency used by the fetch script, not by the core framework.
- **Synthetic**: a generated market with a KNOWN injected overnight-drift edge,
  to validate the detector recovers an edge when one exists (so a null real
  result reads as "market efficient", not "model broken").

## Evaluation (honest, out-of-sample, with costs)

- **Walk-forward**: split by date; on each test day predict using ONLY past data
  (no lookahead), then realise the actual segment return. No survivorship beyond
  the fixed universe (documented).
- **Costs**: realistic per-trade bps (round-trip), swept.
- **Baselines**: buy-and-hold SPY, random ticker, always-open, always-close,
  equal-weight.
- **Metrics**: hit rate, total/annualised return, Sharpe, max drawdown, vs
  baselines; in-sample vs OOS, with/without costs (to expose how much survives).

## Claims / results

1. **Segment decomposition**: the overnight vs intraday split is real and
   ticker-specific; the model recommends entry timing accordingly (and the
   documented overnight premium shows up).
2. **Synthetic validation**: on a market with a known overnight edge, the model's
   top-pick + correct-segment policy beats baselines and recovers the edge.
3. **Honest OOS on real data**: report the true edge - in-sample/no-cost it looks
   best; out-of-sample and after costs it shrinks (overfitting made visible). The
   framework's exact/OOS rigor is the deliverable, whatever the sign.

## Honest boundaries (stated prominently)

- Not investment advice; near-efficient markets mean daily edges are tiny and
  fragile; results are sensitive to costs/slippage and the fixed (survivorship-
  selected) universe and period.
- Fixed, auditable estimators - not a learned predictor; the claim is about
  rigor and auditability, not alpha.

## Deliverables

- `datasets/openworld-market/` (fetch_prices.py [yfinance], prices.csv, README,
  CARD).
- `experiments/e50_trading.py` (+ results), self-checking (synthetic edge
  recovered; no-lookahead enforced).
- Figure (segment decomposition / overnight premium; equity curve vs baselines
  in-sample vs OOS; cost sweep) + table; paper subsection; `\NumExperiments`
  bump. PR targets `main` (stacked on E48; merge #35 first).
