# Dataset card — openworld-market

## Summary
Daily split/dividend-adjusted OHLC for ~30 large-cap US equities + SPY (~5 years),
for the E50 same-day trading world model.

## Provenance
Fetched from Yahoo Finance via `yfinance` (`fetch_prices.py`, `auto_adjust=True`),
committed as `prices.csv`. Refreshable; the committed snapshot makes the backtest
reproducible offline.

## Intended use
Backtesting the E50 world model (ranking tickers by predicted same-day move;
choosing open- vs close-entry) under an honest, no-lookahead, after-cost,
walk-forward protocol. A template: swap the universe or drop in your own OHLC.

## Limitations & ethics
- **Not investment advice.** Markets are near-efficient; daily directional edges
  are tiny and fragile, and any backtest edge may not persist live.
- **Survivorship bias**: the universe is fixed to names that exist/are liquid
  today; delisted/failed names are absent, biasing results upward.
- **Single regime**: ~5 years of a bull-leaning period; not representative of all
  market conditions.
- **Cost sensitivity**: same-day strategies trade daily; results swing from
  positive to negative across realistic transaction-cost assumptions.
- Adjusted prices reconstruct historical open/close relationships approximately;
  intraday microstructure (slippage, the actual executable open/close) is not
  modeled.
