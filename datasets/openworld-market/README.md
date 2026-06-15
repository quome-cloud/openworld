# openworld-market

Daily OHLC for ~30 large-cap liquid stocks + SPY, used by the E50 same-day
trading world model. Committed as `prices.csv` so the backtest reruns fully
offline; `fetch_prices.py` refreshes it via **yfinance** (an optional dependency,
not part of the core framework).

## Files
- `prices.csv` — long format: `date,ticker,open,high,low,close` (split/dividend
  adjusted). ~5 years, 31 tickers.
- `fetch_prices.py` — refresh from yfinance: `python datasets/openworld-market/fetch_prices.py --period 5y`
- `CARD.md` — provenance, intended use, and the (important) limitations.

## Schema
```
date,ticker,open,high,low,close
2021-06-14,AAPL,128.5,129.1,127.9,128.8
```
Each day decomposes into **overnight** (prev close → open) and **intraday**
(open → close); E50 ranks tickers by predicted same-day move and picks the entry
(open or close).

## Use your own universe / data
Edit `UNIVERSE` in `fetch_prices.py`, or drop any OHLC CSV in the schema above
(your own tickers, frequencies, or vendor) — the experiment consumes it unchanged.

## Important
Not investment advice. Daily direction is near a coin flip; markets are
near-efficient. The fixed universe is **survivorship-selected** (today's
survivors), the period is one bull-leaning regime, and results are highly
sensitive to transaction costs. See `CARD.md`.
