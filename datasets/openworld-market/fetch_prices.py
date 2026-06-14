"""Fetch daily OHLC for the openworld-market universe via yfinance.

Writes a long-format CSV (date,ticker,open,high,low,close) to prices.csv, which
is committed so the E50 backtest reruns fully offline. yfinance is only needed to
REFRESH the data; it is an optional dependency, not part of the core framework.

Usage:  python datasets/openworld-market/fetch_prices.py [--period 5y]
"""

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "prices.csv"

# ~30 large-cap, liquid names across sectors + SPY benchmark. Fixed universe
# (note survivorship caveat: these are survivors selected today).
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",          # tech/mega
    "JPM", "BAC", "GS", "V", "MA",                                     # financials
    "JNJ", "UNH", "PFE", "MRK",                                        # health
    "XOM", "CVX",                                                      # energy
    "WMT", "HD", "PG", "KO", "PEP", "MCD", "COST",                     # consumer
    "DIS", "NFLX", "CSCO", "INTC", "AMD",                              # media/semi
    "SPY",                                                             # benchmark
]


def main():
    import yfinance as yf
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="5y")
    args = ap.parse_args()
    print(f"fetching {len(UNIVERSE)} tickers, period {args.period} ...")
    data = yf.download(UNIVERSE, period=args.period, interval="1d",
                       auto_adjust=True, progress=False, group_by="ticker")
    rows = []
    for t in UNIVERSE:
        try:
            df = data[t][["Open", "High", "Low", "Close"]].dropna()
        except Exception:
            print(f"  [skip {t}: no data]")
            continue
        for date, r in df.iterrows():
            rows.append((date.strftime("%Y-%m-%d"), t,
                         round(float(r["Open"]), 4), round(float(r["High"]), 4),
                         round(float(r["Low"]), 4), round(float(r["Close"]), 4)))
    rows.sort()
    with OUT.open("w") as f:
        f.write("date,ticker,open,high,low,close\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    n_t = len({r[1] for r in rows})
    n_d = len({r[0] for r in rows})
    print(f"wrote {len(rows)} rows ({n_t} tickers x ~{n_d} days) -> {OUT}")


if __name__ == "__main__":
    main()
