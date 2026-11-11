#!/usr/bin/env python3
"""fetch_data.py — Download and cache daily OHLCV for the S&P 500 panel.

RESEARCH ONLY — paper trading / analysis. No live trading.

Data source: Yahoo Finance via yfinance (version logged in data/fetch_log.json).
Window: 2023-06-01 .. today. The extra pre-2024 buffer exists only so that
lookback features (up to 252d) are defined from the first DEV date onward.

Cache layout (all under data/):
  universe_sp500.csv     — symbol, name, GICS sector/sub-industry (Wikipedia)
  ohlcv_panel.parquet    — long panel: date, symbol, open, high, low, close, volume
                           (close/open/high/low are auto-adjusted by yfinance)
  spy_benchmark.csv      — SPY adjusted close (context only; the gate benchmark
                           is the equal-weight universe, computed from the panel)
  fetch_log.json         — source, timestamps, per-batch status, dropped tickers
Re-running with the cache present is a no-op unless --force.
"""
import json, sys, time, datetime, pathlib

import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
START = "2023-06-01"


def yahoo_symbol(s: str) -> str:
    return s.replace(".", "-")  # BRK.B -> BRK-B


def main(force: bool = False) -> None:
    out = DATA / "ohlcv_panel.parquet"
    if out.exists() and not force:
        print(f"cache hit: {out}")
        return
    uni = pd.read_csv(DATA / "universe_sp500.csv")
    symbols = [yahoo_symbol(s) for s in uni["symbol"]]
    log = {"source": "yfinance " + yf.__version__, "start": START,
           "fetch_utc": datetime.datetime.utcnow().isoformat() + "Z",
           "batches": [], "n_requested": len(symbols)}
    frames = []
    B = 50
    for i in range(0, len(symbols), B):
        chunk = symbols[i:i + B]
        t0 = time.time()
        df = yf.download(chunk, start=START, auto_adjust=True,
                         group_by="ticker", progress=False, threads=True)
        got = sorted({c[0] for c in df.columns if df[c[0]]["Close"].notna().sum() > 0})
        log["batches"].append({"i": i, "n": len(chunk), "got": len(got),
                               "secs": round(time.time() - t0, 1)})
        for sym in got:
            sub = df[sym].dropna(subset=["Close"]).reset_index()
            sub.columns = [c.lower() for c in sub.columns]
            sub["symbol"] = sym
            frames.append(sub[["date", "symbol", "open", "high", "low", "close", "volume"]])
        print(f"batch {i//B}: {len(got)}/{len(chunk)} in {log['batches'][-1]['secs']}s", flush=True)
        time.sleep(1.0)
    panel = pd.concat(frames, ignore_index=True)
    panel.to_parquet(out, index=False)
    spy = yf.download("SPY", start=START, auto_adjust=True, progress=False)
    spy.to_csv(DATA / "spy_benchmark.csv")
    log["n_got"] = panel["symbol"].nunique()
    log["dropped"] = sorted(set(symbols) - set(panel["symbol"].unique()))
    json.dump(log, open(DATA / "fetch_log.json", "w"), indent=2)
    print(f"panel: {panel.shape}, {panel['symbol'].nunique()} symbols, "
          f"{panel['date'].min()} .. {panel['date'].max()}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
