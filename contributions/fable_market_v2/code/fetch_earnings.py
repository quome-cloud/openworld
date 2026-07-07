#!/usr/bin/env python3
"""fetch_earnings.py — Cache earnings dates + EPS surprises per symbol.

RESEARCH ONLY. Source: Yahoo Finance via yfinance Ticker.get_earnings_dates.
Cache: data/earnings/<SYM>.csv with columns timestamp, eps_estimate,
eps_reported, surprise_pct. Resumable: symbols with an existing cache file
are skipped, so the script can be re-run after rate-limit interruptions.
Failures are recorded in data/earnings_log.json; symbols that never resolve
simply contribute zero PEAD signal (documented in the feature module).
"""
import json, pathlib, sys, time, datetime

import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parent.parent
EDIR = ROOT / "data" / "earnings"
EDIR.mkdir(parents=True, exist_ok=True)


def fetch_one(sym: str) -> str:
    f = EDIR / f"{sym}.csv"
    if f.exists():
        return "cached"
    try:
        ed = yf.Ticker(sym).get_earnings_dates(limit=16)
    except Exception as e:
        return f"error: {type(e).__name__}"
    if ed is None or ed.empty:
        f.write_text("timestamp,eps_estimate,eps_reported,surprise_pct\n")
        return "empty"
    out = ed.reset_index()
    out.columns = ["timestamp", "eps_estimate", "eps_reported", "surprise_pct"][: len(out.columns)]
    out.to_csv(f, index=False)
    return "ok"


def main() -> None:
    uni = pd.read_csv(ROOT / "data" / "universe_sp500.csv")
    symbols = [s.replace(".", "-") for s in uni["symbol"]]
    log = {"source": "yfinance " + yf.__version__ + " get_earnings_dates(limit=16)",
           "fetch_utc": datetime.datetime.utcnow().isoformat() + "Z", "status": {}}
    for i, sym in enumerate(symbols):
        st = fetch_one(sym)
        log["status"][sym] = st
        if st not in ("cached",):
            time.sleep(0.8)
        if i % 25 == 0:
            print(f"{i}/{len(symbols)} {sym}: {st}", flush=True)
    n_ok = sum(1 for v in log["status"].values() if v in ("ok", "cached"))
    log["n_ok"] = n_ok
    json.dump(log, open(ROOT / "data" / "earnings_log.json", "w"), indent=2)
    print(f"done: {n_ok}/{len(symbols)} with earnings cache")


if __name__ == "__main__":
    main()
