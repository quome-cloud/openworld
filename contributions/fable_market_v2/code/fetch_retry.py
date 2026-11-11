#!/usr/bin/env python3
"""fetch_retry.py — Re-fetch symbols dropped by fetch_data.py (rate-limit
casualties) in small batches and merge into the cached parquet. Re-run until
the dropped list stops shrinking. RESEARCH ONLY."""
import json, pathlib, time, datetime

import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
START = "2023-06-01"


def main() -> None:
    panel = pd.read_parquet(DATA / "ohlcv_panel.parquet")
    uni = pd.read_csv(DATA / "universe_sp500.csv")
    want = {s.replace(".", "-") for s in uni["symbol"]}
    have = set(panel["symbol"].unique())
    missing = sorted(want - have)
    print(f"missing {len(missing)}: {missing[:10]}...")
    frames = []
    for i in range(0, len(missing), 10):
        chunk = missing[i:i + 10]
        try:
            df = yf.download(chunk, start=START, auto_adjust=True,
                             group_by="ticker", progress=False, threads=False)
        except Exception as e:
            print("chunk failed:", e)
            time.sleep(20)
            continue
        for sym in chunk:
            try:
                sub = df[sym].dropna(subset=["Close"]).reset_index()
            except KeyError:
                continue
            if sub.empty:
                continue
            sub.columns = [c.lower() for c in sub.columns]
            sub["symbol"] = sym
            frames.append(sub[["date", "symbol", "open", "high", "low", "close", "volume"]])
        print(f"retry {i//10}: cumulative recovered {sum(f['symbol'].nunique() for f in frames)}", flush=True)
        time.sleep(5)
    if frames:
        add = pd.concat(frames, ignore_index=True)
        add["date"] = pd.to_datetime(add["date"])
        panel["date"] = pd.to_datetime(panel["date"])
        merged = pd.concat([panel, add], ignore_index=True) \
            .drop_duplicates(subset=["date", "symbol"], keep="first")
        merged.to_parquet(DATA / "ohlcv_panel.parquet", index=False)
        log = json.load(open(DATA / "fetch_log.json"))
        log.setdefault("retries", []).append(
            {"utc": datetime.datetime.utcnow().isoformat() + "Z",
             "recovered": int(add["symbol"].nunique())})
        log["n_got"] = int(merged["symbol"].nunique())
        log["dropped"] = sorted(want - set(merged["symbol"].unique()))
        json.dump(log, open(DATA / "fetch_log.json", "w"), indent=2)
        print(f"now have {merged['symbol'].nunique()} symbols")
    else:
        print("nothing recovered")


if __name__ == "__main__":
    main()
