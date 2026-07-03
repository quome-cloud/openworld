#!/usr/bin/env python3
"""
download_data.py — Download OHLCV data from Yahoo Finance.

RESEARCH ONLY — NOT INVESTMENT ADVICE.

Downloads daily OHLCV for the 12-symbol universe and saves to data/raw_ohlcv.csv.
"""

import sys
import os

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance pandas numpy matplotlib")
    sys.exit(1)

import pandas as pd
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UNIVERSE = [
    "SPY", "QQQ",
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA",
    "BRK-B", "JPM", "JNJ", "XOM",
]

# Download window covers all splits plus buffer
DOWNLOAD_START = "2009-12-01"
DOWNLOAD_END = "2026-07-01"

DATA_DIR = Path(__file__).parent / "data"
RAW_PATH = DATA_DIR / "raw_ohlcv.csv"


def download_ohlcv(
    symbols: list[str] = UNIVERSE,
    start: str = DOWNLOAD_START,
    end: str = DOWNLOAD_END,
    output_path: Path = RAW_PATH,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download daily OHLCV for all symbols and save to a flat CSV.

    Returns a DataFrame with columns: [date, symbol, open, high, low, close, volume]
    """
    if output_path.exists() and not force:
        print(f"Raw data already exists at {output_path}. Use force=True to re-download.")
        df = pd.read_csv(output_path, parse_dates=["date"])
        _print_summary(df)
        return df

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading OHLCV for {len(symbols)} symbols: {start} to {end}")
    print(f"Symbols: {', '.join(symbols)}")
    print()

    all_dfs = []
    failed = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(start=start, end=end, auto_adjust=True)
            if raw.empty:
                print(f"  {symbol}: NO DATA")
                failed.append(symbol)
                continue

            # Normalize columns to lowercase
            raw = raw.reset_index()
            raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

            # Rename 'date' column (yfinance may call it 'Date')
            date_col = next((c for c in raw.columns if "date" in c), None)
            if date_col and date_col != "date":
                raw = raw.rename(columns={date_col: "date"})

            # Keep standard OHLCV columns
            keep_cols = ["date", "open", "high", "low", "close", "volume"]
            available = [c for c in keep_cols if c in raw.columns]
            raw = raw[available].copy()
            raw["symbol"] = symbol

            # Normalize date to date-only (no time component)
            raw["date"] = pd.to_datetime(raw["date"]).dt.date

            n = len(raw)
            print(f"  {symbol}: {n} rows ({raw['date'].min()} to {raw['date'].max()})")
            all_dfs.append(raw)

        except Exception as e:
            print(f"  {symbol}: ERROR — {e}")
            failed.append(symbol)

    if not all_dfs:
        print("ERROR: No data downloaded for any symbol.")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")

    if failed:
        print(f"WARNING: Failed symbols: {', '.join(failed)}")

    _print_summary(df)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    """Print row counts per symbol."""
    print("\nRow counts per symbol:")
    for symbol, grp in df.groupby("symbol"):
        dates = pd.to_datetime(grp["date"])
        print(f"  {symbol:8s}: {len(grp):5d} rows  ({dates.min().date()} to {dates.max().date()})")
    print(f"\nTotal: {len(df)} rows across {df['symbol'].nunique()} symbols")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download OHLCV data from Yahoo Finance")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--start", default=DOWNLOAD_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DOWNLOAD_END, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    download_ohlcv(
        symbols=UNIVERSE,
        start=args.start,
        end=args.end,
        force=args.force,
    )
