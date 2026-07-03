#!/usr/bin/env python3
"""
download_sentiment.py — Download GDELT GKG v1 data and compute daily market sentiment.

RESEARCH ONLY — NOT INVESTMENT ADVICE.

Strategy: download the first trading day of each month from the GDELT GKG v1 corpus,
filter to rows with ECON_STOCK_MARKET or ECON_MARKET themes, extract the tone field,
and compute a monthly sentiment index. Forward-fill to daily.

Full-resolution download (every trading day) is ~17GB and takes ~30 minutes.
Monthly sampling (15 files for validation period) is ~570MB and takes ~4 minutes.
"""

import sys
import csv
import io
import json
import time
import zipfile
import hashlib
from pathlib import Path
from datetime import date, timedelta, datetime

import requests
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SENTIMENT_DIR = DATA_DIR / "sentiment"
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

GDELT_GKG_V1_URL = "http://data.gdeltproject.org/gkg/{date}.gkg.csv.zip"
ECON_THEMES = {"ECON_STOCK", "ECON_MARKET", "ECON_STOCK_MARKET", "ECON_INTEREST_RATE",
               "ECON_FEDERAL_RESERVE", "ECON_BANKING_CRISIS", "ECON_RECESSION"}

# First Monday (or nearest weekday) of each month in the validation period
VALIDATION_SAMPLE_DATES = [
    "20240102", "20240201", "20240304", "20240401", "20240501", "20240603",
    "20240701", "20240801", "20240903", "20241001", "20241101", "20241202",
    "20250102", "20250203", "20250303",
]


def _parse_gkg_tone(tone_str: str) -> dict | None:
    """Parse GDELT GKG v1 TONE field: AvgTone,PosTone,NegTone,Polarity,ActivityRefDensity,SelfRefDensity.
    GKG v1 has NO WordCount in the TONE field (that's a GKG v2 feature).
    AvgTone (index 0) = PosTone - NegTone (confirmed: 5.62-0.64≈4.98, Polarity=5.62+0.64=6.26).
    """
    parts = tone_str.split(",")
    if len(parts) < 3:
        return None
    try:
        return {
            "avg_tone": float(parts[0]),  # AvgTone = PosTone - NegTone
            "pos": float(parts[1]),
            "neg": float(parts[2]),
        }
    except (ValueError, IndexError):
        return None


def download_gkg_day(date_str: str, force: bool = False) -> dict | None:
    """
    Download and parse one GDELT GKG v1 daily file.
    Returns dict with date, avg_tone, n_articles; None on failure.
    """
    cache_path = SENTIMENT_DIR / f"gdelt_{date_str}.json"
    if cache_path.exists() and not force:
        with open(cache_path) as f:
            return json.load(f)

    url = GDELT_GKG_V1_URL.format(date=date_str)
    print(f"  Downloading {url} ...", end=" ", flush=True)

    try:
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

    print(f"{len(r.content)/1024/1024:.1f}MB", end=" ")

    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            fname = z.namelist()[0]
            with z.open(fname) as f:
                content = f.read().decode("latin-1", errors="replace")
    except Exception as e:
        print(f"Zip error: {e}")
        return None

    lines = content.split("\n")
    tones = []

    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        themes = parts[3].upper()
        if not any(t in themes for t in ECON_THEMES):
            continue
        tone = _parse_gkg_tone(parts[7])
        if tone:
            tones.append(tone["avg_tone"])

    if not tones:
        print("0 rows")
        return None

    avg_tone = float(np.mean(tones))
    result = {
        "date": date_str,
        "avg_tone": avg_tone,
        "n_articles": len(tones),
        "tone_std": float(np.std(tones)),
    }
    with open(cache_path, "w") as f:
        json.dump(result, f)
    print(f"→ tone={avg_tone:.3f}, n={len(tones)}")
    return result


def build_daily_sentiment(
    output_path: Path = SENTIMENT_DIR / "daily_sentiment.csv",
    sample_dates: list[str] = VALIDATION_SAMPLE_DATES,
    sleep_between: float = 2.0,
    force: bool = False,
) -> pd.DataFrame:
    """
    Download monthly sample of GDELT GKG data and build daily sentiment series
    by forward-filling monthly values to trading days.
    """
    if output_path.exists() and not force:
        print(f"Sentiment data found at {output_path}")
        return pd.read_csv(output_path, parse_dates=["date"], index_col="date")

    print(f"Downloading GDELT GKG v1 samples ({len(sample_dates)} files)...")
    monthly = {}
    for i, ds in enumerate(sample_dates):
        result = download_gkg_day(ds, force=False)
        if result:
            monthly[ds] = result["avg_tone"]
        if i < len(sample_dates) - 1:
            time.sleep(sleep_between)

    if not monthly:
        raise RuntimeError("No GDELT data downloaded — check connectivity")

    # Build a date-indexed series and forward-fill to daily
    monthly_series = pd.Series(
        {datetime.strptime(k, "%Y%m%d"): v for k, v in monthly.items()},
        name="gdelt_tone",
    )
    monthly_series.index = pd.to_datetime(monthly_series.index)
    monthly_series = monthly_series.sort_index()

    # Create daily index from first to last date
    daily_idx = pd.date_range(monthly_series.index[0], monthly_series.index[-1], freq="B")
    daily = monthly_series.reindex(daily_idx).ffill()

    # Z-score normalize
    daily_z = (daily - daily.mean()) / daily.std()

    df = pd.DataFrame({"gdelt_tone": daily, "gdelt_tone_z": daily_z})
    df.index.name = "date"
    df.to_csv(output_path)
    print(f"Saved daily sentiment ({len(df)} days) to {output_path}")
    return df


if __name__ == "__main__":
    print("Building GDELT market sentiment signal for validation period...")
    df = build_daily_sentiment()
    print(df.describe())
    print("\nFirst/last rows:")
    print(df.head(3))
    print(df.tail(3))
