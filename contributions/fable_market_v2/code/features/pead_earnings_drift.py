#!/usr/bin/env python3
"""FEATURE FAMILY: PEAD — Post-earnings-announcement drift.

RESEARCH ONLY — paper trading / analysis. No live trading.

Economic rationale (causal story, written before fitting):
  Prices underreact to earnings surprises: after a positive (negative) EPS
  surprise, returns drift in the surprise's direction for weeks, because
  investors anchor on prior earnings levels and update slowly (Ball & Brown
  1968; Bernard & Thomas 1989). The drift is a structural property of the
  announcement EVENT, not a chart pattern; it decays over ~1 quarter.

Feature:
  pead_drift_i(t) = s_i * w(d),  for 1 <= d <= 60 trading days since the
                    announcement became tradeable, else 0 (announcement events
                    are sparse; non-event = neutral, not missing).
    s_i  = surprise% at i's most recent announcement, winsorized at +/-100,
           then divided by 100 (units: fractional EPS surprise, capped).
           surprise% = 100*(reported EPS - consensus estimate)/|estimate|
           as provided by Yahoo Finance.
    w(d) = 1 - d/60  (linear decay to zero over 60 trading days).
  Effective date: announcements timestamped after 15:59 local exchange time
  are tradeable the NEXT trading day; earlier timestamps same day. The first
  drift day d=1 is the first date the feature is nonzero, i.e. positions can
  react no earlier than the first close at which the number was public.

Data dependencies: data/earnings/<SYM>.csv cache built by fetch_earnings.py
  (Yahoo Finance get_earnings_dates; source + fetch timestamp in
  data/earnings_log.json). Symbols with no earnings data get all-zero
  feature (they simply never contribute PEAD signal).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

FAMILY = "PEAD"
FEATURES = ["pead_drift"]

DRIFT_DAYS = 60
WINSOR = 100.0

EARNINGS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "earnings"


def _events(sym: str, dates: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, float]]:
    f = EARNINGS_DIR / f"{sym}.csv"
    if not f.exists():
        return []
    df = pd.read_csv(f)
    if df.empty or "surprise_pct" not in df.columns:
        return []
    out = []
    for _, row in df.iterrows():
        if pd.isna(row["surprise_pct"]):
            continue
        ts = pd.Timestamp(row["timestamp"])
        day = ts.tz_localize(None).normalize()
        # after-market-close (>=16:00) -> tradeable next trading day
        effective = day + pd.Timedelta(days=1) if ts.hour >= 16 else day
        pos = dates.searchsorted(effective)
        if pos >= len(dates):
            continue
        s = float(np.clip(row["surprise_pct"], -WINSOR, WINSOR)) / 100.0
        out.append((dates[pos], s))
    return out


def compute(panel) -> dict[str, pd.DataFrame]:
    dates = panel.dates
    arr = np.zeros((len(dates), len(panel.symbols)))
    w = 1.0 - np.arange(1, DRIFT_DAYS + 1) / DRIFT_DAYS
    for ci, sym in enumerate(panel.symbols):
        for eff_date, s in _events(sym, dates):
            i0 = dates.get_loc(eff_date)
            n = min(DRIFT_DAYS, len(dates) - i0)
            arr[i0:i0 + n, ci] += s * w[:n]
    df = pd.DataFrame(arr, index=dates, columns=panel.symbols)
    return {"pead_drift": df.where(panel.close.notna())}
