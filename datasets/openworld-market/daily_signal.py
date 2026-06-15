"""Live daily signal for the E50 same-day trading world model.

Pulls fresh OHLC via yfinance, runs the SAME verified estimator as the E50
backtest, and prints today's recommendation: which ticker to buy and at which
entry (open or close). It also LOGS each prediction to predictions.jsonl and
scores past predictions against what actually happened, so you can forward-test
the model day by day on real data.

  python datasets/openworld-market/daily_signal.py          # today's pick + score log
  python datasets/openworld-market/daily_signal.py --score  # just score the log

NOT investment advice. Daily direction is near a coin flip; the E50 backtest
shows the edge does not beat holding the index risk-adjusted and is cost-fragile.
This is a research/forward-test harness, nothing more.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / "predictions.jsonl"
WINDOW = 60
from fetch_prices import UNIVERSE  # noqa: E402


def fetch_recent(period="6mo"):
    import numpy as np
    import yfinance as yf
    data = yf.download(UNIVERSE, period=period, interval="1d",
                       auto_adjust=True, progress=False, group_by="ticker")
    series = {}
    for t in UNIVERSE:
        try:
            df = data[t][["Open", "Close"]].dropna()
        except Exception:
            continue
        series[t] = [(d.strftime("%Y-%m-%d"), float(r["Open"]), float(r["Close"]))
                     for d, r in df.iterrows()]
    return series


def recommend(series):
    """Same estimator as E50: trailing-mean of each segment, pick the ticker +
    segment with the best expected same-day move (no lookahead - uses history up
    to the latest close)."""
    import numpy as np
    best = None
    for t, rows in series.items():
        if t == "SPY" or len(rows) < WINDOW + 2:
            continue
        o = np.array([r[1] for r in rows]); c = np.array([r[2] for r in rows])
        r_id = c / o - 1
        r_on = np.full_like(c, np.nan); r_on[1:] = o[1:] / c[:-1] - 1
        id_mu = float(np.nanmean(r_id[-WINDOW:]))
        on_mu = float(np.nanmean(r_on[-WINDOW:]))
        for mu, seg in ((id_mu, "intraday"), (on_mu, "overnight")):
            if best is None or mu > best["expected_move"]:
                best = {"ticker": t, "segment": seg, "buy_at": "open" if seg == "intraday"
                        else "close", "expected_move": mu,
                        "p_up": float(np.mean((r_id if seg == "intraday" else r_on[1:])[-WINDOW:] > 0))}
    best["asof"] = series[best["ticker"]][-1][0]
    return best


def score_log(series):
    """Compare each logged prediction to the realized segment return."""
    if not LOG.exists():
        return {"scored": 0}
    px = {t: {d: (o, c) for d, o, c in rows} for t, rows in series.items()}
    dates = {t: [d for d, _, _ in rows] for t, rows in series.items()}
    hits = realized = n = 0
    for line in LOG.read_text().splitlines():
        p = json.loads(line)
        t, seg, asof = p["ticker"], p["segment"], p["asof"]
        ds = dates.get(t, [])
        if asof not in ds:
            continue
        i = ds.index(asof)
        # intraday: trade is on the NEXT session's open->close; overnight: its close->open
        if i + 1 >= len(ds):
            continue
        o0, c0 = px[t][ds[i]]
        o1, c1 = px[t][ds[i + 1]]
        r = (c1 / o1 - 1) if seg == "intraday" else (o1 / c0 - 1)
        n += 1
        realized += r
        hits += r > 0
    return {"scored": n, "hit_rate": round(hits / n, 3) if n else None,
            "avg_realized_bps": round(realized / n * 1e4, 2) if n else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    series = fetch_recent()
    sc = score_log(series)
    if sc["scored"]:
        print(f"forward-test so far: {sc['scored']} scored, hit rate "
              f"{sc['hit_rate']}, avg realized {sc['avg_realized_bps']}bps")
    if args.score:
        return
    rec = recommend(series)
    print(f"\n[{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}] recommendation "
          f"(as of {rec['asof']} close):")
    print(f"  buy {rec['ticker']} at the {rec['buy_at'].upper()}  "
          f"(segment: {rec['segment']}, expected {rec['expected_move']*1e4:.1f}bps, "
          f"P(up) {rec['p_up']:.2f})")
    print("  >> NOT investment advice; see datasets/openworld-market/CARD.md")
    with LOG.open("a") as f:
        f.write(json.dumps({k: rec[k] for k in ("asof", "ticker", "segment",
                                                "buy_at", "expected_move", "p_up")}) + "\n")
    print(f"  logged to {LOG.name} (run again tomorrow to forward-test)")


if __name__ == "__main__":
    main()
