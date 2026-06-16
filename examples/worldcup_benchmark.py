"""World Cup model benchmark (E62): RPS/Brier head-to-head of the OpenWorld
Elo->Poisson forecaster vs statistical baselines and FiveThirtyEight SPI.

Scores match-level W/D/L forecasts (home perspective) on 2010/2014/2018/2022 with
the Ranked Probability Score (the proper ordinal metric for football), plus Brier,
log-loss and hit-rate. 538 is compared on 2018 & 2022 only. Leakage-free: every
fitted model / rating uses only data strictly before each match or cup.

May use numpy/scipy (experiment-grade). worldcup_history.py stays stdlib-only.

    python experiments/e62_worldcup_benchmark.py
"""

from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
import worldcup_history as wh  # noqa: E402

FTE_DIR = Path(__file__).resolve().parents[1] / "datasets" / "fivethirtyeight"
CUPS = [2010, 2014, 2018, 2022]
FTE_CUPS = [2018, 2022]
_ORDER = ["L", "D", "W"]   # ordinal: away-win < draw < home-win (by home margin)


def _read_results() -> List[dict]:
    rows = []
    with open(wh.RESULTS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                hg, ag = int(r["home_score"]), int(r["away_score"])
            except (ValueError, KeyError):
                continue
            rows.append({"date": r["date"], "home": r["home_team"],
                         "away": r["away_team"], "hg": hg, "ag": ag,
                         "tournament": r["tournament"],
                         "neutral": str(r["neutral"]).strip().upper() == "TRUE"})
    rows.sort(key=lambda r: r["date"])
    return rows


def cup_matches(year: int) -> List[dict]:
    """The cup's 64 real matches (date-sorted), home perspective from results.csv."""
    return [r for r in _read_results()
            if r["tournament"] == "FIFA World Cup" and r["date"][:4] == str(year)]


def actual_outcomes(year: int) -> Dict[tuple, str]:
    """match_key (date, home, away) -> actual W/D/L from the home perspective."""
    out = {}
    for r in cup_matches(year):
        out[(r["date"], r["home"], r["away"])] = (
            "W" if r["hg"] > r["ag"] else ("D" if r["hg"] == r["ag"] else "L"))
    return out


def training_matches(year: int, years: int = 4) -> List[dict]:
    """All internationals strictly before the cup's freeze date, within `years`."""
    cutoff = wh._cup_freeze_date(year)
    lo = f"{year - years:04d}{cutoff[4:]}"
    return [r for r in _read_results() if lo <= r["date"] < cutoff]


def rps(probs: Dict[str, float], actual: str) -> float:
    """Ranked Probability Score for an ordinal W/D/L forecast (home perspective).

    Categories ordered [L, D, W] by home margin. 0 = perfect, larger = worse.
    """
    cum_p = cum_o = 0.0
    total = 0.0
    for cat in _ORDER[:-1]:          # r-1 = 2 cumulative terms
        cum_p += probs[cat]
        cum_o += 1.0 if cat == actual else 0.0
        total += (cum_p - cum_o) ** 2
    return total / (len(_ORDER) - 1)


def _brier(probs: Dict[str, float], actual: str) -> float:
    return sum((probs[c] - (1.0 if c == actual else 0.0)) ** 2 for c in "WDL")


def score_matches(predictions: Dict[tuple, Dict[str, float]],
                  actuals: Dict[tuple, str]) -> dict:
    """Aggregate one model's per-match probabilities vs actual outcomes.

    Only match_keys present in BOTH dicts are scored. Returns rps/brier/logloss/
    hit_rate/decisive_hit_rate/n (means over the scored matches).
    """
    keys = [k for k in predictions if k in actuals]
    n = len(keys)
    if n == 0:
        return {"n": 0, "rps": None, "brier": None, "logloss": None,
                "hit_rate": None, "decisive_hit_rate": None}
    rps_s = brier_s = ll_s = hits = 0.0
    dec_hits = dec_n = 0
    for k in keys:
        p, a = predictions[k], actuals[k]
        rps_s += rps(p, a)
        brier_s += _brier(p, a)
        ll_s += -math.log(max(p[a], 1e-12))
        fav = max(p, key=p.get)
        hits += fav == a
        if a != "D":
            dec_n += 1
            dec_hits += (p["W"] > p["L"]) if a == "W" else (p["L"] > p["W"])
    return {"n": n, "rps": rps_s / n, "brier": brier_s / n, "logloss": ll_s / n,
            "hit_rate": hits / n,
            "decisive_hit_rate": dec_hits / dec_n if dec_n else None}
