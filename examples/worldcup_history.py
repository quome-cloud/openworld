"""Historical World Cup backtest engine (2010 / 2014 / 2018 / 2022).

Computes leakage-free World Football Elo from the full results history
(datasets/openworld-football/results.csv), reuses the 2026 forecaster's
Elo->Poisson goal model in a 32-team format, and scores forecasts against real
results. Zero external deps (stdlib only).
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the IDENTICAL goal model + rating-agnostic standings from the forecaster.
sys.path.insert(0, os.path.dirname(__file__))
from worldcup2026 import sample_goals_from_elo, group_standings, _table  # noqa: E402,F401

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "openworld-football"
RESULTS_CSV = DATA_DIR / "results.csv"
SHOOTOUTS_CSV = DATA_DIR / "shootouts.csv"
PUBLISHED_ELO_CSV = DATA_DIR / "elo_ratings_wc2026.csv"

HOME_ADVANTAGE = 100.0   # World Football Elo home bump (0 on neutral ground)

# K-factor by match importance (World Football Elo conventions), keyed off the
# results.csv `tournament` column; default for anything unlisted.
K_BY_TOURNAMENT = {
    "FIFA World Cup": 60.0,
    "Confederations Cup": 50.0,
    "UEFA Euro": 50.0,
    "Copa América": 50.0,
    "African Cup of Nations": 50.0,
    "AFC Asian Cup": 50.0,
    "Gold Cup": 50.0,
    "FIFA World Cup qualification": 40.0,
    "UEFA Euro qualification": 40.0,
    "Copa América qualification": 40.0,
    "African Cup of Nations qualification": 40.0,
    "AFC Asian Cup qualification": 40.0,
    "Gold Cup qualification": 40.0,
    "UEFA Nations League": 40.0,
    "Friendly": 20.0,
}
K_DEFAULT = 30.0


def _k_for(tournament: str) -> float:
    return K_BY_TOURNAMENT.get(tournament, K_DEFAULT)


def _gd_multiplier(margin: int) -> float:
    """World Football Elo goal-difference weight G."""
    m = abs(margin)
    if m <= 1:
        return 1.0
    if m == 2:
        return 1.5
    return (11.0 + m) / 8.0


class EloEngine:
    """Incremental World Football Elo over a stream of dated matches."""

    def __init__(self, base: float = 1500.0):
        self.base = base
        self._r: Dict[str, float] = {}
        self._history: List[Tuple[str, Dict[str, float]]] = []  # (date, snapshot)

    def rating(self, team: str) -> float:
        return self._r.get(team, self.base)

    def update_match(self, home: str, away: str, hg: int, ag: int,
                     neutral: bool, k: float) -> None:
        ha = 0.0 if neutral else HOME_ADVANTAGE
        rh, ra = self.rating(home), self.rating(away)
        we_home = 1.0 / (1.0 + 10 ** (-((rh + ha) - ra) / 400.0))
        w_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        g = _gd_multiplier(hg - ag)
        delta = k * g * (w_home - we_home)
        self._r[home] = rh + delta
        self._r[away] = ra - delta

    @classmethod
    def from_results(cls, results_csv: Path, until: Optional[str] = None,
                     base: float = 1500.0) -> "EloEngine":
        """Replay all matches in date order (optionally only those with
        date < `until`), recording a dated snapshot after each match."""
        eng = cls(base=base)
        rows = []
        with open(results_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
        rows.sort(key=lambda r: r["date"])
        for r in rows:
            date = r["date"]
            if until is not None and date >= until:
                break
            try:
                hg, ag = int(r["home_score"]), int(r["away_score"])
            except (ValueError, KeyError):
                continue
            neutral = str(r["neutral"]).strip().upper() == "TRUE"
            eng.update_match(r["home_team"], r["away_team"], hg, ag,
                             neutral=neutral, k=_k_for(r["tournament"]))
            eng._history.append((date, dict(eng._r)))
        return eng

    def ratings_asof(self, date: str) -> Dict[str, float]:
        """Snapshot of team ratings using only matches strictly before `date`
        (no look-ahead). Only teams seen before then are present; callers
        default-fill unseen teams via .get(team, base)."""
        snap: Dict[str, float] = {}
        for d, s in self._history:
            if d >= date:
                break
            snap = s
        # default-fill is handled by callers via .get(team, base); return as-is
        return dict(snap)


# results.csv name -> published Elo file (`country`) name, where they differ.
NAME_TO_PUBLISHED = {
    "USA": "United States",
    "South Korea": "South Korea",      # same in both; explicit for clarity
    "China PR": "China",
    "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Ivory Coast",
    "Czech Republic": "Czechia",
}


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else 0.0


def _spearman(xs: List[float], ys: List[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    return _pearson(ranks(xs), ranks(ys))


def published_ratings(snapshot_year: int) -> Dict[str, float]:
    """Year-end published Elo for the given year, keyed by `country` name."""
    out: Dict[str, float] = {}
    with open(PUBLISHED_ELO_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["year"]) == snapshot_year and r["snapshot_date"].endswith("-12-31"):
                out[r["country"]] = float(r["rating"])
    return out


def validate_against_published(eng: "EloEngine", snapshot_year: int) -> dict:
    """Compare our computed end-of-year Elo to the published file on shared teams.

    Returns {n, pearson, spearman, rmse, snapshot_year} for teams present in both.
    """
    ours_all = eng.ratings_asof(f"{snapshot_year + 1}-01-01")  # end-of-year state
    pub = published_ratings(snapshot_year)
    xs, ys = [], []
    for team, ours in ours_all.items():
        key = NAME_TO_PUBLISHED.get(team, team)
        if key in pub:
            xs.append(ours)
            ys.append(pub[key])
    n = len(xs)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(xs, ys)) / n) if n else float("nan")
    return {"n": n, "pearson": _pearson(xs, ys), "spearman": _spearman(xs, ys),
            "rmse": rmse, "snapshot_year": snapshot_year}
