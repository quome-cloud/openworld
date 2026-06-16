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

import numpy as np
from scipy.optimize import minimize_scalar

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


import random as _random


def _host_of(year: int) -> str:
    return wh.load_cup(year).host


def predict_uniform(year: int, eng: "wh.EloEngine") -> Dict[tuple, Dict[str, float]]:
    third = 1 / 3
    return {(r["date"], r["home"], r["away"]): {"W": third, "D": third, "L": third}
            for r in cup_matches(year)}


def predict_ours_frozen(year: int, eng: "wh.EloEngine", sims: int = 20000,
                        seed: int = 2026) -> Dict[tuple, Dict[str, float]]:
    """Elo->Poisson W/D/L from FROZEN pre-tournament Elo (the E61 model)."""
    elo = eng.ratings_asof(wh._cup_freeze_date(year))
    host = _host_of(year)
    rng = _random.Random(seed)
    out = {}
    for r in cup_matches(year):
        out[(r["date"], r["home"], r["away"])] = wh._wdl_probs(
            r["home"], r["away"], elo, host, 1500.0, sims, rng)
    return out


def predict_ours_walk_forward(year: int, eng: "wh.EloEngine", sims: int = 20000,
                              seed: int = 2026) -> Dict[tuple, Dict[str, float]]:
    """Predict each match from Elo as of just before it, then update Elo with the
    real result (K = World Cup) — absorbs in-tournament info like 538 does."""
    elo = dict(eng.ratings_asof(wh._cup_freeze_date(year)))
    host = _host_of(year)
    base = 1500.0
    wc_k = wh._k_for("FIFA World Cup")
    rng = _random.Random(seed)
    out = {}
    for r in cup_matches(year):              # date-sorted
        h, a, hg, ag = r["home"], r["away"], r["hg"], r["ag"]
        out[(r["date"], h, a)] = wh._wdl_probs(h, a, elo, host, base, sims, rng)
        # update a local Elo copy (World Football Elo, same formula as the engine)
        ha = 0.0 if r["neutral"] else wh.HOME_ADVANTAGE
        rh, ra = elo.get(h, base), elo.get(a, base)
        we = 1.0 / (1.0 + 10 ** (-((rh + ha) - ra) / 400.0))
        w = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = wc_k * wh._gd_multiplier(hg - ag) * (w - we)
        elo[h] = rh + delta
        elo[a] = ra - delta
    return out


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


def _davidson_probs(elo_diff: float, nu: float) -> Dict[str, float]:
    f = 10 ** (elo_diff / 400.0)
    g = 10 ** (-elo_diff / 400.0)
    draw = max(nu, 0.0) * math.sqrt(f * g)
    z = f + g + draw
    return {"W": f / z, "D": draw / z, "L": g / z}


def fit_davidson_nu(year: int, eng: "wh.EloEngine") -> float:
    """MLE of the Davidson draw parameter on pre-cup internationals (leakage-free).

    Uses each training match's pre-match Elo gap from a fresh engine frozen at the
    cup's freeze date (ratings_asof), so no post-cup info leaks.
    """
    elo = eng.ratings_asof(wh._cup_freeze_date(year))
    tr = training_matches(year, years=4)
    gaps, outs = [], []
    for r in tr:
        ha = 0.0 if r["neutral"] else wh.HOME_ADVANTAGE
        d = (elo.get(r["home"], 1500.0) + ha) - elo.get(r["away"], 1500.0)
        gaps.append(d)
        outs.append("W" if r["hg"] > r["ag"] else ("D" if r["hg"] == r["ag"] else "L"))

    def neg_ll(log_nu: float) -> float:
        nu = math.exp(log_nu)
        s = 0.0
        for d, o in zip(gaps, outs):
            s += -math.log(max(_davidson_probs(d, nu)[o], 1e-12))
        return s

    res = minimize_scalar(neg_ll, bounds=(-6.0, 3.0), method="bounded")
    return math.exp(res.x)


def predict_elo_logistic(year: int, eng: "wh.EloEngine") -> Dict[tuple, Dict[str, float]]:
    elo = eng.ratings_asof(wh._cup_freeze_date(year))
    host = _host_of(year)
    nu = fit_davidson_nu(year, eng)
    out = {}
    for r in cup_matches(year):
        d = wh._eff(r["home"], elo, host, 1500.0) - wh._eff(r["away"], elo, host, 1500.0)
        out[(r["date"], r["home"], r["away"])] = _davidson_probs(d, nu)
    return out


def _poisson_wdl(lam_home: float, lam_away: float, max_goals: int = 10) -> Dict[str, float]:
    """W/D/L from two independent Poissons over a 0..max_goals score grid."""
    def pmf(lam, k):
        return math.exp(-lam) * lam ** k / math.factorial(k)
    ph = [pmf(lam_home, k) for k in range(max_goals + 1)]
    pa = [pmf(lam_away, k) for k in range(max_goals + 1)]
    w = d = l = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = ph[i] * pa[j]
            if i > j:
                w += p
            elif i == j:
                d += p
            else:
                l += p
    z = w + d + l
    return {"W": w / z, "D": d / z, "L": l / z}


def fit_maher(year: int, years: int = 4) -> dict:
    """MLE of a Maher independent-Poisson attack/defense model on the pre-cup window.

    Returns {teams, idx, atk, dee (defense), home (gamma), mu}. Leakage-free:
    training_matches only includes internationals before the cup's freeze date.
    """
    tr = training_matches(year, years=years)
    teams = sorted({r["home"] for r in tr} | {r["away"] for r in tr})
    idx = {t: i for i, t in enumerate(teams)}
    nt = len(teams)
    hi = np.array([idx[r["home"]] for r in tr])
    ai = np.array([idx[r["away"]] for r in tr])
    hg = np.array([r["hg"] for r in tr], dtype=float)
    ag = np.array([r["ag"] for r in tr], dtype=float)

    def unpack(x):
        return x[0], x[1], x[2:2 + nt], x[2 + nt:2 + 2 * nt]

    def neg_ll(x):
        mu, gamma, atk, dee = unpack(x)
        lh = np.exp(mu + gamma + atk[hi] - dee[ai])
        la = np.exp(mu + atk[ai] - dee[hi])
        ll = np.sum(hg * np.log(lh) - lh) + np.sum(ag * np.log(la) - la)
        pen = 1e3 * (atk.mean() ** 2 + dee.mean() ** 2)   # soft sum-to-zero
        return -ll + pen

    from scipy.optimize import minimize
    x0 = np.concatenate([[0.0, 0.2], np.zeros(nt), np.zeros(nt)])
    res = minimize(neg_ll, x0, method="L-BFGS-B")
    mu, gamma, atk, dee = unpack(res.x)
    return {"teams": teams, "idx": idx, "mu": float(mu), "home": float(gamma),
            "atk": atk, "dee": dee}


# 538 team name -> results.csv spelling, where they differ. Extended during impl
# until every cup's 64 538 rows align 1:1 to the real fixtures (the test is the gate).
FTE_NAME = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "USA": "United States",
    "China PR": "China",
}


def _fte_name(n: str) -> str:
    return FTE_NAME.get(n, n)


def predict_fte(year: int) -> Dict[tuple, Dict[str, float]]:
    """Load 538's per-match W/D/L probs, oriented to our (home=results.csv) keys.

    538 rows give team1/team2 + prob1/probtie/prob2. We match each to the real
    fixture by date + unordered team pair, and orient to the real home team.
    """
    # Key on date + unordered pair: a few cups repeat a pairing (group +
    # 3rd-place playoff, e.g. Belgium/England 2018, Croatia/Morocco 2022), so the
    # pair alone is not unique — the date disambiguates the two meetings.
    real = {(r["date"], frozenset((r["home"], r["away"]))):
            (r["date"], r["home"], r["away"]) for r in cup_matches(year)}
    out = {}
    path = FTE_DIR / f"wc_{year}.csv"
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t1, t2 = _fte_name(row["team1"]), _fte_name(row["team2"])
            p1, ptie, p2 = float(row["prob1"]), float(row["probtie"]), float(row["prob2"])
            key = real.get((row["date"], frozenset((t1, t2))))
            if key is None:
                raise ValueError(f"538 {year}: no real fixture for {t1} vs {t2} "
                                 f"(extend FTE_NAME)")
            _date, home, _away = key
            if home == t1:                       # 538 team1 == real home
                probs = {"W": p1, "D": ptie, "L": p2}
            else:                                # flip orientation
                probs = {"W": p2, "D": ptie, "L": p1}
            z = sum(probs.values())
            out[key] = {k: v / z for k, v in probs.items()}
    return out


def predict_maher(year: int, model: dict) -> Dict[tuple, Dict[str, float]]:
    idx, atk, dee = model["idx"], model["atk"], model["dee"]
    mu, gamma = model["mu"], model["home"]
    mean_atk = float(atk.mean()) if len(atk) else 0.0
    mean_dee = float(dee.mean()) if len(dee) else 0.0

    def a_of(t):
        return atk[idx[t]] if t in idx else mean_atk

    def d_of(t):
        return dee[idx[t]] if t in idx else mean_dee

    out = {}
    for r in cup_matches(year):
        h, a = r["home"], r["away"]
        lam_h = math.exp(mu + gamma + a_of(h) - d_of(a))
        lam_a = math.exp(mu + a_of(a) - d_of(h))
        out[(r["date"], h, a)] = _poisson_wdl(lam_h, lam_a)
    return out
