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
from worldcup2026 import sample_goals_from_elo, group_standings, _table, HOST_ADVANTAGE  # noqa: E402,F401

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "openworld-football"
RESULTS_CSV = DATA_DIR / "results.csv"
SHOOTOUTS_CSV = DATA_DIR / "shootouts.csv"
PUBLISHED_ELO_CSV = DATA_DIR / "elo_ratings_wc2026.csv"

HOME_ADVANTAGE = 100.0   # Elo-engine home bump for computing ratings from results.csv.

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


# Official group draws (results.csv spellings). Verified against real fixtures.
CUP_GROUPS: Dict[int, Dict[str, List[str]]] = {
    2010: {
        "A": ["South Africa", "Mexico", "Uruguay", "France"],
        "B": ["Argentina", "Nigeria", "South Korea", "Greece"],
        "C": ["England", "United States", "Algeria", "Slovenia"],
        "D": ["Germany", "Australia", "Serbia", "Ghana"],
        "E": ["Netherlands", "Denmark", "Japan", "Cameroon"],
        "F": ["Italy", "Paraguay", "New Zealand", "Slovakia"],
        "G": ["Brazil", "North Korea", "Ivory Coast", "Portugal"],
        "H": ["Spain", "Switzerland", "Honduras", "Chile"],
    },
    2014: {
        "A": ["Brazil", "Croatia", "Mexico", "Cameroon"],
        "B": ["Spain", "Netherlands", "Chile", "Australia"],
        "C": ["Colombia", "Greece", "Ivory Coast", "Japan"],
        "D": ["Uruguay", "Costa Rica", "England", "Italy"],
        "E": ["Switzerland", "Ecuador", "France", "Honduras"],
        "F": ["Argentina", "Bosnia and Herzegovina", "Iran", "Nigeria"],
        "G": ["Germany", "Portugal", "Ghana", "United States"],
        "H": ["Belgium", "Algeria", "Russia", "South Korea"],
    },
    2018: {
        "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
        "B": ["Portugal", "Spain", "Morocco", "Iran"],
        "C": ["France", "Australia", "Peru", "Denmark"],
        "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
        "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
        "F": ["Germany", "Mexico", "Sweden", "South Korea"],
        "G": ["Belgium", "Panama", "Tunisia", "England"],
        "H": ["Poland", "Senegal", "Colombia", "Japan"],
    },
    2022: {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "United States", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    },
}

# Host nation per cup (gets the forecaster's host Elo bump in simulation).
CUP_HOST = {2010: "South Africa", 2014: "Brazil", 2018: "Russia", 2022: "Qatar"}

# Fixed Round-of-16 pairing: (group-winner letter, group-runnerup letter).
R16_PAIRS = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H"),
             ("B", "A"), ("D", "C"), ("F", "E"), ("H", "G")]


class Cup:
    """One World Cup's draw + real results, loaded from the vendored data."""

    def __init__(self, year: int):
        self.year = year
        self.groups = CUP_GROUPS[year]
        self.host = CUP_HOST[year]
        self._team_to_group = {t: g for g, ts in self.groups.items() for t in ts}
        self._group_res: Dict[frozenset, Tuple[str, int, int]] = {}
        self._ko: List[dict] = []
        self._shootouts: Dict[frozenset, str] = {}
        self._load()

    def _load(self):
        with open(SHOOTOUTS_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["date"][:4] == str(self.year):
                    self._shootouts[frozenset((r["home_team"], r["away_team"]))] = r["winner"]
        with open(RESULTS_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["tournament"] != "FIFA World Cup" or r["date"][:4] != str(self.year):
                    continue
                h, a = r["home_team"], r["away_team"]
                hg, ag = int(r["home_score"]), int(r["away_score"])
                same_group = (self._team_to_group.get(h) is not None
                              and self._team_to_group.get(h) == self._team_to_group.get(a))
                if same_group:
                    self._group_res[frozenset((h, a))] = (h, hg, ag)
                else:
                    key = frozenset((h, a))
                    winner = h if hg > ag else (a if ag > hg else self._shootouts.get(key))
                    self._ko.append({"date": r["date"], "home": h, "away": a,
                                     "hg": hg, "ag": ag, "winner": winner})
        self._ko.sort(key=lambda m: m["date"])

    def group_result(self, a: str, b: str) -> Optional[Tuple[str, int, int]]:
        """(home, home_goals, away_goals) for the real group match, or None."""
        return self._group_res.get(frozenset((a, b)))

    def group_of(self, team: str) -> str:
        return self._team_to_group[team]

    def knockout_matches(self) -> List[dict]:
        """The cup's real knockout matches, date-sorted. Each is a dict with
        keys home, away, hg, ag, winner, date."""
        return list(self._ko)

    def real_standings(self) -> Dict[str, List[str]]:
        """Real finishing order per group (reuses the forecaster's tiebreak)."""
        out = {}
        for g, teams in self.groups.items():
            res = {}
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    rec = self.group_result(teams[i], teams[j])
                    if rec is None:
                        raise ValueError(
                            f"{self.year} group {g}: missing match {teams[i]} vs {teams[j]}")
                    home, hg, ag = rec
                    away = teams[j] if home == teams[i] else teams[i]
                    res[(home, away)] = (hg, ag)
            out[g] = group_standings(teams, res)
        return out

    def actual_advancers(self) -> Dict[str, object]:
        """Round-name -> advancing teams, derived from the real KO games (date-sorted).

        R16 = winners of the first 8 KO matches, QF = next 4, SF = next 2. The 3rd-place
        playoff is present for some cups (2010, 2014) and absent in the vendored data for
        others (2018, 2022), so the final is identified by predicate — the last match
        whose BOTH teams won their semi-final — rather than by a fixed match count.
        champion = winner of that final.
        """
        ko = self._ko
        r16 = ko[:8]; qf = ko[8:12]; sf = ko[12:14]; last_two = ko[14:16]
        sf_winners = {m["winner"] for m in sf}
        final = next((m for m in last_two
                      if m["home"] in sf_winners and m["away"] in sf_winners), None)
        if final is None:
            raise ValueError(f"could not identify the {self.year} final from KO data")
        return {
            "R16": [m["winner"] for m in r16],
            "QF": [m["winner"] for m in qf],
            "SF": [m["winner"] for m in sf],
            "final_match": final,
            "champion": final["winner"],
        }

    def actual_champion(self) -> str:
        return self.actual_advancers()["champion"]


def load_cup(year: int) -> Cup:
    return Cup(year)


KO_ROUNDS = ["R16", "QF", "SF", "final"]


# Simulation host bump uses the forecaster's HOST_ADVANTAGE (distinct from the
# Elo-engine's HOME_ADVANTAGE above, which is for rating computation).
def _eff(team: str, elo: Dict[str, float], host: str, base: float) -> float:
    return elo.get(team, base) + (HOST_ADVANTAGE if team == host else 0.0)


def _sample(home, away, elo, host, base, rng):
    return sample_goals_from_elo(_eff(home, elo, host, base),
                                 _eff(away, elo, host, base), rng)


def _ko_match(home, away, elo, host, base, rng):
    hg, ag = _sample(home, away, elo, host, base, rng)
    if hg != ag:
        return (home if hg > ag else away), hg, ag, False
    dh = _eff(home, elo, host, base) - _eff(away, elo, host, base)
    p_home = 1.0 / (1.0 + 10 ** (-dh / 400.0))
    return (home if rng.random() < p_home else away), hg, ag, True


def simulate_cup_once(cup: "Cup", elo: Dict[str, float], rng: random.Random,
                      base: float = 1500.0) -> dict:
    """One full 32-team tournament. Returns reached-round per team + champion."""
    reached = {t: "group" for g in cup.groups.values() for t in g}
    winners, runners = {}, {}
    for g, teams in cup.groups.items():
        res = {}
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                h, a = teams[i], teams[j]
                res[(h, a)] = _sample(h, a, elo, cup.host, base, rng)
        order = group_standings(teams, res)
        winners[g], runners[g] = order[0], order[1]
    # Seed R16 in fixed bracket order so the binary tree pairs neighbours.
    seeds = []
    for wl, rl in R16_PAIRS:
        seeds.append(winners[wl]); seeds.append(runners[rl])
    for t in seeds:
        reached[t] = "R16"
    teams = seeds
    for rnd in KO_ROUNDS:
        nxt = []
        for i in range(0, len(teams), 2):
            w, _hg, _ag, _p = _ko_match(teams[i], teams[i + 1], elo, cup.host, base, rng)
            nxt.append(w)
        nxt_round = KO_ROUNDS[KO_ROUNDS.index(rnd) + 1] if rnd != "final" else "champion"
        for w in nxt:
            reached[w] = nxt_round
        teams = nxt
    return {"champion": teams[0], "reached": reached}


_REACH_ORDER = ["group", "R16", "QF", "SF", "final", "champion"]


def _cup_freeze_date(year: int) -> str:
    """Day-before-opener freeze date per cup (no look-ahead)."""
    return {2010: "2010-06-11", 2014: "2014-06-12",
            2018: "2018-06-14", 2022: "2022-11-20"}[year]


def forecast_cup(year: int, eng: "EloEngine", sims: int = 10000, seed: int = 2026,
                 base: float = 1500.0) -> Dict[str, Dict[str, float]]:
    """Monte-Carlo a cup from FROZEN pre-tournament Elo. Per-team probabilities (%)."""
    cup = load_cup(year)
    elo = eng.ratings_asof(_cup_freeze_date(year))
    teams = [t for g in cup.groups.values() for t in g]
    titles = {t: 0 for t in teams}
    reach = {t: {r: 0 for r in _REACH_ORDER} for t in teams}
    for s in range(sims):
        rng = random.Random(seed * 1_000_003 + s)
        out = simulate_cup_once(cup, elo, rng, base=base)
        titles[out["champion"]] += 1
        for t, r in out["reached"].items():
            idx = _REACH_ORDER.index(r)
            for ri in range(1, idx + 1):
                reach[t][_REACH_ORDER[ri]] += 1
    # Keep exact floats (no rounding) so champion probabilities sum to exactly 100%.
    def pct(n): return 100.0 * n / sims
    return {t: {"champion": pct(titles[t]),
                "reach_final": pct(reach[t]["final"]),
                "reach_SF": pct(reach[t]["SF"]),
                "reach_QF": pct(reach[t]["QF"]),
                "reach_R16": pct(reach[t]["R16"])} for t in teams}


def _wdl_probs(home, away, elo, host, base, sims, rng) -> Dict[str, float]:
    w = d = 0
    for _ in range(sims):
        hg, ag = _sample(home, away, elo, host, base, rng)
        if hg > ag:
            w += 1
        elif hg == ag:
            d += 1
    return {"W": w / sims, "D": d / sims, "L": (sims - w - d) / sims}


def score_group_matches(cup: "Cup", elo: Dict[str, float], sims: int = 30000,
                        seed: int = 2026, base: float = 1500.0):
    """W/D/L Brier/hit-rate/skill over a cup's 48 group matches. Returns (rows, summary)."""
    rng = random.Random(seed)
    rows = []
    hits = p_sum = brier_sum = 0.0
    dec_hits = dec_n = draws = 0
    matches = []
    for g, teams in cup.groups.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                rec = cup.group_result(teams[i], teams[j])
                home, hg, ag = rec
                away = teams[j] if home == teams[i] else teams[i]
                matches.append((g, home, away, hg, ag))
    for (g, home, away, hg, ag) in matches:
        probs = _wdl_probs(home, away, elo, cup.host, base, sims, rng)
        actual = "W" if hg > ag else ("D" if hg == ag else "L")
        fav = max(probs, key=probs.get)
        brier = sum((probs[c] - (1.0 if c == actual else 0.0)) ** 2 for c in "WDL")
        hits += fav == actual; p_sum += probs[actual]; brier_sum += brier
        if actual == "D":
            draws += 1
        else:
            dec_n += 1
            dec_hits += (probs["W"] > probs["L"]) if actual == "W" else (probs["L"] > probs["W"])
        rows.append({"group": g, "home": home, "away": away, "score": f"{hg}-{ag}",
                     "actual": actual, "probs": probs, "hit": fav == actual, "brier": brier})
    n = len(matches)
    base_brier = 2 / 3
    return rows, {"n": n, "draws": draws, "hit_rate": hits / n,
                  "decisive_n": dec_n,
                  "decisive_hit_rate": dec_hits / dec_n if dec_n else 0.0,
                  "mean_p_actual": p_sum / n, "mean_brier": brier_sum / n,
                  "baseline_brier": base_brier,
                  "skill_vs_uniform": 1 - (brier_sum / n) / base_brier}


def score_knockout_advancement(cup: "Cup", elo: Dict[str, float], sims: int = 30000,
                               seed: int = 2026, base: float = 1500.0):
    """For each real KO match, model P(home advances) vs the actual advancer."""
    rng = random.Random(seed)
    rows = []
    correct = brier_sum = logloss_sum = 0.0
    n = 0
    for m in cup.knockout_matches():
        home, away, winner = m["home"], m["away"], m["winner"]
        if winner is None:
            continue
        adv = sum(_ko_match(home, away, elo, cup.host, base, rng)[0] == home
                  for _ in range(sims))
        p_home = adv / sims
        p_home = min(max(p_home, 1e-6), 1 - 1e-6)
        actual_home = 1.0 if winner == home else 0.0
        pick_home = p_home >= 0.5
        correct += (pick_home and actual_home) or ((not pick_home) and not actual_home)
        brier_sum += (p_home - actual_home) ** 2
        logloss_sum += -(actual_home * math.log(p_home) + (1 - actual_home) * math.log(1 - p_home))
        n += 1
        rows.append({"home": home, "away": away, "winner": winner,
                     "p_home_adv": p_home, "correct": pick_home == bool(actual_home)})
    return {"n": n, "accuracy": correct / n, "brier": brier_sum / n,
            "logloss": logloss_sum / n, "rows": rows}
