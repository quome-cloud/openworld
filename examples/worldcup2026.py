"""FIFA World Cup 2026 — a deterministic Monte-Carlo forecaster built as an
OpenWorld world model.

Start fresh (no real results applied): every match is predicted from team
strength. Match outcomes come from Elo ratings (eloratings.net, June 2026)
turned into a coded Poisson goal model. The whole tournament — 48 teams, 12
groups, top-2 + 8 best thirds into a Round of 32, then a single-elimination
bracket to the Final — is a plain rules engine. We roll it out thousands of
times to get each team's title and round-by-round probabilities.

The SAME rules functions back two things: the `forecast()` Monte-Carlo driver,
and an OpenWorld `World` (verified `FunctionTransition`) that serialises to a
spec and serves interactively at `/view`. No duplicated logic.

    python examples/worldcup2026.py                # print the forecast table
    python examples/worldcup2026.py --sims 20000   # more simulations
    python examples/worldcup2026.py --serve        # write a spec for `openworld serve`

Zero external deps — stdlib only (Elo->Poisson via Knuth, stdlib `random`).
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Data: PRE-TOURNAMENT Elo ratings (eloratings.net, as of 10 June 2026, the day
# before the opening match) and the final group draw (5 Dec 2025), with the five
# playoff-winner slots resolved (A+Czechia, B+Bosnia, D+Turkey, F+Sweden,
# I+Iraq) and K's third = DR Congo.
#
# Why pre-tournament: eloratings.net's *live* numbers already absorb matchday-1
# World Cup results, so forecasting with them would leak the outcomes we want to
# predict. The 28 teams that had played by 15 June were rolled back by their
# matchday-1 point swing (pre = live - Delta, from the "Latest Results" page);
# the other 20 had not played, so their live rating already IS the pre-tournament
# value. Corroborated: recovered Spain 2157 / Brazil 1991 match the published
# 11 June pre-tournament ratings exactly.
# Edit these tables to correct teams or plug in your own ratings.
# --------------------------------------------------------------------------- #

ELO: Dict[str, float] = {
    "Spain": 2157, "Argentina": 2115, "France": 2063, "England": 2024,
    "Brazil": 1991, "Portugal": 1989, "Colombia": 1982, "Norway": 1914,
    "Turkey": 1911, "Czechia": 1684, "Germany": 1932, "Netherlands": 1948,
    "Croatia": 1912, "Japan": 1906, "Uruguay": 1892, "Belgium": 1894,
    "Switzerland": 1891, "Senegal": 1860, "Ecuador": 1938, "Morocco": 1827,
    "Austria": 1830, "South Korea": 1814, "Paraguay": 1834, "Mexico": 1875,
    "United States": 1726, "Algeria": 1772, "Iran": 1772, "Canada": 1788,
    "Sweden": 1712, "Ivory Coast": 1695, "Panama": 1730, "Uzbekistan": 1714,
    "Egypt": 1696, "Jordan": 1680, "DR Congo": 1652, "Bosnia": 1595,
    "Iraq": 1607, "Cabo Verde": 1578, "Tunisia": 1628, "Australia": 1777,
    "Saudi Arabia": 1576, "New Zealand": 1562, "Haiti": 1548,
    "South Africa": 1517, "Ghana": 1510, "Qatar": 1421, "Curacao": 1434,
    # Reconstructed: eloratings.net live rating 1794 (rank 25, 15 June), minus
    # its matchday-1 win over Haiti (Delta ~+12) -> 1782 pre-tournament.
    "Scotland": 1782,
}

GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Australia", "Paraguay", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curacao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Cabo Verde", "Uruguay", "Saudi Arabia"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

HOSTS = {"United States", "Mexico", "Canada"}

# Round of 32 (official ESPN pairings). Each slot is one of:
#   ("W", group)  winner of a group
#   ("R", group)  runner-up of a group
#   ("3", "CEFHI") a best-third from one of the listed allowed groups
# Order in this list defines the bracket: matches pair up (0,1),(2,3),... at
# each subsequent round (a fixed binary tree — see _knockout_tree).
R32: List[Tuple[tuple, tuple]] = [
    (("W", "A"), ("3", "CEFHI")),
    (("R", "A"), ("R", "B")),
    (("W", "B"), ("3", "EFGIJ")),
    (("W", "C"), ("R", "F")),
    (("R", "C"), ("W", "F")),
    (("W", "D"), ("3", "BEFIJ")),
    (("R", "D"), ("R", "G")),
    (("W", "E"), ("3", "CDFGH")),
    (("R", "E"), ("R", "I")),
    (("W", "G"), ("3", "AEHIJ")),
    (("W", "H"), ("R", "J")),
    (("W", "I"), ("3", "CDFGH")),
    (("W", "J"), ("R", "H")),
    (("W", "K"), ("3", "DEIJL")),
    (("W", "L"), ("3", "EHIJK")),
    (("R", "K"), ("R", "L")),
]

ROUND_NAMES = ["group", "R32", "R16", "QF", "SF", "final", "champion"]

# Tunable model dials (see sample_match). Defaults reflect typical WC scoring.
TOTAL_GOALS = 2.7      # mean combined goals per match
SUPREMACY = 1.9        # how strongly an Elo edge converts to a goal margin
HOST_ADVANTAGE = 60.0  # Elo bump for a host nation (applied in every match)

# Real matchday-1 results (official, as of 15 June 2026). Used ONLY by the
# accuracy backtest (evaluate_predictions); NEVER fed into the clean
# pre-tournament forecast. Because ELO is the 10 June (pre-tournament) snapshot,
# scoring the model against these is a genuine out-of-sample test.
RESULTS_TO_DATE: List[Tuple[str, str, str, int, int]] = [
    ("A", "Mexico", "South Africa", 2, 0),
    ("A", "South Korea", "Czechia", 2, 1),
    ("B", "Canada", "Bosnia", 1, 1),
    ("B", "Qatar", "Switzerland", 1, 1),
    ("C", "Brazil", "Morocco", 1, 1),
    ("C", "Haiti", "Scotland", 0, 1),
    ("D", "United States", "Paraguay", 4, 1),
    ("D", "Australia", "Turkey", 2, 0),
    ("E", "Germany", "Curacao", 7, 1),
    ("E", "Ivory Coast", "Ecuador", 1, 0),
    ("F", "Netherlands", "Japan", 2, 2),
    ("F", "Sweden", "Tunisia", 5, 1),
    ("G", "Belgium", "Egypt", 1, 1),
    ("H", "Spain", "Cabo Verde", 0, 0),
    ("H", "Saudi Arabia", "Uruguay", 1, 1),
]


# --------------------------------------------------------------------------- #
# Outcome model: Elo -> expected score -> Poisson goals.
# --------------------------------------------------------------------------- #

def _eff_elo(team: str) -> float:
    return ELO[team] + (HOST_ADVANTAGE if team in HOSTS else 0.0)


def _poisson(lam: float, rng: random.Random) -> int:
    """Knuth's Poisson sampler (stdlib only)."""
    lam = max(lam, 1e-6)
    el = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= el:
            return k - 1


def sample_match(home: str, away: str, rng: random.Random) -> Tuple[int, int]:
    """Sample (home_goals, away_goals) from the Elo->Poisson model."""
    diff = _eff_elo(home) - _eff_elo(away)
    expected = 1.0 / (1.0 + 10 ** (-diff / 400.0))   # home expected score in [0,1]
    supremacy = SUPREMACY * (2 * expected - 1)        # >0 favours home
    lam_home = max(TOTAL_GOALS / 2 + supremacy / 2, 0.05)
    lam_away = max(TOTAL_GOALS / 2 - supremacy / 2, 0.05)
    return _poisson(lam_home, rng), _poisson(lam_away, rng)


def _knockout_match(home: str, away: str, rng: random.Random):
    """Single-elimination result; ties go to an Elo-leaning penalty flip.

    Returns (winner, home_goals, away_goals, went_to_penalties).
    """
    hg, ag = sample_match(home, away, rng)
    if hg > ag:
        return home, hg, ag, False
    if ag > hg:
        return away, hg, ag, False
    diff = _eff_elo(home) - _eff_elo(away)
    p_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))      # shootout, slight Elo lean
    winner = home if rng.random() < p_home else away
    return winner, hg, ag, True


def _knockout_winner(home: str, away: str, rng: random.Random) -> str:
    """Just the winner (thin wrapper over _knockout_match)."""
    return _knockout_match(home, away, rng)[0]


# --------------------------------------------------------------------------- #
# Rules engine (pure, deterministic) — shared by the Monte-Carlo driver and the
# served World.
# --------------------------------------------------------------------------- #

def _table(teams: List[str], results: Dict[Tuple[str, str], Tuple[int, int]]) -> Dict[str, tuple]:
    """Per-team (points, goal-difference, goals-for, goals-against) for a group."""
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    for (h, a), (hg, ag) in results.items():
        gf[h] += hg; ga[h] += ag; gf[a] += ag; ga[a] += hg
        if hg > ag:
            pts[h] += 3
        elif ag > hg:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1
    return {t: (pts[t], gf[t] - ga[t], gf[t], ga[t]) for t in teams}


def group_standings(teams: List[str], results: Dict[Tuple[str, str], Tuple[int, int]]) -> List[str]:
    """Order a group's teams by points -> GD -> GF -> goals-against (desc).

    `results` maps (home, away) -> (home_goals, away_goals) for that group.
    A stable, deterministic ordering (ties broken by name as a final fallback).
    """
    tbl = _table(teams, results)
    return sorted(
        teams,
        key=lambda t: (tbl[t][0], tbl[t][1], tbl[t][2], -tbl[t][3], t),
        reverse=True,
    )


def _round_robin(
    teams: List[str],
    rng: random.Random,
    group: str = "",
    fixed: Dict[Tuple[str, str, str], Tuple[int, int]] = None,
) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """Play a group's six matches, using any pre-supplied real results in
    `fixed` (keyed by (group, home, away)) and sampling the rest."""
    fixed = fixed or {}
    res = {}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            h, a = teams[i], teams[j]
            res[(h, a)] = fixed.get((group, h, a)) or sample_match(h, a, rng)
    return res


def rank_thirds(thirds: Dict[str, Tuple[str, dict]]) -> List[str]:
    """Best 8 third-placed teams across all 12 groups.

    `thirds` maps group -> (team, {points, gd, gf}). Returns the team names of
    the top 8 by points -> GD -> GF (FIFA's third-place criteria).
    """
    ranked = sorted(
        thirds.items(),
        key=lambda kv: (kv[1][1]["points"], kv[1][1]["gd"], kv[1][1]["gf"], kv[0]),
        reverse=True,
    )
    return [team for _g, (team, _stat) in ranked[:8]]


def assign_thirds(qualified_groups: List[str], third_by_group: Dict[str, str]) -> List[str]:
    """Assign the 8 qualifying thirds to the 8 third-place SLOTS, in bracket order.

    Deterministic greedy fill over each slot's allowed-group set (a documented
    approximation of FIFA's 495-row combination table; second-order effect on
    title odds). Slots are filled in R32 order; each takes the highest-ranked
    still-unassigned third whose group is allowed for that slot. Returns one
    team per "3" slot, aligned to their order of appearance in `R32` (note two
    slots can share the same allowed-group string, so we return a list, not a
    dict keyed by that string).
    """
    slots = [b for match in R32 for b in match if b[0] == "3"]  # in bracket order
    remaining = list(qualified_groups)  # already ranked best->worst
    out: List[str] = []
    for slot in slots:
        allowed = set(slot[1])
        pick = next((g for g in remaining if g in allowed), None)
        if pick is None:  # fallback: any remaining (keeps the bracket complete)
            pick = remaining[0]
        remaining.remove(pick)
        out.append(third_by_group[pick])
    return out


def _knockout_tree(seeds: List[str], rng: random.Random, record: list = None) -> Tuple[str, Dict[str, int]]:
    """Play a single-elimination bracket over `seeds` (len a power of two).

    Returns (champion, {team: furthest_round_index}) where round indices follow
    ROUND_NAMES (R32=1 ... champion=6). Pairs adjacent entries each round. If
    `record` is a list, each round is appended as (round_name, [matches]) where
    a match is (home, away, home_goals, away_goals, winner, went_to_pens).
    """
    reached: Dict[str, int] = {}
    rnd = 1
    teams = list(seeds)
    for t in teams:
        reached[t] = rnd  # everyone reaches the entry round (R32)
    while len(teams) > 1:
        nxt, matches = [], []
        for i in range(0, len(teams), 2):
            w, hg, ag, pens = _knockout_match(teams[i], teams[i + 1], rng)
            reached[w] = rnd + 1
            nxt.append(w)
            matches.append((teams[i], teams[i + 1], hg, ag, w, pens))
        if record is not None:
            record.append((ROUND_NAMES[rnd], matches))
        teams = nxt
        rnd += 1
    return teams[0], reached


def _play_groups(rng: random.Random, fixed=None):
    """Play all 12 groups. Returns (winners, runners, thirds, standings, matches)
    where standings[g] is a list of (team, points, gd, gf) in finishing order and
    matches[g] is the list of (home, away, home_goals, away_goals) played."""
    winners: Dict[str, str] = {}
    runners: Dict[str, str] = {}
    thirds: Dict[str, Tuple[str, dict]] = {}
    standings: Dict[str, list] = {}
    matches: Dict[str, list] = {}
    for g, teams in GROUPS.items():
        res = _round_robin(teams, rng, group=g, fixed=fixed)
        matches[g] = [(h, a, hg, ag) for (h, a), (hg, ag) in res.items()]
        order = group_standings(teams, res)
        tbl = _table(teams, res)
        standings[g] = [(t, tbl[t][0], tbl[t][1], tbl[t][2]) for t in order]
        winners[g], runners[g], third = order[0], order[1], order[2]
        st = tbl[third]
        thirds[g] = (third, {"points": st[0], "gd": st[1], "gf": st[2]})
    return winners, runners, thirds, standings, matches


def _seed_r32(winners, runners, thirds) -> Tuple[List[str], List[str]]:
    """Determine the 32 R32 entrants in bracket order, plus the 8 qualifying
    third-placed teams (best-first)."""
    best_thirds = rank_thirds(thirds)
    qualified_groups = [g for g in GROUPS if thirds[g][0] in best_thirds]
    qualified_groups.sort(  # best->worst so assign_thirds fills in rank order
        key=lambda g: (thirds[g][1]["points"], thirds[g][1]["gd"], thirds[g][1]["gf"], g),
        reverse=True,
    )
    third_by_group = {g: thirds[g][0] for g in qualified_groups}
    slot_thirds = iter(assign_thirds(qualified_groups, third_by_group))

    def resolve(slot: tuple) -> str:
        kind, key = slot
        if kind == "W":
            return winners[key]
        if kind == "R":
            return runners[key]
        return next(slot_thirds)  # "3" slots consumed in bracket order

    seeds = [resolve(s) for match in R32 for s in match]
    return seeds, [third_by_group[g] for g in qualified_groups]


def simulate_tournament(
    rng: random.Random,
    fixed: Dict[Tuple[str, str, str], Tuple[int, int]] = None,
) -> Tuple[str, Dict[str, int]]:
    """Play one full tournament. Returns (champion, {team: furthest round}).

    `fixed` supplies already-played group results (keyed (group, home, away));
    every other match is sampled. Knockout matches are always sampled.
    """
    reached: Dict[str, int] = {t: 0 for g in GROUPS.values() for t in g}
    winners, runners, thirds, _, _ = _play_groups(rng, fixed)
    seeds, _q = _seed_r32(winners, runners, thirds)
    for t in seeds:
        reached[t] = max(reached[t], 1)  # reached R32
    champion, ko_reached = _knockout_tree(seeds, rng)
    for t, r in ko_reached.items():
        reached[t] = max(reached[t], r)
    return champion, reached


def simulate_detailed(rng: random.Random, fixed=None) -> dict:
    """Play one full tournament and return the whole bracket for display:

        {standings, qualified_thirds, seeds, rounds, champion}

    `rounds` is a list of (round_name, [(home, away, hg, ag, winner, pens), ...]).
    """
    winners, runners, thirds, standings, group_matches = _play_groups(rng, fixed)
    seeds, qualified = _seed_r32(winners, runners, thirds)
    rounds: list = []
    champion, _reached = _knockout_tree(seeds, rng, record=rounds)
    return {
        "standings": standings,
        "group_matches": group_matches,
        "qualified_thirds": qualified,
        "seeds": seeds,
        "rounds": rounds,
        "champion": champion,
    }


# --------------------------------------------------------------------------- #
# Monte-Carlo forecast.
# --------------------------------------------------------------------------- #

def forecast(
    sims: int = 10000,
    seed: int = 2026,
    fixed: Dict[Tuple[str, str, str], Tuple[int, int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Run `sims` independent tournaments; return per-team probabilities (%).

    Each team gets: champion, reach_final, reach_SF, reach_QF, reach_R16.
    Deterministic in `seed` (each sim uses a derived sub-seed). `fixed` lets you
    forecast a tournament already in progress by pinning known group results.
    """
    all_teams = [t for g in GROUPS.values() for t in g]
    titles = {t: 0 for t in all_teams}
    # furthest-round counts: index by ROUND_NAMES position
    reach = {t: [0] * len(ROUND_NAMES) for t in all_teams}
    for s in range(sims):
        rng = random.Random(seed * 1_000_003 + s)
        champ, reached = simulate_tournament(rng, fixed=fixed)
        titles[champ] += 1
        for t, r in reached.items():
            # count every round up to the furthest reached
            for ri in range(1, r + 1):
                reach[t][ri] += 1

    def pct(n: int) -> float:
        return round(100.0 * n / sims, 2)

    out = {}
    for t in all_teams:
        out[t] = {
            "champion": pct(titles[t]),
            "reach_final": pct(reach[t][5]),
            "reach_SF": pct(reach[t][4]),
            "reach_QF": pct(reach[t][3]),
            "reach_R16": pct(reach[t][2]),
            "reach_R32": pct(reach[t][1]),
        }
    return out


def forecast_table(results: Dict[str, Dict[str, float]], top: int = 20) -> str:
    rows = sorted(results.items(), key=lambda kv: kv[1]["champion"], reverse=True)
    lines = [f"{'Team':<16}{'Champ':>8}{'Final':>8}{'SF':>8}{'QF':>8}{'R16':>8}"]
    lines.append("-" * 56)
    for team, p in rows[:top]:
        lines.append(
            f"{team:<16}{p['champion']:>7.1f}%{p['reach_final']:>7.1f}%"
            f"{p['reach_SF']:>7.1f}%{p['reach_QF']:>7.1f}%{p['reach_R16']:>7.1f}%"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Backtest: how well does the pre-tournament model predict matches that have
# actually been played? (Out-of-sample: ELO predates these results.)
# --------------------------------------------------------------------------- #

def match_probabilities(home: str, away: str, sims: int = 30000, seed: int = 1) -> Dict[str, float]:
    """Model P(home win), P(draw), P(away win) for one matchup, by sampling."""
    rng = random.Random(seed)
    w = d = 0
    for _ in range(sims):
        hg, ag = sample_match(home, away, rng)
        if hg > ag:
            w += 1
        elif hg == ag:
            d += 1
    return {"W": w / sims, "D": d / sims, "L": (sims - w - d) / sims}


def evaluate_predictions(results=RESULTS_TO_DATE, sims: int = 30000, seed: int = 2026):
    """Score the model's W/D/L predictions against real results.

    Returns (rows, summary). For each match we record the predicted W/D/L
    probabilities, the actual outcome, the probability the model put on it, and
    a Brier score. Summary aggregates hit-rate (model's most-likely outcome was
    right), mean probability on the actual outcome, and mean Brier vs the
    uniform 1/3 baseline (Brier 0.667) as a skill score.
    """
    rows = []
    hits = p_sum = brier_sum = 0.0
    dec_hits = dec_n = draws = 0
    for i, (g, home, away, hg, ag) in enumerate(results):
        probs = match_probabilities(home, away, sims=sims, seed=seed * 7919 + i)
        actual = "W" if hg > ag else ("D" if hg == ag else "L")
        fav = max(probs, key=probs.get)
        hit = fav == actual
        brier = sum((probs[c] - (1.0 if c == actual else 0.0)) ** 2 for c in "WDL")
        hits += hit; p_sum += probs[actual]; brier_sum += brier
        if actual == "D":
            draws += 1
        else:                         # decisive game: did we name the winner?
            dec_n += 1
            dec_hits += probs["W"] > probs["L"] if actual == "W" else probs["L"] > probs["W"]
        rows.append({"group": g, "home": home, "away": away, "score": f"{hg}-{ag}",
                     "actual": actual, "probs": probs, "fav": fav, "hit": hit,
                     "brier": brier})
    n = len(results)
    base = 2 / 3  # uniform (1/3,1/3,1/3) Brier vs a one-hot outcome
    return rows, {
        "n": n,
        "draws": draws,
        "hit_rate": hits / n,
        "decisive_n": dec_n,
        "decisive_hit_rate": dec_hits / dec_n if dec_n else 0.0,
        "mean_p_actual": p_sum / n,
        "mean_brier": brier_sum / n,
        "baseline_brier": base,
        "skill_vs_uniform": 1 - (brier_sum / n) / base,
    }


def evaluation_table(rows, summary) -> str:
    out = [f"{'Match':<34}{'Actual':>7}{'P(actual)':>11}{'Model pick':>13}{'hit':>5}"]
    out.append("-" * 70)
    name = {"W": "home win", "D": "draw", "L": "away win"}
    for r in rows:
        match = f"{r['home']} {r['score']} {r['away']}"
        out.append(f"{match:<34}{name[r['actual']]:>9}"
                   f"{r['probs'][r['actual']] * 100:>9.0f}% {name[r['fav']]:>11}"
                   f"{'  ✓' if r['hit'] else '  ✗':>5}")
    out.append("-" * 70)
    out.append(
        f"All {summary['n']} matches ({summary['draws']} draws): "
        f"most-likely-outcome hit rate {summary['hit_rate'] * 100:.0f}% "
        f"(coin-among-3 ≈ 33%)")
    out.append(
        f"Decisive games only ({summary['decisive_n']}): named the winner "
        f"{summary['decisive_hit_rate'] * 100:.0f}% of the time")
    out.append(
        f"Mean prob on the actual result: {summary['mean_p_actual'] * 100:.0f}%   "
        f"Brier {summary['mean_brier']:.3f} vs {summary['baseline_brier']:.3f} uniform   "
        f"(skill {summary['skill_vs_uniform'] * 100:+.0f}%)")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Bracket rendering: one played-out tournament, as text and as a self-contained
# SVG in OpenWorld's "atlas" card aesthetic.
# --------------------------------------------------------------------------- #

_KO_LABELS = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-finals",
              "SF": "Semi-finals", "final": "Final"}


def render_bracket_text(detail: dict) -> str:
    """A terminal-friendly rendering of one tournament's bracket."""
    lines = []
    lines.append("GROUP STAGE — every fixture (round-robin); winners (1) & "
                 "runners-up (2) advance, plus the 8 best thirds")
    for g, table in detail["standings"].items():
        lines.append("")
        lines.append(f"Group {g}")
        for home, away, hg, ag in detail.get("group_matches", {}).get(g, []):
            lines.append(f"    {home:<16} {hg}-{ag} {away}")
        order = "  ".join(f"{i+1}.{t}({pts}pts,{gd:+d})"
                          for i, (t, pts, gd, _gf) in enumerate(table))
        lines.append(f"  -> {order}")
    lines.append("")
    lines.append("Best thirds (8 qualify): " + ", ".join(detail["qualified_thirds"]))
    for name, matches in detail["rounds"]:
        lines.append("")
        lines.append(_KO_LABELS[name].upper())
        for home, away, hg, ag, winner, pens in matches:
            tag = " (pens)" if pens else ""
            star_h = "*" if winner == home else " "
            star_a = "*" if winner == away else " "
            lines.append(f"  {star_h}{home:<16} {hg}-{ag} {away:<16}{star_a}{tag}")
    lines.append("")
    lines.append(f"CHAMPION: {detail['champion']}")
    return "\n".join(lines)


def render_bracket_svg(detail: dict, eval_rows=None, summary=None) -> str:
    """A self-contained SVG (atlas palette): group-stage band on top, then the
    knockout bracket R32 -> champion left-to-right.

    If `eval_rows`/`summary` from evaluate_predictions() are passed, played
    fixtures show the ACTUAL score with a right/wrong (✓/✗) mark, and the header
    carries a model-vs-actual accuracy strip.
    """
    C = {"bg0": "#fcfbf8", "bg1": "#eef0ec", "text": "#16202e", "muted": "#5b6675",
         "accent": "#1d4ed8", "accent2": "#b45309", "teal": "#0f766e",
         "line": "#dde2ea", "node": "#ffffff", "win": "#1e3a8a",
         "right": "#0f766e", "wrong": "#b91c1c"}
    actual = {frozenset((r["home"], r["away"])): r for r in (eval_rows or [])}
    rounds = detail["rounds"]                      # [(name, [matches]), ...]
    champ = detail["champion"]
    standings = detail["standings"]
    group_matches = detail.get("group_matches", {})
    qualified = set(detail["qualified_thirds"])
    n0 = len(rounds[0][1])                          # 16 R32 matches

    # group-stage band geometry: 12 cards in a 4 x 3 grid
    gcols, card_w, card_h, gx, gy = 4, 300, 230, 14, 16
    gband_top = 122
    ko_top = gband_top + 3 * (card_h + gy) + 60      # knockout starts below groups

    # knockout geometry
    colw, gap, box_h, row_h = 212, 26, 44, 21
    band = n0 * (box_h + gap)
    width = max(40 + len(rounds) * colw + 230, 40 + gcols * (card_w + gx) + 26)
    height = ko_top + band + 60

    def cy(r, m):                                  # vertical center of match m in round r
        nr = len(rounds[r][1])
        return ko_top + band * (m + 0.5) / nr

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def tr(s, n):                                  # truncate long names
        s = str(s)
        return s if len(s) <= n else s[:n - 1] + "…"

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="\'Iowan Old Style\',Georgia,serif">',
        f'<rect width="{width}" height="{height}" fill="{C["bg0"]}"/>',
        # header
        f'<rect x="0" y="0" width="{width}" height="84" fill="{C["bg1"]}"/>',
        f'<rect x="0" y="84" width="{width}" height="3" fill="{C["accent"]}"/>',
        f'<text x="40" y="44" font-size="27" font-weight="700" fill="{C["text"]}">'
        f'World Cup 2026 — a modelled tournament</text>',
        f'<text x="40" y="68" font-size="13" fill="{C["muted"]}">'
        f'One Elo-driven simulation · group stage → knockout · '
        f'champion: <tspan font-weight="700" fill="{C["accent2"]}">{esc(champ)}</tspan></text>',
        f'<text x="40" y="{gband_top - 6}" font-size="12" font-weight="700" '
        f'letter-spacing="1.4" fill="{C["muted"]}">GROUP STAGE — '
        f'<tspan fill="{C["accent"]}">1–2 advance</tspan> · '
        f'<tspan fill="{C["teal"]}">best-third qualifies</tspan> · '
        f'<tspan fill="{C["right"]}">✓</tspan>/<tspan fill="{C["wrong"]}">✗</tspan> '
        f'= pre-match pick vs actual · grey = not yet played</text>',
    ]

    # header accuracy strip (model vs actual, matchday 1) — the "side graph"
    if summary and eval_rows:
        px = width - 478
        out.append(f'<text x="{px}" y="30" font-size="12" font-weight="700" '
                   f'letter-spacing="1.2" fill="{C["text"]}">MODEL vs ACTUAL · matchday 1</text>')
        out.append(f'<text x="{px}" y="49" font-size="11.5" fill="{C["muted"]}">'
                   f'overall <tspan font-weight="700" fill="{C["text"]}">'
                   f'{summary["hit_rate"]*100:.0f}%</tspan> · decisive '
                   f'<tspan font-weight="700" fill="{C["text"]}">'
                   f'{summary["decisive_hit_rate"]*100:.0f}%</tspan> · Brier '
                   f'{summary["mean_brier"]:.2f}</text>')
        for i, r in enumerate(eval_rows):
            tx = px + i * 29
            col = C["right"] if r["hit"] else C["wrong"]
            out.append(f'<text x="{tx}" y="71" font-size="13" font-weight="700" '
                       f'fill="{col}">{"✓" if r["hit"] else "✗"}</text>')

    # group cards
    for idx, (g, table) in enumerate(standings.items()):
        col, row = idx % gcols, idx // gcols
        gxp = 40 + col * (card_w + gx)
        gyp = gband_top + row * (card_h + gy)
        out.append(f'<rect x="{gxp}" y="{gyp}" width="{card_w}" height="{card_h}" rx="10" '
                   f'fill="{C["node"]}" stroke="{C["line"]}" stroke-width="1"/>')
        out.append(f'<rect x="{gxp}" y="{gyp}" width="{card_w}" height="4" rx="2" fill="{C["accent2"]}"/>')
        out.append(f'<text x="{gxp + 14}" y="{gyp + 27}" font-size="14.5" font-weight="700" '
                   f'fill="{C["text"]}">Group {g}</text>')
        for pos, (t, pts, gd, _gf) in enumerate(table):
            ry = gyp + 46 + pos * 18
            third_ok = pos == 2 and t in qualified
            if pos < 2:
                col_t, wt = C["win"], "700"
            elif third_ok:
                col_t, wt = C["teal"], "700"
            else:
                col_t, wt = C["muted"], "400"
            marker = "✓" if (pos < 2 or third_ok) else " "
            out.append(f'<text x="{gxp + 14}" y="{ry}" font-size="11.5" font-weight="{wt}" '
                       f'fill="{col_t}">{pos + 1}. {esc(tr(t, 16))}</text>')
            out.append(f'<text x="{gxp + card_w - 12}" y="{ry}" font-size="11.5" '
                       f'font-weight="{wt}" fill="{col_t}" text-anchor="end">'
                       f'{pts}pt {gd:+d} {marker}</text>')
        out.append(f'<line x1="{gxp + 12}" y1="{gyp + 124}" x2="{gxp + card_w - 12}" '
                   f'y2="{gyp + 124}" stroke="{C["line"]}" stroke-width="0.8"/>')
        for j, (h, a, shg, sag) in enumerate(group_matches.get(g, [])):
            my = gyp + 140 + j * 15
            r = actual.get(frozenset((h, a)))
            if r:                                    # played: show ACTUAL score + right/wrong
                mark, mcol = ("✓", C["right"]) if r["hit"] else ("✗", C["wrong"])
                out.append(f'<text x="{gxp + 14}" y="{my}" font-size="10" fill="{C["text"]}">'
                           f'{esc(tr(r["home"], 10))} <tspan font-weight="700">{r["score"]}</tspan> '
                           f'{esc(tr(r["away"], 10))}</text>')
                out.append(f'<text x="{gxp + card_w - 12}" y="{my}" font-size="11" '
                           f'font-weight="700" fill="{mcol}" text-anchor="end">{mark}</text>')
            else:                                    # not yet played
                out.append(f'<text x="{gxp + 14}" y="{my}" font-size="10" fill="{C["muted"]}" '
                           f'opacity="0.7">{esc(tr(h, 11))} v {esc(tr(a, 11))}</text>')

    # knockout section heading + round headers
    out.append(f'<text x="40" y="{ko_top - 24}" font-size="12" font-weight="700" '
               f'letter-spacing="1.4" fill="{C["muted"]}">KNOCKOUT</text>')
    for r, (name, _m) in enumerate(rounds):
        x = 40 + r * colw
        out.append(f'<text x="{x + 6}" y="{ko_top - 8}" font-size="11.5" font-weight="700" '
                   f'letter-spacing="1.2" fill="{C["muted"]}">{_KO_LABELS[name].upper()}</text>')
    out.append(f'<text x="{40 + len(rounds) * colw + 6}" y="{ko_top - 8}" font-size="11.5" '
               f'font-weight="700" letter-spacing="1.2" fill="{C["accent2"]}">CHAMPION</text>')

    # connector elbows from round r to r+1
    for r in range(len(rounds) - 1):
        x_out = 40 + r * colw + (colw - gap)       # right edge of this round's boxes
        x_in = 40 + (r + 1) * colw                 # left edge of next round's boxes
        xm = (x_out + x_in) / 2
        for m in range(0, len(rounds[r][1]), 2):
            y1, y2 = cy(r, m), cy(r, m + 1)
            yt = cy(r + 1, m // 2)
            out.append(
                f'<path d="M{x_out:.0f} {y1:.0f} H{xm:.0f} V{yt:.0f} H{x_in:.0f} '
                f'M{x_out:.0f} {y2:.0f} H{xm:.0f} V{yt:.0f}" fill="none" '
                f'stroke="{C["line"]}" stroke-width="1.4"/>')

    # match boxes
    def match_box(x, ycenter, home, away, hg, ag, winner, pens):
        y = ycenter - box_h / 2
        w = colw - gap
        s = [f'<rect x="{x:.0f}" y="{y:.0f}" width="{w}" height="{box_h}" rx="7" '
             f'fill="{C["node"]}" stroke="{C["line"]}" stroke-width="1"/>',
             f'<line x1="{x:.0f}" y1="{y + row_h:.0f}" x2="{x + w:.0f}" '
             f'y2="{y + row_h:.0f}" stroke="{C["line"]}" stroke-width="0.7"/>']
        for k, (team, goals) in enumerate([(home, hg), (away, ag)]):
            ty = y + row_h * k + 15
            won = team == winner
            col = C["win"] if won else C["muted"]
            wt = "700" if won else "400"
            if won:  # winner accent bar
                s.append(f'<rect x="{x:.0f}" y="{y + row_h * k + 2:.0f}" width="3.5" '
                         f'height="{row_h - 4}" rx="1.5" fill="{C["accent"]}"/>')
            label = esc(team) + (" (p)" if pens and won else "")
            s.append(f'<text x="{x + 11:.0f}" y="{ty:.0f}" font-size="12.5" '
                     f'font-weight="{wt}" fill="{col}">{label}</text>')
            s.append(f'<text x="{x + w - 12:.0f}" y="{ty:.0f}" font-size="12.5" '
                     f'font-weight="{wt}" fill="{col}" text-anchor="end">{goals}</text>')
        return "".join(s)

    for r, (_name, matches) in enumerate(rounds):
        x = 40 + r * colw
        for m, mm in enumerate(matches):
            out.append(match_box(x, cy(r, m), *mm))

    # champion node
    cx = 40 + len(rounds) * colw
    cyc = cy(len(rounds) - 1, 0)
    w = colw - gap
    out.append(f'<rect x="{cx:.0f}" y="{cyc - 26:.0f}" width="{w + 30}" height="52" rx="9" '
               f'fill="{C["accent2"]}"/>')
    out.append(f'<text x="{cx + (w + 30) / 2:.0f}" y="{cyc - 4:.0f}" font-size="11" '
               f'font-weight="700" letter-spacing="1.5" fill="#fff" '
               f'text-anchor="middle" opacity="0.85">★ WINNERS ★</text>')
    out.append(f'<text x="{cx + (w + 30) / 2:.0f}" y="{cyc + 16:.0f}" font-size="16" '
               f'font-weight="700" fill="#fff" text-anchor="middle">{esc(champ)}</text>')
    out.append('</svg>')
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# OpenWorld world: the rules engine as a verified FunctionTransition, so the
# tournament serialises to a spec and serves interactively.
# --------------------------------------------------------------------------- #

def _transition(state: dict, action: dict) -> dict:
    """Dynamics for the served world.

    actions:
      play_match     params {group, home, away, home_goals, away_goals}
                     -> record a real/sampled group result.
      simulate_rest  params {seed} -> play out the entire tournament from a
                     seed and write the champion + furthest-round map.
    """
    s = dict(state)
    name = action.get("name")
    params = action.get("params") or {}
    if name == "play_match":
        results = dict(s.get("results", {}))
        key = f"{params['group']}|{params['home']}|{params['away']}"
        results[key] = [int(params["home_goals"]), int(params["away_goals"])]
        s["results"] = results
        return s
    if name == "simulate_rest":
        rng = random.Random(int(params.get("seed", 2026)))
        # honor any group results recorded via play_match; sample the rest
        fixed = {}
        for key, (hg, ag) in (s.get("results") or {}).items():
            grp, home, away = key.split("|")
            fixed[(grp, home, away)] = (int(hg), int(ag))
        champ, reached = simulate_tournament(rng, fixed=fixed)
        s["champion"] = champ
        s["reached"] = {t: ROUND_NAMES[r] for t, r in reached.items()}
        s["phase"] = "done"
        return s
    return s


def build_world():
    """Construct the OpenWorld World (lazy import keeps the module importable
    without the package on the path during pure-forecast use)."""
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from openworld import FunctionTransition, World

    initial = {
        "groups": GROUPS,
        "elo": ELO,
        "hosts": sorted(HOSTS),
        "results": {},
        "phase": "group",
        "champion": None,
        "reached": {},
    }
    return World(
        name="worldcup2026",
        description=(
            "FIFA World Cup 2026 (Canada/Mexico/USA): 48 teams, 12 groups, "
            "top-2 + 8 best thirds into a Round of 32, then a single-elimination "
            "bracket to the Final. Outcomes from an Elo->Poisson model."
        ),
        initial_state=initial,
        actions=["play_match", "simulate_rest"],
        rules=[
            "Group: win=3 pts, draw=1; rank by points, then goal difference, "
            "then goals for.",
            "Top two of each group plus the eight best third-placed teams "
            "advance to the Round of 32.",
            "Knockout matches are single-elimination; ties go to penalties.",
            "Host nations (USA, Mexico, Canada) get a small Elo advantage.",
        ],
        transition=FunctionTransition(_transition),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="World Cup 2026 forecaster")
    ap.add_argument("--sims", type=int, default=10000, help="Monte-Carlo runs")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--serve", action="store_true",
                    help="write worldcup2026.spec.json for `openworld serve`")
    ap.add_argument("--bracket", action="store_true",
                    help="play ONE tournament; print it and write worldcup2026_bracket.svg")
    ap.add_argument("--evaluate", action="store_true",
                    help="backtest the pre-tournament model vs real matchday-1 results")
    args = ap.parse_args()

    if args.evaluate:
        rows, summary = evaluate_predictions()
        print("Backtest — pre-tournament model vs real results (out-of-sample):\n")
        print(evaluation_table(rows, summary))
        return

    if args.bracket:
        detail = simulate_detailed(random.Random(args.seed))
        print(render_bracket_text(detail))
        rows, summary = evaluate_predictions()  # overlay actual results + right/wrong
        svg = render_bracket_svg(detail, eval_rows=rows, summary=summary)
        path = "worldcup2026_bracket.svg"
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        print(f"\nWrote {path} (open in a browser to view the visual bracket).")
        return

    if args.serve:
        world = build_world()  # also puts the repo root on sys.path
        from openworld import spec_to_json, to_spec
        spec = to_spec(world)
        path = "worldcup2026.spec.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(spec_to_json(spec))
        print(f"Wrote {path}. Serve with:\n  openworld serve {path} --allow-code --open")
        return

    print(f"Simulating the 2026 World Cup {args.sims}x (seed={args.seed})...\n")
    results = forecast(sims=args.sims, seed=args.seed)
    print(forecast_table(results))
    champ_sum = sum(p["champion"] for p in results.values())
    print(f"\n(Champion probabilities sum to {champ_sum:.1f}%.)")


if __name__ == "__main__":
    main()
