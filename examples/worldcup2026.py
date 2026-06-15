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


def _knockout_winner(home: str, away: str, rng: random.Random) -> str:
    """Single-elimination result; ties go to an Elo-leaning penalty flip."""
    hg, ag = sample_match(home, away, rng)
    if hg > ag:
        return home
    if ag > hg:
        return away
    diff = _eff_elo(home) - _eff_elo(away)
    p_home = 1.0 / (1.0 + 10 ** (-diff / 400.0))      # shootout, slight Elo lean
    return home if rng.random() < p_home else away


# --------------------------------------------------------------------------- #
# Rules engine (pure, deterministic) — shared by the Monte-Carlo driver and the
# served World.
# --------------------------------------------------------------------------- #

def group_standings(teams: List[str], results: Dict[Tuple[str, str], Tuple[int, int]]) -> List[str]:
    """Order a group's teams by points -> GD -> GF -> goals-against (desc).

    `results` maps (home, away) -> (home_goals, away_goals) for that group.
    A stable, deterministic ordering (ties broken by name as a final fallback).
    """
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
    return sorted(
        teams,
        key=lambda t: (pts[t], gf[t] - ga[t], gf[t], -ga[t], t),
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


def _knockout_tree(seeds: List[str], rng: random.Random) -> Tuple[str, Dict[str, int]]:
    """Play a single-elimination bracket over `seeds` (len a power of two).

    Returns (champion, {team: furthest_round_index}) where round indices follow
    ROUND_NAMES (R32=1 ... champion=6). Pairs adjacent entries each round.
    """
    reached: Dict[str, int] = {}
    rnd = 1
    teams = list(seeds)
    for t in teams:
        reached[t] = rnd  # everyone reaches the entry round (R32)
    while len(teams) > 1:
        nxt = []
        for i in range(0, len(teams), 2):
            w = _knockout_winner(teams[i], teams[i + 1], rng)
            reached[w] = rnd + 1
            nxt.append(w)
        teams = nxt
        rnd += 1
    return teams[0], reached


def simulate_tournament(
    rng: random.Random,
    fixed: Dict[Tuple[str, str, str], Tuple[int, int]] = None,
) -> Tuple[str, Dict[str, int]]:
    """Play one full tournament. Returns (champion, {team: furthest round}).

    `fixed` supplies already-played group results (keyed (group, home, away));
    every other match is sampled. Knockout matches are always sampled.
    """
    reached: Dict[str, int] = {t: 0 for g in GROUPS.values() for t in g}
    winners: Dict[str, str] = {}
    runners: Dict[str, str] = {}
    thirds: Dict[str, Tuple[str, dict]] = {}

    for g, teams in GROUPS.items():
        res = _round_robin(teams, rng, group=g, fixed=fixed)
        order = group_standings(teams, res)
        winners[g], runners[g], third = order[0], order[1], order[2]
        # third-place stats for cross-group ranking
        pts = gf = ga = 0
        for (h, a), (hg, ag) in res.items():
            if third in (h, a):
                mine, theirs = (hg, ag) if h == third else (ag, hg)
                gf += mine; ga += theirs
                pts += 3 if mine > theirs else (1 if mine == theirs else 0)
        thirds[g] = (third, {"points": pts, "gd": gf - ga, "gf": gf})

    best_thirds = rank_thirds(thirds)
    qualified_groups = [g for g in GROUPS if thirds[g][0] in best_thirds]
    # rank the qualifying groups best->worst so assign_thirds fills in order
    qualified_groups.sort(
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
    for t in seeds:
        reached[t] = max(reached[t], 1)  # reached R32
    champion, ko_reached = _knockout_tree(seeds, rng)
    for t, r in ko_reached.items():
        reached[t] = max(reached[t], r)
    return champion, reached


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
    args = ap.parse_args()

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
