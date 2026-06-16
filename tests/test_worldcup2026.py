"""Tests for the World Cup 2026 forecaster (examples/worldcup2026.py)."""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import worldcup2026 as wc  # noqa: E402


# --------------------------------------------------------------------------- #
# Data integrity
# --------------------------------------------------------------------------- #

def test_48_distinct_teams_all_rated():
    teams = [t for g in wc.GROUPS.values() for t in g]
    assert len(teams) == 48
    assert len(set(teams)) == 48, "duplicate team across groups"
    for t in teams:
        assert t in wc.ELO, f"{t} has no Elo rating"


def test_twelve_groups_of_four():
    assert len(wc.GROUPS) == 12
    assert all(len(v) == 4 for v in wc.GROUPS.values())


def test_r32_has_16_matches_and_8_third_slots():
    assert len(wc.R32) == 16
    third_slots = [b for m in wc.R32 for b in m if b[0] == "3"]
    assert len(third_slots) == 8
    # 12 winners + 12 runners-up + 8 thirds = 32 entrants
    kinds = [b[0] for m in wc.R32 for b in m]
    assert kinds.count("W") == 12
    assert kinds.count("R") == 12
    assert kinds.count("3") == 8


# --------------------------------------------------------------------------- #
# Outcome model
# --------------------------------------------------------------------------- #

def test_elo_win_probability_monotonic():
    # The much stronger team should win a head-to-head far more often.
    rng = random.Random(0)
    strong, weak = "Spain", "Qatar"
    wins = sum(wc._knockout_winner(strong, weak, rng) == strong for _ in range(3000))
    assert wins / 3000 > 0.75


def test_even_match_is_roughly_balanced():
    # Algeria and Iran are both rated 1772 -> a coin-flip head-to-head.
    assert wc.ELO["Algeria"] == wc.ELO["Iran"]
    rng = random.Random(1)
    a = sum(wc._knockout_winner("Algeria", "Iran", rng) == "Algeria" for _ in range(3000))
    assert 0.4 < a / 3000 < 0.6


def test_poisson_nonnegative_and_mean():
    rng = random.Random(2)
    draws = [wc._poisson(1.5, rng) for _ in range(5000)]
    assert all(d >= 0 for d in draws)
    assert 1.3 < sum(draws) / len(draws) < 1.7


# --------------------------------------------------------------------------- #
# Rules engine
# --------------------------------------------------------------------------- #

def test_group_standings_points_then_gd():
    teams = ["A", "B", "C", "D"]
    # A beats everyone; B and C both 1 win; D loses all. A first, D last.
    res = {
        ("A", "B"): (2, 0), ("A", "C"): (1, 0), ("A", "D"): (3, 0),
        ("B", "C"): (0, 0), ("B", "D"): (2, 0), ("C", "D"): (2, 0),
    }
    order = wc.group_standings(teams, res)
    assert order[0] == "A"
    assert order[-1] == "D"


def test_rank_thirds_takes_best_eight():
    thirds = {
        g: (f"team{g}", {"points": p, "gd": 0, "gf": 0})
        for g, p in zip("ABCDEFGHIJKL", [9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0])
    }
    best = wc.rank_thirds(thirds)
    assert len(best) == 8
    assert "teamA" in best and "teamB" in best
    # the four lowest-pointed thirds are excluded
    assert "teamL" not in best and "teamK" not in best


# --------------------------------------------------------------------------- #
# Full simulation invariants
# --------------------------------------------------------------------------- #

def test_simulate_tournament_structure():
    rng = random.Random(42)
    champ, reached = wc.simulate_tournament(rng)
    all_teams = {t for g in wc.GROUPS.values() for t in g}
    assert champ in all_teams
    # exactly 32 teams reach R32 (round index >= 1)
    assert sum(1 for r in reached.values() if r >= 1) == 32
    # round sizes: R16=16, QF=8, SF=4, final=2, champion=1
    assert sum(1 for r in reached.values() if r >= 2) == 16
    assert sum(1 for r in reached.values() if r >= 3) == 8
    assert sum(1 for r in reached.values() if r >= 4) == 4
    assert sum(1 for r in reached.values() if r >= 5) == 2
    assert sum(1 for r in reached.values() if r >= 6) == 1


def test_no_team_advances_twice():
    # A subtle bug would let one third fill two slots; check 32 DISTINCT entrants.
    rng = random.Random(7)
    # re-run the seeding portion by simulating and counting R32 entrants
    _, reached = wc.simulate_tournament(rng)
    entrants = [t for t, r in reached.items() if r >= 1]
    assert len(entrants) == len(set(entrants)) == 32


def test_determinism_same_seed():
    a = wc.simulate_tournament(random.Random(123))
    b = wc.simulate_tournament(random.Random(123))
    assert a == b


# --------------------------------------------------------------------------- #
# Detailed bracket + rendering
# --------------------------------------------------------------------------- #

def test_simulate_detailed_structure():
    d = wc.simulate_detailed(random.Random(2026))
    # round sizes 16 -> 8 -> 4 -> 2 -> 1
    assert [len(matches) for _name, matches in d["rounds"]] == [16, 8, 4, 2, 1]
    assert [name for name, _m in d["rounds"]] == ["R32", "R16", "QF", "SF", "final"]
    assert len(d["standings"]) == 12 and all(len(v) == 4 for v in d["standings"].values())
    assert len(d["qualified_thirds"]) == 8
    # every group plays its full round-robin: 6 matches each, 72 total
    assert len(d["group_matches"]) == 12
    assert all(len(m) == 6 for m in d["group_matches"].values())
    assert sum(len(m) for m in d["group_matches"].values()) == 72
    # the pairings are exactly the 6 unordered pairs of each group's 4 teams
    for g, teams in wc.GROUPS.items():
        pairs = {frozenset((h, a)) for h, a, _hg, _ag in d["group_matches"][g]}
        assert pairs == {frozenset((teams[i], teams[j]))
                         for i in range(4) for j in range(i + 1, 4)}
    # the final's winner is the champion
    _h, _a, _hg, _ag, winner, _p = d["rounds"][-1][1][0]
    assert winner == d["champion"]


def test_detailed_matches_simulate_tournament():
    # same seed -> same RNG draws -> same champion as the aggregate simulator
    seed = 77
    assert wc.simulate_detailed(random.Random(seed))["champion"] == \
        wc.simulate_tournament(random.Random(seed))[0]


def test_svg_is_self_contained():
    d = wc.simulate_detailed(random.Random(1))
    svg = wc.render_bracket_svg(d)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert d["champion"] in svg
    # 31 knockout boxes (16+8+4+2+1) at rx="7"; 12 group cards at rx="10"
    assert svg.count('rx="7"') == 31
    assert svg.count('rx="10"') == 12
    assert "GROUP STAGE" in svg and "Group A" in svg and "KNOCKOUT" in svg
    # no fetched resources (xmlns URL is fine; no http(s) image/script refs)
    assert "http://" not in svg.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert "https://" not in svg


# --------------------------------------------------------------------------- #
# Backtest / accuracy
# --------------------------------------------------------------------------- #

def test_match_probabilities_normalize_and_favor_strong():
    p = wc.match_probabilities("Germany", "Curacao", sims=4000, seed=3)
    assert abs(p["W"] + p["D"] + p["L"] - 1.0) < 1e-9
    assert p["W"] > p["L"]            # huge favourite wins far more than loses
    assert p["W"] > 0.6


def test_evaluate_predictions_structure_and_determinism():
    rows, summary = wc.evaluate_predictions(sims=2000, seed=4)
    assert len(rows) == len(wc.RESULTS_TO_DATE) == 15
    assert summary["draws"] + summary["decisive_n"] == 15
    assert 0.0 <= summary["hit_rate"] <= 1.0
    assert 0.0 <= summary["decisive_hit_rate"] <= 1.0
    # deterministic given seed/sims
    assert wc.evaluate_predictions(sims=2000, seed=4)[1] == summary
    # the actual results really do contain that many draws (data sanity)
    draws = sum(1 for _g, _h, _a, hg, ag in wc.RESULTS_TO_DATE if hg == ag)
    assert draws == summary["draws"] == 7


def test_evaluation_table_renders():
    rows, summary = wc.evaluate_predictions(sims=1000, seed=1)
    txt = wc.evaluation_table(rows, summary)
    assert "hit rate" in txt and "Brier" in txt


def test_svg_overlays_actuals_and_right_wrong():
    d = wc.simulate_detailed(random.Random(2026))
    rows, summary = wc.evaluate_predictions(sims=1000, seed=2026)
    svg = wc.render_bracket_svg(d, eval_rows=rows, summary=summary)
    assert "MODEL vs ACTUAL" in svg
    assert "7-1" in svg          # Germany 7-1 Curacao, an actual result
    assert "✗" in svg            # at least one wrong pick is marked (matchday 1 had upsets)
    assert "https://" not in svg  # still self-contained
    # without eval data, no actual-results overlay
    assert "MODEL vs ACTUAL" not in wc.render_bracket_svg(d)


# --------------------------------------------------------------------------- #
# Forecast aggregation
# --------------------------------------------------------------------------- #

def test_forecast_probabilities_normalize():
    res = wc.forecast(sims=400, seed=5)
    assert len(res) == 48
    champ_sum = sum(p["champion"] for p in res.values())
    assert champ_sum == pytest.approx(100.0, abs=0.5)
    final_sum = sum(p["reach_final"] for p in res.values())
    assert final_sum == pytest.approx(200.0, abs=1.0)  # 2 finalists each sim
    for p in res.values():
        for v in p.values():
            assert 0.0 <= v <= 100.0


def test_forecast_deterministic():
    assert wc.forecast(sims=200, seed=9) == wc.forecast(sims=200, seed=9)


def test_fixed_results_are_honored():
    # Pin every Group A match so the standings are forced, then check the table
    # reflects it: if Mexico wins all three, it must reach R32 100% of the time.
    a = wc.GROUPS["A"]  # ["Mexico", "South Korea", "South Africa", "Czechia"]
    fixed = {}
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            # Mexico (a[0]) wins big; otherwise low-scoring draws
            fixed[("A", a[i], a[j])] = (5, 0) if a[i] == "Mexico" else (0, 0)
    res = wc.forecast(sims=300, seed=11, fixed=fixed)
    assert res["Mexico"]["reach_R32"] == 100.0
    # The other three tie at 2 pts / -5 GD; the name tiebreak puts Czechia 4th,
    # and a 4th-placed team can never advance.
    assert res["Czechia"]["reach_R32"] == 0.0


def test_host_advantage_helps_hosts():
    base = wc.HOST_ADVANTAGE
    try:
        rng = random.Random(3)
        wc.HOST_ADVANTAGE = 0.0
        usa_no = sum(
            wc._knockout_winner("United States", "Austria", random.Random(i)) == "United States"
            for i in range(2000)
        )
        wc.HOST_ADVANTAGE = 150.0
        usa_yes = sum(
            wc._knockout_winner("United States", "Austria", random.Random(i)) == "United States"
            for i in range(2000)
        )
        assert usa_yes > usa_no
    finally:
        wc.HOST_ADVANTAGE = base


# --------------------------------------------------------------------------- #
# OpenWorld integration
# --------------------------------------------------------------------------- #

def test_build_world_steps():
    from openworld import Action

    world = wc.build_world()
    assert world.name == "worldcup2026"
    # play_match records a result
    world.step(Action("play_match", params={
        "group": "C", "home": "Brazil", "away": "Scotland",
        "home_goals": 3, "away_goals": 0,
    }))
    assert "C|Brazil|Scotland" in world.state["results"]
    # simulate_rest names a champion
    world.step(Action("simulate_rest", params={"seed": 2026}))
    assert world.state["phase"] == "done"
    assert world.state["champion"] in {t for g in wc.GROUPS.values() for t in g}


def test_world_spec_roundtrips():
    from openworld import from_spec, to_spec

    world = wc.build_world()
    spec = to_spec(world)
    assert spec["name"] == "worldcup2026"
    # reconstruct with code allowed; should rebuild without error
    rebuilt = from_spec(spec, allow_code=True)
    assert rebuilt.name == "worldcup2026"
