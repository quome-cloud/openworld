"""Tests for the historical World Cup backtest (examples/worldcup_history.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import worldcup_history as wh  # noqa: E402


def test_elo_engine_runs_and_rates_known_strong_teams():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    # Pre-2014 (frozen at the day before the 2014 opener): Brazil & Germany strong.
    ratings = eng.ratings_asof("2014-06-11")
    assert ratings["Brazil"] > 1850
    assert ratings["Germany"] > 1850
    assert ratings["Brazil"] > ratings["United States"]


def test_no_look_ahead():
    # Ratings frozen before a cup must not change if we only append matches that
    # happen on/after the freeze date.
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    before = eng.ratings_asof("2014-06-11")
    # Recompute using only matches strictly before the freeze date -> identical.
    eng2 = wh.EloEngine.from_results(wh.RESULTS_CSV, until="2014-06-11")
    after = eng2.ratings_asof("2014-06-11")
    for team in ["Brazil", "Germany", "Italy", "Costa Rica"]:
        assert before[team] == after[team]


def test_elo_update_is_zero_sum_per_match():
    # A single match shifts the two teams' ratings by equal and opposite amounts.
    eng = wh.EloEngine(base=1500.0)
    a0, b0 = eng.rating("A"), eng.rating("B")
    eng.update_match("A", "B", 2, 0, neutral=True, k=60.0)
    da = eng.rating("A") - a0
    db = eng.rating("B") - b0
    assert abs(da + db) < 1e-9
    assert da > 0  # winner gains


def test_validation_against_published_elo_is_strong():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    stats = wh.validate_against_published(eng, snapshot_year=2013)
    # Our reconstructed Elo should track eloratings.net well on shared teams.
    assert stats["n"] >= 20
    assert stats["spearman"] >= 0.7
    assert stats["pearson"] >= 0.7
    assert stats["rmse"] < 250.0


@pytest.mark.parametrize("year", [2010, 2014, 2018, 2022])
def test_encoded_groups_match_real_data(year):
    cup = wh.load_cup(year)
    # 8 groups of 4 distinct teams.
    assert len(cup.groups) == 8
    assert sorted(cup.groups) == list("ABCDEFGH")
    teams = [t for g in cup.groups.values() for t in g]
    assert len(teams) == 32 and len(set(teams)) == 32
    # Every encoded group's 6 round-robin pairings appear as real group matches.
    for letter, four in cup.groups.items():
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = four[i], four[j]
                assert cup.group_result(a, b) is not None, (year, letter, a, b)


@pytest.mark.parametrize("year,champion", [
    (2010, "Spain"), (2014, "Germany"), (2018, "France"), (2022, "Argentina")])
def test_actual_champion_recovered(year, champion):
    cup = wh.load_cup(year)
    assert cup.actual_champion() == champion


def test_forecast_cup_is_deterministic_and_normalised():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    f1 = wh.forecast_cup(2014, eng, sims=300, seed=7)
    f2 = wh.forecast_cup(2014, eng, sims=300, seed=7)
    assert f1 == f2  # deterministic in seed
    champ = sum(v["champion"] for v in f1.values())
    assert abs(champ - 100.0) < 1e-6  # title probs sum to 100%
    # The pre-2014 favourites should top the title odds.
    top = max(f1, key=lambda t: f1[t]["champion"])
    assert top in {"Brazil", "Germany", "Argentina", "Spain"}
