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


def test_group_match_skill_beats_uniform():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    elo = eng.ratings_asof(wh._cup_freeze_date(2014))
    rows, summary = wh.score_group_matches(wh.load_cup(2014), elo, sims=4000, seed=3)
    assert summary["n"] == 48
    assert 0.0 <= summary["hit_rate"] <= 1.0
    assert summary["skill_vs_uniform"] > 0.0  # better than a 1/3 coin


def test_knockout_advancement_metric():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    elo = eng.ratings_asof(wh._cup_freeze_date(2014))
    summary = wh.score_knockout_advancement(wh.load_cup(2014), elo, sims=4000, seed=3)
    assert summary["n"] == 16
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert summary["brier"] >= 0.0


def test_tournament_calibration_fields():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    f = wh.forecast_cup(2014, eng, sims=800, seed=5)
    cal = wh.tournament_calibration(wh.load_cup(2014), f)
    assert cal["champion"] == "Germany"
    assert 1 <= cal["champion_rank"] <= 32
    assert cal["champion_prob"] >= 0.0
    assert cal["champion_logloss"] >= 0.0


def test_chalk_baseline_picks_higher_elo():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    elo = eng.ratings_asof(wh._cup_freeze_date(2014))
    base = wh.chalk_baseline(wh.load_cup(2014), elo)
    assert 0.0 <= base["group_hit_rate"] <= 1.0
    assert base["champion"] in [t for g in wh.load_cup(2014).groups.values() for t in g]


def test_reach_qf_calibration_buckets():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    cfs = [(wh.load_cup(y), wh.forecast_cup(y, eng, sims=200, seed=1))
           for y in (2010, 2014, 2018, 2022)]
    buckets = wh.reach_round_calibration(cfs, key="reach_QF")
    assert len(buckets) == 5
    # exactly 8 teams reach the QF per cup -> 32 positives total across 4 cups
    total_pos = sum(b["observed"] * b["n"] for b in buckets if b["n"])
    assert abs(total_pos - 32) < 1e-6
