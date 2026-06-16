"""Tests for the World Cup model benchmark (examples/worldcup_benchmark.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import worldcup_benchmark as wb  # noqa: E402
import worldcup_history as wh  # noqa: E402


def test_rps_perfect_is_zero():
    assert wb.rps({"W": 1.0, "D": 0.0, "L": 0.0}, "W") == 0.0
    assert wb.rps({"W": 0.0, "D": 0.0, "L": 1.0}, "L") == 0.0


def test_rps_uniform_values():
    u = {"W": 1 / 3, "D": 1 / 3, "L": 1 / 3}
    assert abs(wb.rps(u, "W") - 5 / 18) < 1e-9   # decisive
    assert abs(wb.rps(u, "L") - 5 / 18) < 1e-9   # decisive
    assert abs(wb.rps(u, "D") - 1 / 9) < 1e-9    # draw


def test_rps_ordering_sensitivity():
    # Predicting the adjacent outcome (draw) beats predicting the far one (away win)
    # when home actually won.
    near = {"W": 0.0, "D": 1.0, "L": 0.0}
    far = {"W": 0.0, "D": 0.0, "L": 1.0}
    assert wb.rps(near, "W") < wb.rps(far, "W")


def test_score_matches_aggregates():
    preds = {("d", "A", "B"): {"W": 0.7, "D": 0.2, "L": 0.1}}
    actuals = {("d", "A", "B"): "W"}
    s = wb.score_matches(preds, actuals)
    assert s["n"] == 1
    assert 0.0 <= s["rps"] <= 1.0
    assert 0.0 <= s["brier"] <= 2.0
    assert s["hit_rate"] == 1.0


def test_cup_matches_are_64_in_date_order():
    m = wb.cup_matches(2014)
    assert len(m) == 64
    dates = [r["date"] for r in m]
    assert dates == sorted(dates)
    r = m[0]
    assert set(r) >= {"date", "home", "away", "hg", "ag", "neutral"}


def test_actual_outcomes_home_perspective():
    out = wb.actual_outcomes(2014)
    assert len(out) == 64
    # Brazil opened 2014 by beating Croatia 3-1 at home -> home win "W".
    key = next(k for k in out if k[1] == "Brazil" and k[2] == "Croatia")
    assert out[key] == "W"


def test_training_matches_are_leakage_free():
    cutoff = wh._cup_freeze_date(2014)
    tr = wb.training_matches(2014, years=4)
    assert tr, "expected pre-cup internationals"
    assert all(r["date"] < cutoff for r in tr)
    assert all(r["date"] >= "2010-06-12" for r in tr)  # 4-year window


def test_uniform_predictions():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    p = wb.predict_uniform(2014, eng)
    assert len(p) == 64
    v = next(iter(p.values()))
    assert v == {"W": 1 / 3, "D": 1 / 3, "L": 1 / 3}


def test_ours_frozen_predictions_normalised():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    p = wb.predict_ours_frozen(2014, eng, sims=2000)
    assert len(p) == 64
    for v in p.values():
        assert abs(sum(v.values()) - 1.0) < 1e-9


def test_walk_forward_beats_uniform_on_2014():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    preds = wb.predict_ours_walk_forward(2014, eng, sims=3000)
    actuals = wb.actual_outcomes(2014)
    s = wb.score_matches(preds, actuals)
    u = wb.score_matches(wb.predict_uniform(2014, eng), actuals)
    assert len(preds) == 64
    assert s["rps"] < u["rps"]   # real skill beats the floor


def test_davidson_probs_normalise_and_favour_stronger():
    p = wb._davidson_probs(elo_diff=200.0, nu=1.0)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert p["W"] > p["L"]                      # home stronger by 200 Elo
    eq = wb._davidson_probs(elo_diff=0.0, nu=1.0)
    assert abs(eq["W"] - eq["L"]) < 1e-9        # symmetric when even


def test_fit_davidson_nu_positive_and_predicts():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    nu = wb.fit_davidson_nu(2014, eng)
    assert nu > 0.0
    preds = wb.predict_elo_logistic(2014, eng)
    actuals = wb.actual_outcomes(2014)
    s = wb.score_matches(preds, actuals)
    u = wb.score_matches(wb.predict_uniform(2014, eng), actuals)
    assert len(preds) == 64
    assert s["rps"] < u["rps"]


def test_poisson_grid_probs_normalise():
    p = wb._poisson_wdl(1.6, 0.9, max_goals=10)
    assert abs(sum(p.values()) - 1.0) < 1e-6
    assert p["W"] > p["L"]            # home expects more goals


@pytest.mark.parametrize("year", [2018, 2022])
def test_fte_aligns_to_real_fixtures(year):
    preds = wb.predict_fte(year)
    actuals = wb.actual_outcomes(year)
    assert len(preds) == 64
    # every 538 match_key must be a real fixture (1:1 alignment after name mapping)
    assert all(k in actuals for k in preds), set(preds) - set(actuals)
    for v in preds.values():
        assert abs(sum(v.values()) - 1.0) < 1e-6


def test_fte_probs_oriented_home_perspective():
    # 2022 opener Qatar(home) vs Ecuador: 538 favoured Ecuador, so L > W from
    # Qatar's home perspective.
    preds = wb.predict_fte(2022)
    key = next(k for k in preds if k[1] == "Qatar" and k[2] == "Ecuador")
    assert preds[key]["L"] > preds[key]["W"]


def test_maher_fit_and_predict_beats_uniform():
    eng = wh.EloEngine.from_results(wh.RESULTS_CSV)
    model = wb.fit_maher(2014)
    assert "atk" in model and "dee" in model and "home" in model and "mu" in model
    preds = wb.predict_maher(2014, model)
    actuals = wb.actual_outcomes(2014)
    s = wb.score_matches(preds, actuals)
    u = wb.score_matches(wb.predict_uniform(2014, eng), actuals)
    assert len(preds) == 64
    assert s["rps"] < u["rps"]


def test_run_benchmark_structure_and_floor():
    res = wb.run_benchmark(sims=2000)
    assert set(res["per_model"]) >= {"uniform", "elo_logistic", "ours_frozen",
                                     "ours_walk_forward", "maher"}
    # every probabilistic model beats the uniform floor on pooled RPS
    u = res["per_model"]["uniform"]["pooled"]["rps"]
    for name in ("elo_logistic", "ours_frozen", "ours_walk_forward", "maher"):
        assert res["per_model"][name]["pooled"]["rps"] < u, name
    # 538 head-to-head present on 128 matches
    assert res["head_to_head_538"]["n"] == 128
    assert "five_thirty_eight" in res["head_to_head_538"]["per_model"]
