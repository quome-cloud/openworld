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
