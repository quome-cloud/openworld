"""Tests for the automated tuning module."""

import random

import pytest

from openworld import (
    Action,
    Agent,
    Choice,
    FunctionTransition,
    IntRange,
    Simulation,
    Tuner,
    Uniform,
    World,
)


def make_noop_simulation(params):
    world = World(
        name="toy",
        description="inert world for scoring parameter landscapes",
        initial_state={"t": 0},
        actions=["wait"],
        transition=FunctionTransition(lambda s, a: dict(s)),
    )
    return Simulation(world, [Agent(name="a", policy=lambda s, acts: Action("wait"))])


def quadratic_tuner(seed):
    # Score depends only on params: a smooth bowl with its peak at x=0.7, k='b'.
    def score(trajectory, params):
        return -((params["x"] - 0.7) ** 2) + (1.0 if params["k"] == "b" else 0.0)

    return Tuner(
        build=make_noop_simulation,
        space={"x": Uniform(0.0, 1.0), "k": Choice(["a", "b", "c"])},
        score=score,
        steps=1,
        seed=seed,
    )


def test_search_finds_optimum_region():
    tuner = quadratic_tuner(seed=3)
    tuner.search(n_trials=200)
    best = tuner.study.best
    assert best.params["k"] == "b"
    assert abs(best.params["x"] - 0.7) < 0.05


def test_refine_improves_on_search():
    tuner = quadratic_tuner(seed=3)
    tuner.search(n_trials=30)  # coarse on purpose
    search_best_score = tuner.study.best.score
    tuner.refine(n_trials=60, scale=0.15)
    tuner.refine(n_trials=40, scale=0.03)
    best = tuner.study.best
    assert best.score > search_best_score
    assert abs(best.params["x"] - 0.7) < 0.01
    assert any(t.stage == "refine" for t in tuner.study.trials)


def test_tuner_is_deterministic_under_seed():
    a = quadratic_tuner(seed=11)
    b = quadratic_tuner(seed=11)
    a.search(50)
    b.search(50)
    assert a.study.best.params == b.study.best.params
    assert a.study.best.score == b.study.best.score


def test_success_rate_and_solved():
    def success(trajectory, params):
        return params["x"] > 0.5

    tuner = Tuner(
        build=make_noop_simulation,
        space={"x": Uniform(0.0, 1.0), "k": Choice(["a"])},
        score=lambda traj, params: params["x"],
        success=success,
        steps=1,
        seed=1,
    )
    tuner.search(100)
    rate = tuner.study.success_rate()
    assert 0.3 < rate < 0.7
    assert tuner.study.best.solved  # best x is high, so it satisfies success
    assert "refine" not in {t.stage for t in tuner.study.trials}


def test_param_bounds_respected():
    rng = random.Random(0)
    span = IntRange(2, 5)
    uni = Uniform(-1.0, 1.0)
    choice = Choice(["p", "q"])
    for _ in range(200):
        assert 2 <= span.sample(rng) <= 5
        assert 2 <= span.perturb(5, rng, 0.5) <= 5
        assert -1.0 <= uni.perturb(-1.0, rng, 0.9) <= 1.0
        assert choice.perturb("p", rng, 0.3) in ("p", "q")


def test_study_table_and_empty_guard():
    tuner = quadratic_tuner(seed=2)
    with pytest.raises(ValueError):
        _ = tuner.study.best
    tuner.search(5)
    table = tuner.study.table(k=3)
    assert "search" in table and "score" in table
