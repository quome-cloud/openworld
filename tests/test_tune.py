"""Tests for the automated tuning module."""

import random

import pytest

from openworld import (
    Action,
    Agent,
    Choice,
    Dial,
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


def test_dials_are_first_class_in_space():
    # A Dial in the space is tuned as a Uniform over its [minimum, maximum].
    dial = Dial("morality", value=0.5, minimum=0.2, maximum=0.8)
    tuner = Tuner(
        build=make_noop_simulation,
        space={"morality": dial},
        score=lambda traj, params: params["morality"],
        steps=1,
        seed=4,
    )
    tuner.search(50)
    values = [t.params["morality"] for t in tuner.study.trials]
    assert all(0.2 <= v <= 0.8 for v in values)
    assert tuner.study.best.params["morality"] > 0.75  # maximizing finds the top


def test_parallel_search_matches_serial():
    serial = quadratic_tuner(seed=9)
    serial.search(40)
    parallel = quadratic_tuner(seed=9)
    parallel.workers = 4
    parallel.search(40)
    assert [t.params for t in serial.study.trials] == [t.params for t in parallel.study.trials]
    assert [t.score for t in serial.study.trials] == [t.score for t in parallel.study.trials]


def test_parallel_refine_runs_and_improves():
    tuner = quadratic_tuner(seed=9)
    tuner.workers = 4
    tuner.search(20)
    before = tuner.study.best.score
    tuner.refine(40, scale=0.1)
    assert tuner.study.best.score >= before
    assert sum(t.stage == "refine" for t in tuner.study.trials) == 40


def test_optuna_tpe_strategy():
    pytest.importorskip("optuna")
    tuner = quadratic_tuner(seed=5)
    tuner.search(40, strategy="tpe")
    best = tuner.study.best
    assert best.params["k"] == "b"
    assert abs(best.params["x"] - 0.7) < 0.2
    assert {t.stage for t in tuner.study.trials} == {"tpe"}


def test_unknown_strategy_raises():
    tuner = quadratic_tuner(seed=5)
    with pytest.raises(ValueError):
        tuner.search(5, strategy="genetic")


def test_study_to_csv(tmp_path):
    tuner = quadratic_tuner(seed=6)
    tuner.search(10)
    path = tuner.study.to_csv(tmp_path / "study.csv")
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 11  # header + 10 trials
    header = lines[0].split(",")
    assert "score" in header and "x" in header and "k" in header
    assert any(name.startswith("total_") for name in header)


def test_study_table_and_empty_guard():
    tuner = quadratic_tuner(seed=2)
    with pytest.raises(ValueError):
        _ = tuner.study.best
    tuner.search(5)
    table = tuner.study.table(k=3)
    assert "search" in table and "score" in table
