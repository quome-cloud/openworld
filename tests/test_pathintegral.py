"""Tests for the path-integral over learning trajectories."""

import math

from openworld import Skill, TrajectorySpace
from openworld.pathintegral import COUNTING, LOG, TROPICAL


def simple_space():
    # a, b primitives; c needs a,b; goal needs c (+a). cheap to compose, dear from scratch.
    skills = [
        Skill("a", compose_cost=1, scratch_cost=1),       # primitive
        Skill("b", compose_cost=1, scratch_cost=1),
        Skill("c", prereqs=("a", "b"), compose_cost=1, scratch_cost=20),
        Skill("goal", prereqs=("c",), compose_cost=1, scratch_cost=50),
    ]
    return TrajectorySpace(skills, initial=(), goal="goal")


def test_least_action_path_uses_composition():
    sp = simple_space()
    steps, cost = sp.least_action_path()
    assert steps[-1] == "goal"
    assert set(["a", "b", "c"]) <= set(steps)            # learned the prereqs
    # a(1)+b(1)+c(1)+goal(1) = 4, far below learning goal from scratch (50)
    assert abs(cost - 4) < 1e-9
    assert cost < sp.goal_cost_from_scratch()


def test_transfer_beats_scratch():
    sp = simple_space()
    _, cost = sp.least_action_path()
    assert cost < sp.goal_cost_from_scratch()


def test_counting_sums_orderings_without_enumerating():
    sp = simple_space()
    n = sp.count_trajectories()
    # a and b are independent prereqs of c -> at least the two orders a,b and b,a
    assert n >= 2


def test_partition_is_dominated_by_least_action_as_beta_grows():
    sp = simple_space()
    _, best = sp.least_action_path()
    # -1/beta * logZ -> best cost as beta -> infinity (free energy -> ground state)
    free_hi = -sp.partition(LOG, beta=20.0) / 20.0
    assert abs(free_hi - best) < 0.2


def test_marginals_normalized_and_prioritize_prereqs():
    sp = simple_space()
    marg = sp.node_marginals(beta=5.0)
    assert all(0.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in marg.values())
    # every optimal trajectory must learn c and the goal -> marginal ~1
    assert marg["c"] > 0.9 and marg["goal"] > 0.9


def test_irrelevant_skills_excluded():
    skills = [Skill("a"), Skill("distractor"), Skill("goal", prereqs=("a",))]
    sp = TrajectorySpace(skills, initial=(), goal="goal")
    assert "distractor" not in sp.universe           # never on the way to the goal
