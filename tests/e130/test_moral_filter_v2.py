# tests/e130/test_moral_filter_v2.py
import numpy as np
from experiments.e130 import moral_filter as mf, perception as P
from experiments.e130.world_model import WorldModel
from experiments.e130.weights import ExpertWeights


def _stereo():
    f = np.zeros((8, 8), dtype=int); f[1, 1] = 3; f[6, 6] = 7; f[6, 7] = 7
    return P.extrospect(f, avail=[1, 2, 3, 4, 6])


def test_realize_reach_uses_dir_map_navigation():
    s = _stereo()                                   # avatar color 7 at (6,6) (the size-2 obj centroid is (6,6))
    dir_map = {1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}
    wp = mf.Waypoint("reach", 6, 1, "x")            # reach (6,1): move left 5 -> 5 action-4 steps
    plan = mf.realize(wp, s, dir_map=dir_map, avatar=7)
    assert plan and all(a[0] == 4 for a in plan)    # all 'left', not a click


def test_weighted_select_prefers_high_weight_expert():
    s = _stereo(); rng = np.random.default_rng(0)
    w = ExpertWeights(["reach_rare_color", "click_smallest"], eta=2.0)
    for _ in range(5): w.reward("click_smallest", 1.0); w.reward("reach_rare_color", 0.0)
    wp, plan, score = mf.select(s, [], WorldModel(), mf.DEFAULT_EXPERTS, rng, weights=w)
    assert wp.source == "click_smallest"            # the upweighted expert wins


def test_v1_select_still_works_without_weights():
    s = _stereo(); rng = np.random.default_rng(0)
    wp, plan, score = mf.select(s, [], WorldModel(), mf.DEFAULT_EXPERTS, rng)
    assert isinstance(wp, mf.Waypoint)
