import numpy as np
from experiments.e130 import navigation as nav
from experiments.e130 import perception as P


def test_plan_reach_steps_toward_target():
    # avatar (color 5) at (2,2); target at (2,5); dir_map: action 3 = move +x (right)
    dir_map = {3: (0, 1), 4: (0, -1), 1: (-1, 0), 2: (1, 0)}
    f = np.zeros((8, 8), dtype=int); f[2, 2] = 5
    s = P.extrospect(f, avail=[1, 2, 3, 4])
    plan = nav.plan_reach(s, 5, dir_map, (2, 5), max_steps=10)
    assert plan == [(3, None, None), (3, None, None), (3, None, None)]   # three rights


def test_plan_reach_empty_when_no_dir_map():
    f = np.zeros((8, 8), dtype=int); f[2, 2] = 5
    s = P.extrospect(f, avail=[1, 2, 3, 4])
    assert nav.plan_reach(s, 5, {}, (2, 5)) == []


def test_avatar_pos_finds_the_color():
    f = np.zeros((8, 8), dtype=int); f[3, 4] = 7
    s = P.extrospect(f, avail=[])
    assert nav.avatar_pos(s, 7) == (3, 4)
    assert nav.avatar_pos(s, 9) is None
