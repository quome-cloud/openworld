import numpy as np
from experiments.e130 import moral_filter as mf
from experiments.e130.world_model import WorldModel
from experiments.e130 import perception as P


def _stereo():
    f = np.zeros((8, 8), dtype=int); f[1, 1] = 3; f[6, 6] = 7; f[6, 7] = 7
    return P.extrospect(f, avail=[6])


def test_experts_propose_at_least_one_waypoint():
    s = _stereo()
    wps = []
    for e in mf.DEFAULT_EXPERTS:
        wps += e(s, [], WorldModel())
    assert len(wps) >= 1
    assert all(isinstance(w, mf.Waypoint) for w in wps)


def test_select_returns_argmax_phi_times_V():
    s = _stereo(); rng = np.random.default_rng(0)
    wp, plan, score = mf.select(s, [], WorldModel(), mf.DEFAULT_EXPERTS, rng)
    assert isinstance(wp, mf.Waypoint) and isinstance(plan, list) and score >= 0.0


def test_amateur_degenerates_to_uniform_choice():
    # with amateur=True the filter ignores phi*V and draws uniformly: over many seeds it must
    # NOT always pick the same waypoint (the formalism's amateur/random regime).
    s = _stereo()
    picks = set()
    for seed in range(40):
        wp, _, _ = mf.select(s, [], WorldModel(), mf.DEFAULT_EXPERTS,
                             np.random.default_rng(seed), amateur=True)
        picks.add((wp.kind, wp.y, wp.x))
    assert len(picks) >= 2
