import numpy as np
from experiments.e130 import perception as P


def _frame():
    f = np.zeros((8, 8), dtype=int)      # bg=0
    f[1, 1] = 3                          # a rare singleton (click target)
    f[5, 5] = 3
    f[2:6, 6] = 7                        # a larger bar (not a click target)
    return f


def test_extrospect_returns_objects_and_key():
    s = P.extrospect(_frame(), avail=[6])
    assert len(s.objects) >= 2
    assert isinstance(s.key, tuple)
    assert s.vec.shape[0] == 64


def test_click_targets_are_small_components():
    s = P.extrospect(_frame(), avail=[6])
    sizes = {(t["y"], t["x"]) for t in s.click_targets}
    assert (1, 1) in sizes and (5, 5) in sizes      # the singletons
    assert all(t["size"] <= 16 for t in s.click_targets)


def test_extrospect_is_deterministic():
    a = P.extrospect(_frame(), avail=[6]); b = P.extrospect(_frame(), avail=[6])
    assert a.key == b.key and np.allclose(a.vec, b.vec)
