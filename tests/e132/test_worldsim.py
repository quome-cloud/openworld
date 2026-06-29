from experiments.e132.worldsim import WorldSim


def test_predict_is_pure_lookup():
    w = WorldSim()
    assert w.predict(("s0",), [1]) is None and w.known(("s0",), [1]) is False
    w.learn(("s0",), [1], ("s1",), 0)
    assert w.predict(("s0",), [1]) == (("s1",), 0)
    assert w.known(("s0",), [1]) is True
    assert ("s1",) in w.seen


def test_predict_has_no_side_effects():
    w = WorldSim()
    w.learn(("s0",), [6, 3, 4], ("s1",), 1)
    before = dict(w.trans)
    for _ in range(5):
        assert w.predict(("s0",), [6, 3, 4]) == (("s1",), 1)   # repeatable, pure
    assert w.trans == before                                    # predict never mutates
