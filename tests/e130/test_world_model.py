from experiments.e130.world_model import WorldModel


def test_unknown_transition_flags_known_false():
    wm = WorldModel()
    nxt, known = wm.predict(("s0",), 1)
    assert nxt is None and known is False


def test_update_then_predict_is_known():
    wm = WorldModel()
    wm.update(("s0",), 1, ("s1",))
    nxt, known = wm.predict(("s0",), 1)
    assert known is True and nxt == ("s1",)


def test_simulate_stops_at_first_unseen():
    wm = WorldModel()
    wm.update(("s0",), 1, ("s1",))
    wm.update(("s1",), 1, ("s2",))           # s2 -> 1 is unseen
    preds, known = wm.simulate(("s0",), [1, 1, 1])
    assert preds == [("s1",), ("s2",)] and known is False


def test_contradiction_increments_conflicts():
    wm = WorldModel()
    wm.update(("s0",), 1, ("s1",))
    wm.update(("s0",), 1, ("sX",))           # regime change at the same (state, action)
    assert wm.conflicts == 1
    assert wm.predict(("s0",), 1)[0] == ("sX",)   # last-write wins


def test_bank_and_lookup_subroutine():
    wm = WorldModel()
    wm.bank_subroutine("reach_goal", [1, 2, 6])
    assert wm.lookup("reach_goal") == [1, 2, 6]
    assert wm.lookup("missing") is None
