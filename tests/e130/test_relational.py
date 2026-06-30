from experiments.e130.world_model import WorldModel


def test_lifted_rule_transfers_across_positions():
    wm = WorldModel()
    wm.learn_rule(5, 3, (2, 2), (2, 3))              # avatar 5, action 3 -> +x learned at (2,2)
    assert wm.predict_rel(5, 3) == (0, 1)            # applies anywhere (position-independent)
    assert wm.predict_rel(5, 1) is None              # unseen action


def test_rule_is_separate_from_absolute_table():
    wm = WorldModel()
    wm.learn_rule(5, 3, (2, 2), (2, 3))
    assert wm.predict(("s0",), 3) == (None, False)   # absolute table untouched
