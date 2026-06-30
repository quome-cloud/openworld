# tests/e132/test_plan.py
from experiments.e132.worldsim import WorldSim
from experiments.e132.plan import plan_in_model


def _chain_sim():
    # a model where the ONLY way to raise levels is the length-4 path 1->1->1->1 (deep horizon needed)
    w = WorldSim(); states = ["s0", "s1", "s2", "s3", "s4"]
    for i in range(4):
        w.learn((states[i],), [1], (states[i + 1],), 1 if i == 3 else 0)
        w.learn((states[i],), [2], (states[i],), 0)   # action 2 = no-op self-loop
    return w


def test_deep_plan_finds_length4_win():
    w = _chain_sim()
    actions_of = lambda k: [[1], [2]]
    plan, val, leaf = plan_in_model(w, ("s0",), 0, actions_of, depth=8, beam=8)
    assert val[0] == 1                          # found the +1-level leaf...
    assert plan[:4] == [[1], [1], [1], [1]]     # ...via the full depth-4 chain (impossible at depth<4)


def test_shallow_depth_cannot_see_it():
    w = _chain_sim()
    actions_of = lambda k: [[1], [2]]
    plan, val, leaf = plan_in_model(w, ("s0",), 0, actions_of, depth=2, beam=8)
    assert val[0] == 0                          # depth-2 cannot reach the depth-4 win (the E131 ceiling)
