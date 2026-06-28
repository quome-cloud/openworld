# tests/test_e125_world.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import openworld as O
from e125 import world

PRED = ("def predict(state, action):\n"
        "    ns = {'bg': state['bg'], 'objects': [dict(o) for o in state['objects']]}\n"
        "    if action == [4]:\n"
        "        for o in ns['objects']: o['x'] = min(9, o['x'] + 1)\n"
        "    return ns, bool(ns['objects'] and ns['objects'][0]['x'] == 5)")
GOAL = ("def reward(state, action, next_state):\n"
        "    o = next_state['objects'][0]\n    return float(-(5 - o['x']))")
S0 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 1}]}

def test_build_world_returns_openworld_world():
    w = world.build_world(PRED, GOAL, S0, [[1],[2],[3],[4]], "synthA")
    assert isinstance(w, O.World)
    assert w.perceptors and w.objectives

def test_world_transition_steps_object_state():
    w = world.build_world(PRED, GOAL, S0, [[4]], "synthA")
    ns = w.transition.step(dict(S0, level_up=False), {"name": "[4]"})
    assert ns["objects"][0]["x"] == 2

def test_to_spec_round_trips_and_has_map():
    w = world.build_world(PRED, GOAL, S0, [[1],[2],[3],[4]], "synthA")
    spec = O.to_spec(w)
    assert O.from_spec(spec, allow_code=True) is not None
    # M2: {} is not None but is empty — assert the graph actually has reachable nodes
    assert spec["preview"]["graph"]["nodes"], (
        f"Expected non-empty graph nodes, got: {spec['preview']['graph']}"
    )
    assert world.round_trip_ok(w)


# --- M1: perceptor produces must include both 'bg' and 'objects' (TDD: FAILS before fix) ---

def test_perceptor_produces_bg_and_objects():
    """PERCEIVE_SRC returns {'bg', 'objects'}; produces must declare both."""
    w = world.build_world(PRED, GOAL, S0, [[1],[2],[3],[4]], "synthA")
    produces = w.perceptors[0].produces
    assert "bg" in produces and "objects" in produces, (
        f"perceptor.produces should include 'bg' and 'objects', got: {produces}"
    )
