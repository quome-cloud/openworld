from e119 import world
import openworld as O


def _chain():
    # 2-step solved path; the second step raises levels 0 -> 1.
    return [
        {"key": "aa", "action": (7,), "next_key": "bb", "levels": 0},
        {"key": "bb", "action": (6, 60, 32), "next_key": "cc", "levels": 1},
    ]


def test_action_name_encodes_directional_and_click():
    assert world.action_name((7,)) == "a7"
    assert world.action_name((6, 60, 32)) == "click_60_32"


def test_solver_world_rollout_follows_the_learned_table():
    w = world.solver_world("tn36", _chain())
    assert isinstance(w, O.World)
    s0 = w.initial_state
    s1 = w.transition.step(s0, O.Action("a7", agent="solver"))
    assert s1["key"] == "bb"
    s2 = w.transition.step(s1, O.Action("click_60_32", agent="solver"))
    assert s2["key"] == "cc" and s2["levels"] == 1


def test_solver_world_serializes_to_spec():
    w = world.solver_world("tn36", _chain())
    spec = O.to_spec(w)
    assert spec["name"] == "arc_tn36"
    assert "a7" in spec["actions"] and "click_60_32" in spec["actions"]
