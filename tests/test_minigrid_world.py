"""The MiniGrid (DoorKey) verified world: dynamics + serialization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from minigrid_world import (MINIGRID_INITIAL, SOLUTION, build_minigrid_world, solved)  # noqa: E402
from openworld import from_spec, to_spec, validate_spec  # noqa: E402
from openworld.state import Action  # noqa: E402


def _run(world, actions):
    s = dict(MINIGRID_INITIAL)
    for a in actions:
        s = dict(world.transition.step(s, Action(a)))
    return s


def test_solution_reaches_goal():
    s = _run(build_minigrid_world(), SOLUTION)
    assert solved(s) and (s["x"], s["y"]) == (4, 4)


def test_walls_block_forward():
    w = build_minigrid_world()
    # facing up (dir 3) from start (1,1): (1,0) is a border wall -> no move
    s = dict(MINIGRID_INITIAL); s["dir"] = 3
    s2 = dict(w.transition.step(s, Action("forward")))
    assert (s2["x"], s2["y"]) == (1, 1)


def test_door_requires_key():
    w = build_minigrid_world()
    # at (2,2) facing the door (3,2) without the key: toggle does nothing
    s = {"x": 2, "y": 2, "dir": 0, "has_key": False, "door_open": False, "terminated": False}
    assert w.transition.step(s, Action("toggle"))["door_open"] is False
    s["has_key"] = True
    assert w.transition.step(s, Action("toggle"))["door_open"] is True


def test_world_round_trips():
    w = build_minigrid_world()
    spec = to_spec(w)
    assert validate_spec(spec) == []
    w2 = from_spec(spec, allow_code=True)
    assert _run(w2, SOLUTION) == _run(w, SOLUTION)
