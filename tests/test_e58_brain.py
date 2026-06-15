"""Tests for E58 brain simulator: planning, memory learning, brain-world round-trip."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "experiments"))
import e58_brain as b                                       # noqa: E402

from openworld import from_spec, to_spec, validate_spec     # noqa: E402


def test_env_reachable_and_plan_tree():
    e = b.make_env(58)
    assert e["L"] is not None and e["L"] >= 1
    plan = b.plan_tree(b.full_model(e), e["start"], e["goal"], e["L"])
    assert plan is not None and len(plan) == e["L"]         # shortest path within depth
    node = e["start"]
    for t in plan:
        node = e["edges"][node][t]
    assert node == e["goal"]                                # the plan actually works


def test_plan_tree_respects_depth_horizon():
    e = b.make_env(58)
    if e["L"] >= 2:                                         # too-shallow lookahead fails
        assert b.plan_tree(b.full_model(e), e["start"], e["goal"], e["L"] - 1) is None


def test_brain_learns_and_beats_memoryless():
    e = next(x for x in (b.make_env(s) for s in range(58, 358))
             if x["L"] and 3 <= x["L"] <= 6)
    mem = b.fresh_mem()
    first = b.run_episode(e, mem, depth=12)
    last = first
    for _ in range(5):
        last = b.run_episode(e, mem, depth=12)             # persistent memory
    memoryless = b.run_episode(e, b.fresh_mem(), depth=12)  # wiped
    assert last < first and last <= e["L"] + 1             # learns, near-optimal
    assert last < memoryless                                # memory wins


def test_brain_world_round_trips():
    brain = b.brain_world(b.make_env(58))
    spec = to_spec(brain, card={"tags": ["brain"]})
    assert validate_spec(spec) == []
    w2 = from_spec(spec, allow_code=True)
    acts = ["tick", "environment:tool0", "tick"]
    assert b._rollout(brain, acts) == b._rollout(w2, acts)
    assert spec["composite"]["children"].keys() >= {"conscious", "unconscious", "environment"}
