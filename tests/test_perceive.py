"""Tests for the perception boundary (multimodal inputs -> symbolic state)."""

import pytest

from openworld import (
    Action, MockLLM, MockPerceptor, Observation, PerceptionError,
    PerceptionGate, TextPerceptor, World,
)
from openworld.transition import FunctionTransition


def test_observation_validates_modality_and_hashes():
    obs = Observation("text", "patient is stable")
    assert obs.sha256 == Observation("text", "patient is stable").sha256
    assert obs.sha256 != Observation("text", "patient is critical").sha256
    assert Observation("image", b"\x00\x01").sha256  # bytes payload hashes too
    with pytest.raises(PerceptionError):
        Observation("smell", "nope")


def test_gate_accepts_owned_in_range_fields():
    p = MockPerceptor(produces=["hr"], deltas=[{"hr": 80}],
                      schema={"hr": (int, (0, 250))})
    assert PerceptionGate().check(p, {"hr": 80}) == {"hr": 80}


def test_gate_rejects_unowned_field():
    p = MockPerceptor(produces=["hr"], deltas=[{}])
    with pytest.raises(PerceptionError, match="does not"):
        PerceptionGate().check(p, {"bp": 120})


def test_gate_rejects_wrong_type_and_out_of_range():
    p = MockPerceptor(produces=["hr"], deltas=[{}], schema={"hr": (int, (0, 250))})
    with pytest.raises(PerceptionError, match="expected int"):
        PerceptionGate().check(p, {"hr": "fast"})
    with pytest.raises(PerceptionError, match="out of range"):
        PerceptionGate().check(p, {"hr": 999})


def test_text_perceptor_extracts_json_via_mock_llm():
    llm = MockLLM(['{"hr": 88, "note": "ignored"}'])
    p = TextPerceptor(llm, produces=["hr"], schema={"hr": int})
    assert p.perceive(Observation("text", "heart rate 88")) == {"hr": 88}  # owned only


def _counter_world():
    def step(state, action):
        s = dict(state)
        if action["name"] == "tick":
            s["count"] += s.get("hr", 0)
        return s
    return World(name="c", description="counter", initial_state={"count": 0, "hr": 0},
                 actions=["tick"], transition=FunctionTransition(step))


def test_observe_commits_delta_records_provenance_then_steps():
    world = _counter_world()
    p = MockPerceptor(produces=["hr"], deltas=[{"hr": 10}], schema={"hr": (int, (0, 250))})
    world.observe(Observation("text", "hr is 10"), p)
    assert world.state["hr"] == 10                      # perception committed to state
    assert world.perceptions[0]["delta"] == {"hr": 10}  # provenance recorded
    assert world.perceptions[0]["input_sha256"]
    world.step(Action("tick"))                          # dynamics run normally over it
    assert world.state["count"] == 10


def test_observe_supports_dict_of_perceptors_by_modality():
    world = _counter_world()
    perceptors = {"text": MockPerceptor(["hr"], [{"hr": 5}])}
    world.observe([Observation("text", "hr 5")], perceptors)
    assert world.state["hr"] == 5


def test_world_without_observe_is_unchanged():
    world = _counter_world()
    world.step(Action("tick"))
    assert world.state == {"count": 0, "hr": 0}
    assert "perceptions" not in world.__dict__   # nothing added unless observe() is used


def test_gate_rejection_surfaces_through_observe():
    world = _counter_world()
    bad = MockPerceptor(produces=["hr"], deltas=[{"hr": 999}], schema={"hr": (int, (0, 250))})
    with pytest.raises(PerceptionError):
        world.observe(Observation("text", "hr 999"), bad)
    assert world.state["hr"] == 0  # rejected percept never touched the state
