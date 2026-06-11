"""Core unit tests: state, parsing, sandbox, objectives."""

import pytest

from openworld import Action, Dial, Objective, ObjectiveSuite, SandboxError, WorldState
from openworld.parsing import extract_code, extract_json
from openworld.sandbox import run_transition_code


def test_world_state_copy_and_diff():
    a = WorldState({"apples": 10, "nested": {"x": 1}})
    b = a.copy()
    b["apples"] = 9
    b["nested"]["x"] = 2
    assert a["apples"] == 10
    assert a["nested"]["x"] == 1  # deep copy
    diff = a.diff(b)
    assert diff["apples"] == (10, 9)
    assert "nested" in diff


def test_world_state_json_roundtrip():
    state = WorldState({"b": 2, "a": 1})
    assert WorldState.from_json(state.to_json()) == state


def test_extract_code_from_fence():
    text = "Here you go:\n```python\ndef transition(s, a):\n    return s\n```\nDone."
    assert extract_code(text).startswith("def transition")


def test_extract_json_finds_first_balanced_object():
    text = 'Sure! {"action": "pick", "params": {"n": 1}} hope that helps'
    assert extract_json(text) == {"action": "pick", "params": {"n": 1}}
    assert extract_json("no json here") is None


def test_sandbox_runs_valid_code():
    code = "def transition(state, action):\n    state['x'] = state['x'] + 1\n    return state"
    result = run_transition_code(code, {"x": 1}, Action("noop").to_dict())
    assert result == {"x": 2}


def test_sandbox_blocks_imports_and_bad_returns():
    with pytest.raises(SandboxError):
        run_transition_code("import os\ndef transition(s, a):\n    return s", {}, {})
    with pytest.raises(SandboxError):
        run_transition_code("def transition(s, a):\n    return 42", {}, {})
    with pytest.raises(SandboxError):
        run_transition_code("x = 1", {}, {})  # no transition function


def test_sandbox_never_leaks_mutations_to_caller():
    # Generated code commonly shallow-copies state and mutates nested dicts;
    # the caller's structures must stay untouched.
    code = (
        "def transition(state, action):\n"
        "    next_state = state.copy()\n"
        "    next_state['harvested'].setdefault(action.get('agent'), 0)\n"
        "    next_state['harvested'][action.get('agent')] += 1\n"
        "    return next_state\n"
    )
    state = {"harvested": {"alice": 0}}
    run_transition_code(code, state, Action("pick", agent="bob").to_dict())
    assert state == {"harvested": {"alice": 0}}


def test_objective_suite_weights_and_dials():
    dial = Dial("morality", value=0.5)
    suite = ObjectiveSuite([
        Objective("welfare", fn=lambda s, a, ns: 2.0, weight=1.0),
        Objective("harmony", fn=lambda s, a, ns: 4.0, weight=dial),
    ])
    scores = suite.score(WorldState(), Action("noop"), WorldState())
    assert scores["welfare"] == 2.0
    assert scores["harmony"] == 4.0
    assert scores["aggregate"] == pytest.approx(2.0 + 0.5 * 4.0)
    dial.set(1.0)
    assert suite.score(WorldState(), Action("noop"), WorldState())["aggregate"] == 6.0
    with pytest.raises(ValueError):
        dial.set(2.0)
    assert suite.dials() == {"morality": dial}
