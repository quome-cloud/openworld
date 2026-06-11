"""Tests for the generate -> verify -> repair synthesis loop."""

import pytest

from openworld import (
    Action,
    MockLLM,
    SynthesisError,
    Verifier,
    World,
    WorldState,
    synthesize_transition,
)

GOOD_CODE = """\
```python
def transition(state, action):
    next_state = dict(state)
    if action["name"] == "pick" and next_state["apples"] > 0:
        next_state["apples"] = next_state["apples"] - 1
    return next_state
```"""

BROKEN_CODE = """\
```python
def transition(state, action):
    return state["missing_key"]
```"""

NEGATIVE_CODE = """\
```python
def transition(state, action):
    next_state = dict(state)
    next_state["apples"] = -5
    return next_state
```"""


def make_world(llm):
    return World(
        name="orchard",
        description="An orchard with apples to pick.",
        initial_state={"apples": 3},
        actions=["pick"],
        rules=["pick removes one apple while any remain"],
        llm=llm,
    )


def test_synthesis_accepts_good_code_first_try():
    llm = MockLLM([GOOD_CODE])
    world = make_world(llm)
    transition = world.compile()
    assert "def transition" in transition.code
    world.step(Action("pick"))
    assert world.state["apples"] == 2


def test_synthesis_repairs_after_failure():
    llm = MockLLM([BROKEN_CODE, GOOD_CODE])
    transition = synthesize_transition(
        llm,
        description="orchard",
        initial_state=WorldState({"apples": 3}),
        actions=["pick"],
    )
    assert "missing_key" not in transition.code
    # The repair prompt should contain the verifier feedback.
    repair_prompt = llm.calls[1][-1]["content"]
    assert "failed verification" in repair_prompt


def test_synthesis_enforces_invariants():
    llm = MockLLM([NEGATIVE_CODE, GOOD_CODE])
    verifier = Verifier(
        initial_state=WorldState({"apples": 3}),
        sample_actions=[Action("pick")],
        invariants=[("apples never negative", lambda s: s["apples"] >= 0)],
    )
    transition = synthesize_transition(
        llm,
        description="orchard",
        initial_state=WorldState({"apples": 3}),
        actions=["pick"],
        verifier=verifier,
    )
    assert "-5" not in transition.code


def test_synthesis_gives_up_with_history():
    llm = MockLLM([BROKEN_CODE])
    with pytest.raises(SynthesisError) as excinfo:
        synthesize_transition(
            llm,
            description="orchard",
            initial_state=WorldState({"apples": 3}),
            actions=["pick"],
            max_iters=2,
        )
    assert len(excinfo.value.attempts) == 2


def test_semantic_critic_feedback_loop():
    generator = MockLLM([GOOD_CODE, GOOD_CODE])
    critic = MockLLM(["FAIL: does not handle the wait action", "PASS"])
    verifier = Verifier(
        initial_state=WorldState({"apples": 3}),
        sample_actions=[Action("pick")],
        critic=critic,
    )
    transition = synthesize_transition(
        generator,
        description="orchard",
        initial_state=WorldState({"apples": 3}),
        actions=["pick"],
        verifier=verifier,
    )
    assert transition is not None
    assert len(critic.calls) == 2


def test_code_transition_save_and_load(tmp_path):
    llm = MockLLM([GOOD_CODE])
    world = make_world(llm)
    path = tmp_path / "dynamics.py"
    world.compile(save_to=path)
    assert path.exists()
    from openworld import CodeTransition

    loaded = CodeTransition.load(path)
    assert loaded.step(WorldState({"apples": 1}), Action("pick"))["apples"] == 0
