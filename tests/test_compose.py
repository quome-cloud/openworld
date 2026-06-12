"""Tests for composing worlds: composites, bridges, aggregators, bindings."""

from openworld import Action, World
from openworld.compose import AGG_KEY, Aggregator, Binding, Bridge, CompositeWorld
from openworld.transition import Transition


class AddTransition(Transition):
    """'work' adds `rate` to `output`; anything else is a no-op."""

    def step(self, state, action):
        s = state.copy()
        if action.name == "work":
            s["output"] += s.get("rate", 1)
        return s


def make_city(name="city", output=0, rate=1):
    return World(
        name=name,
        description="a city that produces output at its rate",
        initial_state={"output": output, "rate": rate},
        actions=["work", "wait"],
        transition=AddTransition(),
    )


def make_pair(**kwargs):
    return CompositeWorld(
        name="pair",
        children={"x": make_city(rate=2), "y": make_city(rate=3)},
        **kwargs,
    )


def test_initial_state_nests_children_and_actions_are_namespaced():
    comp = make_pair()
    assert comp.state["x"] == {"output": 0, "rate": 2}
    assert comp.state["y"] == {"output": 0, "rate": 3}
    assert AGG_KEY in comp.state
    assert "x:work" in comp.actions and "y:wait" in comp.actions
    assert "tick" in comp.actions


def test_routing_steps_only_the_named_child():
    comp = make_pair()
    s = comp.step(Action("x:work"))
    assert s["x"]["output"] == 2
    assert s["y"]["output"] == 0


def test_unknown_namespace_or_action_is_a_noop():
    comp = make_pair()
    before = comp.state.copy()
    assert comp.step(Action("zz:work")) == before
    assert comp.step(Action("dance")) == before


def test_tick_uses_default_actions_and_timescales():
    comp = make_pair(
        default_actions={"x": "work", "y": "work"},
        timescales={"x": 3},
    )
    s = comp.step(Action("tick"))
    assert s["x"]["output"] == 6   # 3 sub-steps at rate 2
    assert s["y"]["output"] == 3   # 1 sub-step at rate 3


def test_tick_skips_children_without_a_default_action():
    comp = make_pair(default_actions={"x": "work"})
    s = comp.step(Action("tick"))
    assert s["x"]["output"] == 2
    assert s["y"]["output"] == 0


def test_reset_restores_nested_initial_state():
    comp = make_pair()
    comp.step(Action("x:work"))
    comp.reset()
    assert comp.state["x"]["output"] == 0
