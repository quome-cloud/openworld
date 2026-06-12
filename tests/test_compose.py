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


class TransferTransition(Transition):
    """Move 1 unit of output from the richer side to the poorer, if unequal."""

    def step(self, state, action):
        s = state.copy()
        if action.name != "flow":
            return s
        if s["a"]["output"] > s["b"]["output"]:
            s["a"]["output"] -= 1
            s["b"]["output"] += 1
        elif s["b"]["output"] > s["a"]["output"]:
            s["b"]["output"] -= 1
            s["a"]["output"] += 1
        return s


class StampTransition(Transition):
    """Append this bridge's tag to b's log - used to observe firing order."""

    def __init__(self, tag):
        self.tag = tag

    def step(self, state, action):
        s = state.copy()
        s["b"]["log"] = s["b"].get("log", "") + self.tag
        return s


def test_binding_injects_value_before_child_step():
    comp = CompositeWorld(
        name="bound",
        children={"x": make_city(rate=1), "y": make_city(rate=5)},
        bindings=[Binding(("y", "rate"), "x", "rate")],
    )
    s = comp.step(Action("x:work"))
    assert s["x"]["rate"] == 5      # bound down from y before the step
    assert s["x"]["output"] == 5    # the step used the bound rate


def test_bridge_conserves_total_across_rollout():
    comp = CompositeWorld(
        name="bridged",
        children={"x": make_city(output=10), "y": make_city(output=0)},
        bridges=[Bridge("transfer", "x", "y", TransferTransition())],
    )
    for _ in range(5):
        comp.step(Action("x:wait"))   # the action is a no-op; bridges still fire
    assert comp.state["x"]["output"] == 5
    assert comp.state["y"]["output"] == 5
    assert comp.state["x"]["output"] + comp.state["y"]["output"] == 10


def test_bridges_fire_in_declared_order():
    comp = CompositeWorld(
        name="ordered",
        children={"x": make_city(), "y": make_city()},
        bridges=[
            Bridge("first", "x", "y", StampTransition("A")),
            Bridge("second", "x", "y", StampTransition("B")),
        ],
    )
    s = comp.step(Action("x:wait"))
    assert s["y"]["log"] == "AB"


def test_aggregators_present_initially_and_track_leaves():
    total = Aggregator("total_output", lambda kids: sum(c["output"] for c in kids.values()))
    comp = make_pair(aggregators=[total])
    assert comp.state[AGG_KEY]["total_output"] == 0
    s = comp.step(Action("x:work"))
    assert s[AGG_KEY]["total_output"] == 2
    s = comp.step(Action("y:work"))
    assert s[AGG_KEY]["total_output"] == 5
