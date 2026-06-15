"""Tests for portable world-model specs: serialize, validate, round-trip, safety."""

import pytest

from openworld import (CodeTransition, CompositeWorld, FunctionTransition, World,
                       from_spec, spec_from_json, spec_to_json, to_mermaid,
                       to_spec, validate_spec)
from openworld.compose import Aggregator, Binding, Bridge
from openworld.spec import SPEC_VERSION, SpecError
from openworld.state import Action

# A module-level transition so inspect.getsource succeeds (round-trippable).
COUNTER_CODE = """
def transition(state, action):
    s = dict(state)
    if action["name"] == "inc":
        s["n"] = s["n"] + 1
    elif action["name"] == "dec":
        s["n"] = max(0, s["n"] - 1)
    return s
"""


def counter_world():
    return World(
        name="counter",
        description="A bounded integer counter.",
        initial_state={"n": 3, "label": "demo", "history": [1, 2, 3]},
        actions=["inc", "dec", "noop"],
        rules=["'inc' adds one; 'dec' subtracts one but never below zero."],
        transition=CodeTransition(COUNTER_CODE),
    )


def market_growth(state, action):
    s = dict(state)
    if action["name"] == "grow":
        s["price"] = round(s["price"] * 1.1, 4)
        s["volume"] = s["volume"] + 5
    return s


def market_world():
    return World(
        name="market",
        description="A market with a price and volume.",
        initial_state={"price": 2.0, "volume": 10},
        actions=["grow", "hold"],
        rules=["'grow' raises price 10% and volume by 5."],
        transition=FunctionTransition(market_growth),
    )


# --------------------------------------------------------------------------- #
# leaf serialization
# --------------------------------------------------------------------------- #
def test_leaf_to_spec_shape():
    spec = to_spec(counter_world())
    assert spec["openworld_spec_version"] == SPEC_VERSION
    assert spec["name"] == "counter"
    assert spec["state_schema"] == {"n": "int", "label": "str", "history": "list[int]"}
    assert spec["initial_state"]["n"] == 3
    assert spec["actions"] == ["inc", "dec", "noop"]
    assert spec["transition"]["kind"] == "code"
    assert "def transition" in spec["transition"]["code"]


def test_leaf_preview_series_present_and_lively():
    spec = to_spec(counter_world(), preview_steps=6)
    # auto-picked action should be 'inc' (moves n the most); n rises 3..9
    assert spec["preview"]["series"]["n"][0] == 3
    assert spec["preview"]["series"]["n"][-1] > 3


def test_function_transition_serializes_via_source():
    spec = to_spec(market_world())
    assert spec["transition"]["kind"] == "code"          # source recovered
    assert spec["transition"]["from_function"] is True
    assert spec["transition"]["func_name"] == "market_growth"


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_validate_good_spec_is_clean():
    assert validate_spec(to_spec(counter_world())) == []


def test_validate_catches_problems():
    spec = to_spec(counter_world())
    spec["openworld_spec_version"] = "9.9"
    del spec["name"]
    spec["transition"]["kind"] = "telepathy"
    problems = validate_spec(spec)
    assert any("openworld_spec_version" in p for p in problems)
    assert any("name" in p for p in problems)
    assert any("telepathy" in p for p in problems)


def test_json_round_trips_the_dict():
    spec = to_spec(counter_world())
    assert spec_from_json(spec_to_json(spec)) == spec


# --------------------------------------------------------------------------- #
# deserialization, round-trip, safety
# --------------------------------------------------------------------------- #
def _rollout(world, actions, agent=None):
    states = []
    s = world.initial_state.copy()
    for a in actions:
        s = world.transition.step(s, Action(a, agent=agent))
        states.append(dict(s))
    return states


def test_round_trip_behavioral_with_allow_code():
    w = counter_world()
    w2 = from_spec(to_spec(w), allow_code=True)
    acts = ["inc", "inc", "dec", "noop", "inc"]
    assert _rollout(w, acts) == _rollout(w2, acts)


def test_function_world_round_trips_through_code():
    w = market_world()
    w2 = from_spec(to_spec(w), allow_code=True)
    acts = ["grow", "grow", "hold", "grow"]
    assert _rollout(w, acts) == _rollout(w2, acts)


def test_code_is_inert_by_default():
    w2 = from_spec(to_spec(counter_world()))      # allow_code defaults False
    assert w2.name == "counter"
    assert w2.actions == ["inc", "dec", "noop"]   # described
    with pytest.raises(SpecError):
        w2.transition.step(w2.initial_state.copy(), Action("inc"))


# --------------------------------------------------------------------------- #
# composite round-trip
# --------------------------------------------------------------------------- #
def total_value(children):
    return children["market"]["price"] * children["market"]["volume"]


BRIDGE_CODE = """
def transition(state, action):
    a = dict(state["a"]); b = dict(state["b"])
    b["volume"] = b["volume"] + a["n"]
    return {"a": a, "b": b}
"""


def economy_world():
    return CompositeWorld(
        name="economy",
        children={"shop": counter_world(), "market": market_world()},
        bridges=[Bridge(name="restock", a="shop", b="market",
                        transition=CodeTransition(BRIDGE_CODE),
                        description="shop count feeds market volume")],
        aggregators=[Aggregator(name="total_value", fn=total_value)],
        bindings=[Binding(source_path=("_agg", "total_value"),
                          child="market", key="ref_value")],
        default_actions={"shop": "inc", "market": "grow"},
        timescales={"shop": 1, "market": 1},
        agents={"trader": {"loc": "market"}},
        description="A two-world economy: a shop and a market, bridged.",
    )


def test_composite_to_spec_structure():
    spec = to_spec(economy_world())
    assert "composite" in spec
    comp = spec["composite"]
    assert set(comp["children"]) == {"shop", "market"}
    assert comp["bridges"][0]["name"] == "restock"
    assert comp["aggregators"][0]["name"] == "total_value"
    assert comp["default_actions"] == {"shop": "inc", "market": "grow"}
    assert comp["agents"] == {"trader": {"loc": "market"}}
    assert validate_spec(spec) == []


def test_composite_round_trips_behaviorally():
    w = economy_world()
    w2 = from_spec(to_spec(w), allow_code=True)
    acts = ["tick", "tick", "shop:inc", "tick"]
    assert _rollout(w, acts) == _rollout(w2, acts)


# --------------------------------------------------------------------------- #
# mermaid export
# --------------------------------------------------------------------------- #
def test_mermaid_leaf_is_state_machine():
    mm = to_mermaid(to_spec(counter_world()))
    assert mm.startswith("flowchart LR")
    assert "-->|inc|" in mm                      # action-labeled transition
    assert ":::start" in mm                      # initial state marked


def test_mermaid_composite_is_dataflow():
    mm = to_mermaid(to_spec(economy_world()))
    assert mm.startswith("flowchart TD")
    assert "c_shop" in mm and "c_market" in mm
    assert "|restock|" in mm                     # bridge edge
    assert "Σ total_value" in mm                 # aggregator node
