"""Tests for agent traversal: routes, the registry, travel, scoped views."""

from openworld import Action, World
from openworld.compose import (
    AGENTS_KEY,
    AGG_KEY,
    Aggregator,
    Bridge,
    CompositeWorld,
    Route,
    CompositeTransition,  # noqa: F401 - imported to assert it stays public
    legal_actions,
    observe,
)
from openworld.transition import Transition


class AddTransition(Transition):
    """'work' adds `rate` to `output`; anything else is a no-op."""

    def step(self, state, action):
        s = state.copy()
        if action.name == "work":
            s["output"] += s.get("rate", 1)
        return s


class TollTransition(Transition):
    """Crossing costs the agent 2 coins, paid into the destination treasury."""

    def step(self, state, action):
        s = state.copy()
        if action.name == "cross" and s["agent"].get("coins", 0) >= 2:
            s["agent"]["coins"] -= 2
            s["b"]["treasury"] = s["b"].get("treasury", 0) + 2
        return s


def make_town(name, treasury=0):
    return World(
        name=name,
        description="a town",
        initial_state={"output": 0, "rate": 1, "treasury": treasury},
        actions=["work", "wait"],
        transition=AddTransition(),
    )


def make_map(**kwargs):
    return CompositeWorld(
        name="map",
        children={"sf": make_town("sf"), "ny": make_town("ny")},
        agents={"alice": {"at": "sf", "coins": 5}},
        **kwargs,
    )


def test_registry_initialised_and_travel_action_advertised():
    comp = make_map(bridges=[Route("road", "sf", "ny", transition=None)])
    assert comp.state[AGENTS_KEY] == {"alice": {"at": "sf", "coins": 5}}
    assert "travel" in comp.actions


def test_travel_moves_agent_along_route_in_both_directions():
    comp = make_map(bridges=[Route("road", "sf", "ny", transition=None)])
    s = comp.step(Action("travel", params={"agent": "alice", "to": "ny"}))
    assert s[AGENTS_KEY]["alice"]["at"] == "ny"
    s = comp.step(Action("travel", params={"agent": "alice", "to": "sf"}))
    assert s[AGENTS_KEY]["alice"]["at"] == "sf"


def test_illegal_travel_is_a_noop():
    comp = make_map(bridges=[Route("road", "sf", "ny", transition=None)])
    before = comp.state.copy()
    assert comp.step(Action("travel", params={"agent": "bob", "to": "ny"})) == before
    assert comp.step(Action("travel", params={"agent": "alice", "to": "sf"})) == before
    comp2 = make_map()  # no routes at all
    before2 = comp2.state.copy()
    assert comp2.step(Action("travel", params={"agent": "alice", "to": "ny"})) == before2


def test_on_cross_charges_toll_and_conserves_coins():
    comp = make_map(bridges=[Route("toll-road", "sf", "ny",
                                   transition=None, on_cross=TollTransition())])
    s = comp.step(Action("travel", params={"agent": "alice", "to": "ny"}))
    assert s[AGENTS_KEY]["alice"] == {"at": "ny", "coins": 3}
    assert s["ny"]["treasury"] == 2
    total = s[AGENTS_KEY]["alice"]["coins"] + s["ny"]["treasury"] + s["sf"]["treasury"]
    assert total == 5


def test_pure_route_does_not_fire_as_coupling_bridge():
    comp = make_map(bridges=[Route("road", "sf", "ny", transition=None)])
    s = comp.step(Action("sf:work"))
    assert s["sf"]["output"] == 1 and s["ny"]["output"] == 0


def test_deep_path_route_in_nested_composite():
    inner = CompositeWorld(name="usa",
                           children={"sf": make_town("sf"), "ny": make_town("ny")})
    comp = CompositeWorld(
        name="earth",
        children={"usa": inner},
        agents={"alice": {"at": "usa:sf", "coins": 5}},
        bridges=[Route("flight", "usa:sf", "usa:ny",
                       transition=None, on_cross=TollTransition())],
    )
    s = comp.step(Action("travel", params={"agent": "alice", "to": "usa:ny"}))
    assert s[AGENTS_KEY]["alice"]["at"] == "usa:ny"
    assert s["usa"]["ny"]["treasury"] == 2


def test_registry_survives_tick_and_routing():
    comp = make_map(default_actions={"sf": "work"},
                    bridges=[Route("road", "sf", "ny", transition=None)])
    comp.step(Action("tick"))
    comp.step(Action("sf:work"))
    assert comp.state[AGENTS_KEY]["alice"]["at"] == "sf"


def make_earth():
    west = CompositeWorld(
        name="west",
        children={"sf": make_town("sf"), "la": make_town("la")},
        aggregators=[Aggregator("gdp", lambda kids: sum(c["output"] for c in kids.values()))],
    )
    east = CompositeWorld(
        name="east",
        children={"ny": make_town("ny")},
        aggregators=[Aggregator("gdp", lambda kids: kids["ny"]["output"])],
    )
    return CompositeWorld(
        name="earth",
        children={"west": west, "east": east},
        agents={"alice": {"at": "west:sf", "coins": 5}},
        bridges=[Route("flight", "west:sf", "east:ny", transition=None)],
        aggregators=[Aggregator("world_gdp",
                                lambda kids: kids["west"][AGG_KEY]["gdp"]
                                + kids["east"][AGG_KEY]["gdp"])],
    )


def test_observe_scopes_detail_aggregates_and_neighbors():
    earth = make_earth()
    earth.step(Action("west:sf:work"))
    view = observe(earth, earth.state, "alice")
    assert view["location"] == "west:sf"
    assert view["local"]["output"] == 1                      # leaf detail
    assert view["ancestors"]["<root>"]["world_gdp"] == 1     # root aggregates
    assert view["ancestors"]["west"]["gdp"] == 1             # mid-level aggregates
    assert "east:ny" in view["neighbors"]                    # route-adjacent
    assert "la" not in view["neighbors"]                     # non-adjacent sibling
    assert "la" not in view and "west:la" not in view.get("neighbors", {})


def test_legal_actions_lists_local_actions_and_travel_options():
    earth = make_earth()
    acts = legal_actions(earth, earth.state, "alice")
    assert "west:sf:work" in acts and "west:sf:wait" in acts
    assert "travel:east:ny" in acts
    assert not any(a.startswith("east:ny:") for a in acts)


def test_observe_unknown_agent_raises():
    earth = make_earth()
    try:
        observe(earth, earth.state, "nobody")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")
