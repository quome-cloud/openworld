# Agent Traversal (`Route`, registry, `travel`, `observe`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sub-project A2 of the composite-worlds spec: agents located in (and moving between) the worlds of a composite — `Route`, the `_agents` registry, the `travel` action, and the `observe`/`legal_actions` helpers.

**Architecture:** Extends `openworld/compose.py`. A `Route` is a `Bridge` with crossing rights (optional `on_cross` transition over `{"agent","a","b"}`); agent locations are full namespace paths stored in root-level `state["_agents"]`; bridge endpoint resolution is generalized to deep paths so routes (and coupling bridges) can join nodes at any depth. Observation helpers expose local detail + ancestor aggregates + route-adjacent summaries.

**Tech Stack:** stdlib; existing compose/state/transition primitives. Spec: `docs/superpowers/specs/2026-06-12-composite-worlds-design.md` §A2. Base branch: `composite-worlds`.

---

### Task 1: Route, registry, travel, deep-path bridges

**Files:**
- Modify: `openworld/compose.py`
- Create: `tests/test_traverse.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_traverse.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_traverse.py -v`
Expected: ImportError (`AGENTS_KEY`, `Route` not in `openworld.compose`).

- [ ] **Step 3: Implement.** In `openworld/compose.py`:

(a) Add the constant under `AGG_KEY`:

```python
AGENTS_KEY = "_agents"
```

(b) Make `Bridge.transition` optional and add `Route` after `Bridge` (redeclare the field with a default in the dataclass):

```python
@dataclass
class Route(Bridge):
    """A bridge agents may cross.

    Inherits the optional per-step coupling `transition` (None = pure path);
    adds `on_cross`: an optional Transition over the three-slot dict
    {"agent": <attrs>, "a": <source slice>, "b": <destination slice>},
    stepped with Action("cross") - tolls, visas, capacity - synthesizable
    and verifiable like any other dynamics.
    """

    on_cross: Optional[Transition] = None
```

and change `Bridge`'s field to `transition: Optional[Transition] = None`
(moving it after `b`, keeping `description`/`rules` defaults after it).

(c) In `CompositeWorld.__init__`, accept `agents: Optional[Dict[str, dict]] = None`,
store `self.agents = {k: dict(v) for k, v in (agents or {}).items()}`, add
`initial[AGENTS_KEY] = {k: dict(v) for k, v in self.agents.items()}` before
the `super().__init__` call **only when agents were given**, and extend the
actions list with `"travel"` when any bridge is a `Route`:

```python
        if any(isinstance(b, Route) for b in self.bridges):
            actions.append("travel")
```

(d) Generalize endpoint access with two private helpers on `CompositeWorld`
(deep colon-paths; a flat name is the one-part case):

```python
    def _resolve(self, state: Dict[str, Any], path: str) -> dict:
        node: Any = state
        for part in path.split(":"):
            node = node[part]
        return dict(node)

    def _write_back(self, state: Dict[str, Any], path: str, value: dict) -> None:
        parts = path.split(":")
        node: Any = state
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = dict(value)
```

(e) In `CompositeTransition.step`, change the bridge-firing loop to skip
pure paths and use deep resolution:

```python
        for bridge in c.bridges:
            if bridge.transition is None:
                continue
            sa, sb = bridge.flow(c._resolve(s, bridge.a), c._resolve(s, bridge.b))
            c._write_back(s, bridge.a, sa)
            c._write_back(s, bridge.b, sb)
```

and add a `travel` branch between the `tick` and namespaced-routing branches:

```python
        elif action.name == "travel":
            return c._travel(s, action)
```

(f) Add `_travel` to `CompositeWorld`:

```python
    def _travel(self, s: WorldState, action: Action) -> WorldState:
        """Move an agent along a Route; tolerant no-op when illegal."""
        name = action.params.get("agent")
        dest = action.params.get("to")
        registry = s.get(AGENTS_KEY, {})
        if name not in registry or not dest:
            return s
        here = registry[name]["at"]
        route = next(
            (r for r in self.bridges
             if isinstance(r, Route) and {r.a, r.b} == {here, dest}),
            None,
        )
        if route is None or dest == here:
            return s
        attrs = dict(registry[name])
        src, dst = self._resolve(s, here), self._resolve(s, dest)
        if route.on_cross is not None:
            out = route.on_cross.step(
                WorldState({"agent": attrs, "a": src, "b": dst}),
                Action("cross", agent=name),
            )
            attrs, src, dst = dict(out["agent"]), dict(out["a"]), dict(out["b"])
        attrs["at"] = dest
        s[AGENTS_KEY] = {**registry, name: attrs}
        self._write_back(s, here, src)
        self._write_back(s, dest, dst)
        s[AGG_KEY] = self._aggregates(s)
        return s
```

NOTE: `_aggregates` reads `state[ns]` for each direct child only, so the
registry key never feeds aggregators. Travel does NOT fire coupling bridges
(crossing is an administrative move, not a world tick) but does refresh
aggregates since slices may have changed.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_traverse.py tests/test_compose.py -v` — all PASS (the compose suite guards the Bridge-default refactor). Then `python -m pytest tests/ -q`.

- [ ] **Step 5: Commit**

```bash
git add openworld/compose.py tests/test_traverse.py
git commit -m "compose: agent traversal - Route, _agents registry, travel with on_cross, deep-path bridges"
```

---

### Task 2: observe + legal_actions helpers, exports

**Files:**
- Modify: `openworld/compose.py` (append)
- Modify: `tests/test_traverse.py` (append)
- Modify: `openworld/__init__.py`

- [ ] **Step 1: Write the failing tests** (append; extend the compose import with `legal_actions, observe`)

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_traverse.py -v` — ImportError for `observe`.

- [ ] **Step 3: Implement** (append to `openworld/compose.py`)

```python
def observe(composite: CompositeWorld, state: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """An agent's scoped view: full detail at its own node, aggregates of
    every ancestor level, and summaries of route-adjacent locations.

    This is the hierarchical-attention contract: planners reason locally
    against exact local state and globally against derived aggregates,
    without ever holding the whole macrostate.
    """
    registry = state.get(AGENTS_KEY, {})
    if agent_name not in registry:
        raise KeyError(f"unknown agent {agent_name!r}")
    path = registry[agent_name]["at"]
    view: Dict[str, Any] = {
        "agent": dict(registry[agent_name]),
        "location": path,
        "local": composite._resolve(state, path),
        "ancestors": {},
        "neighbors": {},
    }
    node: Any = state
    prefix: list = []
    for part in path.split(":"):
        if AGG_KEY in node:
            view["ancestors"][":".join(prefix) or "<root>"] = dict(node[AGG_KEY])
        node = node[part]
        prefix.append(part)
    for bridge in composite.bridges:
        if not isinstance(bridge, Route):
            continue
        other = bridge.b if bridge.a == path else bridge.a if bridge.b == path else None
        if other is not None:
            slice_ = composite._resolve(state, other)
            view["neighbors"][other] = dict(slice_.get(AGG_KEY, slice_))
    return view


def legal_actions(composite: CompositeWorld, state: Dict[str, Any], agent_name: str) -> list:
    """The namespaced actions available to an agent at its location, plus a
    'travel:<dest>' token per incident route. Callers turn a travel token
    into Action('travel', params={'agent': name, 'to': dest})."""
    registry = state.get(AGENTS_KEY, {})
    if agent_name not in registry:
        raise KeyError(f"unknown agent {agent_name!r}")
    path = registry[agent_name]["at"]
    world: Any = composite
    for part in path.split(":"):
        world = world.children[part]
    actions = [f"{path}:{act}" for act in world.actions]
    for bridge in composite.bridges:
        if not isinstance(bridge, Route):
            continue
        other = bridge.b if bridge.a == path else bridge.a if bridge.b == path else None
        if other is not None:
            actions.append(f"travel:{other}")
    return actions
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_traverse.py -v` — all PASS.

- [ ] **Step 5: Exports.** In `openworld/__init__.py`, extend the compose import to
`from .compose import (Aggregator, Binding, Bridge, CompositeWorld, Route, compile_bridge, legal_actions, observe)`
and add `"Route", "legal_actions", "observe"` to `__all__` (matching the file's existing case-block convention).

- [ ] **Step 6: Full verification**

```bash
python -m pytest tests/ -q
python -c "from openworld import Route, observe, legal_actions; print('exports ok')"
```

Expected: full suite passes (~167); exports import.

- [ ] **Step 7: Commit**

```bash
git add openworld/compose.py openworld/__init__.py tests/test_traverse.py
git commit -m "compose: observe and legal_actions - scoped views for agents in composites"
```
