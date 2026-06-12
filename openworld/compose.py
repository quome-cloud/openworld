"""Composing worlds: composites, bridges, aggregators, downward bindings.

A CompositeWorld is a World whose state nests its children's states under
namespace keys; children run UNMODIFIED (the composite slices a child's
namespace out, steps the child's own transition, and writes it back).
Coupling is explicit: sideways through Bridges, upward through Aggregators
(derived, never simulated), downward through Bindings. Because a
CompositeWorld is itself a World, composition is closed - composites nest,
which is the worlds-within-worlds story (earth > country > state > city).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from .llm import BaseLLM
from .state import Action, WorldState
from .transition import Transition
from .verify import Verifier, synthesize_transition
from .world import World

AGG_KEY = "_agg"
AGENTS_KEY = "_agents"


@dataclass
class Bridge:
    """A coupling between children `a` and `b`.

    `transition` is an ordinary Transition over the two-slot dict
    {"a": <state of a>, "b": <state of b>}, stepped with Action("flow") -
    which means the existing synthesis + verification pipeline applies to
    bridges unchanged (see compile_bridge).
    """

    name: str
    a: str
    b: str
    transition: Optional[Transition] = None
    description: str = ""
    rules: Sequence[str] = ()

    def flow(self, state_a: dict, state_b: dict) -> Tuple[dict, dict]:
        pair = WorldState({"a": state_a, "b": state_b})
        out = self.transition.step(pair, Action("flow"))
        return dict(out["a"]), dict(out["b"])


@dataclass
class Route(Bridge):
    """A bridge agents may cross.

    Inherits the optional per-step coupling `transition` (None = pure path);
    adds `on_cross`: an optional Transition over the three-slot dict
    {"agent": <attrs>, "a": <source slice>, "b": <destination slice>},
    stepped with Action("cross") - tolls, visas, capacity - synthesizable
    and verifiable like any other dynamics.

    on_cross mediates the crossing's effects and may VETO it by setting
    "denied": True on the agent slot (the agent then stays put, while any
    fees or attempt effects it wrote still apply). Without a veto the move
    always completes; gating logic belongs inside on_cross, not the caller.
    """

    on_cross: Optional[Transition] = None


@dataclass
class Aggregator:
    """A derived parent-level quantity: fn(children_states) -> value.

    Recomputed after every composite step into state[AGG_KEY][name]; never
    independently simulated, so summaries cannot drift from the leaves.
    """

    name: str
    fn: Callable[[Dict[str, dict]], Any]


@dataclass
class Binding:
    """Downward parameter flow: copy state[source_path] into a child slice
    before each step. Upward influence happens only through aggregators,
    sideways only through bridges."""

    source_path: Tuple[str, ...]
    child: str
    key: str


class CompositeTransition(Transition):
    """Dynamics of a composite: bind down, route, bridge, aggregate."""

    def __init__(self, composite: "CompositeWorld"):
        self.composite = composite

    def step(self, state: WorldState, action: Action) -> WorldState:
        c = self.composite
        s = state.copy()

        if action.name == "tick":
            c._apply_bindings(s)
            for ns, child in c.children.items():
                act = c.default_actions.get(ns)
                if act is None:
                    continue
                for _ in range(c.timescales.get(ns, 1)):
                    s[ns] = dict(child.transition.step(
                        WorldState(s[ns]), Action(act, agent=action.agent)))
        elif action.name == "travel":
            return c._travel(s, action)
        elif ":" in action.name and action.name.split(":", 1)[0] in c.children:
            ns, act = action.name.split(":", 1)
            c._apply_bindings(s)
            child = c.children[ns]
            s[ns] = dict(child.transition.step(
                WorldState(s[ns]),
                Action(act, params=action.params, agent=action.agent)))
        else:
            return s  # unknown namespace or action: unchanged, no side effects

        for bridge in c.bridges:
            if bridge.transition is None:
                continue
            sa, sb = bridge.flow(c._resolve(s, bridge.a), c._resolve(s, bridge.b))
            c._write_back(s, bridge.a, sa)
            c._write_back(s, bridge.b, sb)
        s[AGG_KEY] = c._aggregates(s)
        return s


class CompositeWorld(World):
    """A World whose children are Worlds. Closed under composition."""

    def __init__(
        self,
        name: str,
        children: Dict[str, World],
        bridges: Sequence[Bridge] = (),
        aggregators: Sequence[Aggregator] = (),
        bindings: Sequence[Binding] = (),
        timescales: Optional[Dict[str, int]] = None,
        default_actions: Optional[Dict[str, str]] = None,
        description: str = "",
        rules: Optional[list] = None,
        agents: Optional[Dict[str, dict]] = None,
    ):
        self.children = dict(children)
        self.bridges = list(bridges)
        self.aggregators = list(aggregators)
        self.bindings = list(bindings)
        self.timescales = dict(timescales or {})
        self.default_actions = dict(default_actions or {})
        self.agents = {k: dict(v) for k, v in (agents or {}).items()}
        initial: Dict[str, Any] = {
            ns: dict(child.initial_state) for ns, child in self.children.items()
        }
        if agents:
            initial[AGENTS_KEY] = {k: dict(v) for k, v in self.agents.items()}
        initial[AGG_KEY] = self._aggregates(initial)
        actions = [
            f"{ns}:{act}"
            for ns, child in self.children.items()
            for act in child.actions
        ] + ["tick"]
        if any(isinstance(b, Route) for b in self.bridges):
            actions.append("travel")
        super().__init__(
            name=name,
            description=description or (
                "Composite of " + ", ".join(self.children) + ". Actions are "
                "namespaced child actions plus 'tick'."),
            initial_state=initial,
            actions=actions,
            rules=rules or [],
            transition=None,
        )
        self.transition = CompositeTransition(self)

    def _aggregates(self, state: Dict[str, Any]) -> Dict[str, Any]:
        kids = {ns: state[ns] for ns in self.children}
        return {agg.name: agg.fn(kids) for agg in self.aggregators}

    def _apply_bindings(self, state: Dict[str, Any]) -> None:
        for binding in self.bindings:
            value: Any = state
            for part in binding.source_path:
                value = value[part]
            state[binding.child][binding.key] = value

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

    def _refresh_path_aggregates(self, state: Dict[str, Any], path: str) -> None:
        """Recompute _agg bottom-up in every composite along `path` (root included)."""
        levels = [(self, state)]
        world: Any = self
        node: Any = state
        for part in path.split(":")[:-1]:
            world = world.children[part]
            node = node[part]
            if isinstance(world, CompositeWorld):
                levels.append((world, node))
        for composite, slice_ in reversed(levels):
            slice_[AGG_KEY] = composite._aggregates(slice_)

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
        denied = False
        if route.on_cross is not None:
            out = route.on_cross.step(
                WorldState({"agent": attrs, "a": src, "b": dst}),
                Action("cross", agent=name),
            )
            attrs, src, dst = dict(out["agent"]), dict(out["a"]), dict(out["b"])
            denied = bool(attrs.pop("denied", False))
        if not denied:
            attrs["at"] = dest
        s[AGENTS_KEY] = {**registry, name: attrs}
        self._write_back(s, here, src)
        self._write_back(s, dest, dst)
        self._refresh_path_aggregates(s, here)
        self._refresh_path_aggregates(s, dest)
        return s


def compile_bridge(
    llm: BaseLLM,
    name: str,
    a: str,
    b: str,
    description: str,
    rules: Sequence[str] = (),
    sample_a: Optional[dict] = None,
    sample_b: Optional[dict] = None,
    invariants: Sequence[tuple] = (),
    critic: Optional[BaseLLM] = None,
    max_iters: int = 4,
) -> Bridge:
    """Synthesize and verify a bridge from plain-language rules.

    The bridge's dynamics are an ordinary transition over the two-slot dict
    {"a": ..., "b": ...} with the single action "flow", so the standard
    generate -> verify relay (sandboxed smoke runs, invariants, optional
    critic) applies unchanged. Invariants are predicates over the two-slot
    state, which is where cross-world conservation laws live.
    """
    initial = WorldState({"a": dict(sample_a or {}), "b": dict(sample_b or {})})
    verifier = Verifier(
        initial_state=initial,
        sample_actions=[Action("flow", agent="smoke_test_agent")],
        invariants=list(invariants),
        critic=critic,
    )
    transition = synthesize_transition(
        llm,
        description=(
            f"A bridge between two worlds, '{a}' (slot 'a') and '{b}' "
            f"(slot 'b'). {description}"),
        initial_state=initial,
        actions=["flow"],
        rules=list(rules),
        verifier=verifier,
        max_iters=max_iters,
    )
    return Bridge(name=name, a=a, b=b, transition=transition,
                  description=description, rules=tuple(rules))


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
