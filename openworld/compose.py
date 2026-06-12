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

from .state import Action, WorldState
from .transition import Transition
from .world import World

AGG_KEY = "_agg"


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
    transition: Transition
    description: str = ""
    rules: Sequence[str] = ()

    def flow(self, state_a: dict, state_b: dict) -> Tuple[dict, dict]:
        pair = WorldState({"a": state_a, "b": state_b})
        out = self.transition.step(pair, Action("flow"))
        return dict(out["a"]), dict(out["b"])


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
            s[bridge.a], s[bridge.b] = bridge.flow(s[bridge.a], s[bridge.b])
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
    ):
        self.children = dict(children)
        self.bridges = list(bridges)
        self.aggregators = list(aggregators)
        self.bindings = list(bindings)
        self.timescales = dict(timescales or {})
        self.default_actions = dict(default_actions or {})
        initial: Dict[str, Any] = {
            ns: dict(child.initial_state) for ns, child in self.children.items()
        }
        initial[AGG_KEY] = self._aggregates(initial)
        actions = [
            f"{ns}:{act}"
            for ns, child in self.children.items()
            for act in child.actions
        ] + ["tick"]
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
