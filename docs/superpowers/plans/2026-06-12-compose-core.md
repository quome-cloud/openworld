# Composite Worlds Core (`openworld/compose.py`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sub-project A of the composite-worlds spec: `CompositeWorld`, `Bridge`, `Aggregator`, `Binding`, and `compile_bridge` — composition closed under nesting, children unmodified.

**Architecture:** One flat module `openworld/compose.py` (repo style). `CompositeWorld(World)` nests child states under namespace keys; `CompositeTransition` routes namespaced actions to child transitions, fires bridges, recomputes aggregates. Bridges are ordinary `Transition`s over a `{"a":…, "b":…}` two-slot dict, so the existing synthesis/verifier pipeline applies to them unchanged.

**Tech Stack:** stdlib only; existing `openworld` primitives (`World`, `WorldState`, `Action`, `Transition`, `Verifier`, `synthesize_transition`). Spec: `docs/superpowers/specs/2026-06-12-composite-worlds-design.md`.

**Base branch:** `composite-worlds`.

**Contract note for the implementer:** test fixtures subclass `Transition` directly (never `FunctionTransition`) so the plan is independent of `FunctionTransition`'s callable contract. Before Task 3's `compile_bridge` test, read `tests/test_synthesis.py` to copy the exact MockLLM code-block format `synthesize_transition` expects, and adapt the mock response if it differs from the plan's.

---

### Task 1: CompositeWorld core — namespacing, routing, tick, reset

**Files:**
- Create: `openworld/compose.py`
- Create: `tests/test_compose.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compose.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compose.py -v`
Expected: collection error — no module `openworld.compose`.

- [ ] **Step 3: Implement**

Create `openworld/compose.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compose.py -v` — all 6 PASS. Then `python -m pytest tests/ -q`.

- [ ] **Step 5: Commit**

```bash
git add openworld/compose.py tests/test_compose.py
git commit -m "Add CompositeWorld core: namespaced routing, tick with timescales, nesting-ready state"
```

---

### Task 2: Bindings, bridges, aggregators

**Files:**
- Modify: `tests/test_compose.py` (append)
- Modify: `openworld/compose.py` (only if a test exposes a defect — the Task 1 code already implements these paths)

- [ ] **Step 1: Write the failing-or-passing tests** (append to `tests/test_compose.py`)

```python
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
```

- [ ] **Step 2: Run the tests**

Run: `python -m pytest tests/test_compose.py -v`
Expected: all PASS against the Task 1 implementation. If any fail, the implementation has a defect — fix `openworld/compose.py` (do not weaken tests) and note what was wrong in the commit message.

- [ ] **Step 3: Commit**

```bash
git add tests/test_compose.py openworld/compose.py
git commit -m "compose: cover bindings, conservation bridges, bridge order, aggregators"
```

---

### Task 3: Nesting, compile_bridge, exports

**Files:**
- Modify: `openworld/compose.py` (append `compile_bridge`)
- Modify: `tests/test_compose.py` (append)
- Modify: `openworld/__init__.py` (exports)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_compose.py`; add `compile_bridge` to the compose import; add `from openworld import MockLLM`)

```python
def test_nested_composite_routes_two_levels_and_keeps_aggregates():
    inner = CompositeWorld(
        name="country",
        children={"city": make_city(rate=2)},
        aggregators=[Aggregator("gdp", lambda kids: kids["city"]["output"])],
    )
    outer = CompositeWorld(
        name="earth",
        children={"usa": inner},
        aggregators=[Aggregator("world_gdp", lambda kids: kids["usa"][AGG_KEY]["gdp"])],
    )
    s = outer.step(Action("usa:city:work"))
    assert s["usa"]["city"]["output"] == 2          # leaf stepped two levels down
    assert s["usa"][AGG_KEY]["gdp"] == 2            # inner aggregate recomputed
    assert s[AGG_KEY]["world_gdp"] == 2             # outer aggregate sees it
    assert "usa:city:work" in outer.actions          # namespacing composes


def test_compile_bridge_synthesizes_and_verifies_a_two_slot_transition():
    # MockLLM returns ready-made bridge code; the verifier must accept it and
    # the resulting Bridge must conserve the total. (Adapt the response format
    # to whatever tests/test_synthesis.py uses if it differs.)
    code = (
        "```python\n"
        "def transition(state, action):\n"
        "    s = {k: dict(v) if isinstance(v, dict) else v for k, v in state.items()}\n"
        "    if action.get('name') == 'flow' and s['a']['water'] > 0:\n"
        "        s['a']['water'] -= 1\n"
        "        s['b']['water'] += 1\n"
        "    return s\n"
        "```"
    )
    bridge = compile_bridge(
        MockLLM([code]),
        name="river",
        a="uphill",
        b="downhill",
        description="One unit of water flows downhill per tick while any remains.",
        rules=["water is conserved", "flow stops at zero"],
        sample_a={"water": 3},
        sample_b={"water": 0},
        invariants=[("water conserved",
                     lambda s: s["a"]["water"] + s["b"]["water"] == 3)],
    )
    sa, sb = bridge.flow({"water": 3}, {"water": 0})
    assert (sa["water"], sb["water"]) == (2, 1)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python -m pytest tests/test_compose.py -v`
Expected: nesting test PASSES already (composition is closed by construction); `compile_bridge` test fails with ImportError.

- [ ] **Step 3: Implement `compile_bridge`** (append to `openworld/compose.py`; add imports `from .llm import BaseLLM` and `from .verify import Verifier, synthesize_transition`)

```python
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
```

Before running: read `tests/test_synthesis.py` and `openworld/verify.py` to confirm (i) the fenced-code response format `synthesize_transition` extracts, and (ii) the action dict shape the generated `transition(state, action)` receives (`action.get('name')` vs an object). Adjust the TEST's mock code string to the real contract — not the library.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_compose.py -v` — all PASS.

- [ ] **Step 5: Export** — in `openworld/__init__.py`, add (alphabetical with the existing import block):

```python
from .compose import Aggregator, Binding, Bridge, CompositeWorld, compile_bridge
```

and append `"Aggregator", "Binding", "Bridge", "CompositeWorld", "compile_bridge"` to `__all__`.

- [ ] **Step 6: Full verification**

```bash
python -m pytest tests/ -q
python -c "from openworld import CompositeWorld, Bridge, Aggregator, Binding, compile_bridge; print('exports ok')"
```

Expected: full suite passes (145 pre-existing + ~12 new); exports import cleanly.

- [ ] **Step 7: Commit**

```bash
git add openworld/compose.py openworld/__init__.py tests/test_compose.py
git commit -m "compose: nesting test, compile_bridge synthesis path, package exports"
```
