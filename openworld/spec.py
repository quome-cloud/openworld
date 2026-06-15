"""Portable JSON specs for world models: serialize, validate, reconstruct.

A *spec* is a plain JSON-serializable dict that captures everything needed to
describe and (with code execution opted in) reconstruct a runnable world model:
its name, description, state schema, concrete initial state, actions, rules, and
dynamics --- recursively, for composites. Specs are the publishable unit of a
"marketplace for world models": one file per world, renderable to a model card
(see ``openworld.card``).

Round-trip is lossless for the common cases: ``from_spec(to_spec(w),
allow_code=True)`` reproduces ``w``'s behavior. Dynamics that are genuinely
un-serializable (a ``FunctionTransition`` whose source is unavailable, a callable
phase trigger, a lambda aggregator) are flagged ``lossy`` rather than silently
dropped.

Safety: ``from_spec`` does NOT compile any embedded code unless ``allow_code=True``
--- a downloaded spec loads fully described (schema, metadata, preview intact) but
its transition raises :class:`SpecError` if stepped, until you opt in. This is a
trust gate, not a sandbox against adversarial code.

Zero-dependency: standard library only.
"""

from __future__ import annotations

import copy
import inspect
import json
import textwrap
from typing import Any, Dict, List, Optional

from .compose import (AGG_KEY, AGENTS_KEY, Aggregator, Binding, Bridge,
                      CompositeTransition, CompositeWorld, Route)
from .sandbox import SandboxError, load_transition_code
from .state import Action, WorldState
from .transition import (CodeTransition, FunctionTransition, LLMTransition,
                         PhasedTransition, Transition)
from .world import World

SPEC_VERSION = "1.0"
_TRANSITION_KINDS = {"code", "function", "phased", "llm", "composite", "none"}


class SpecError(Exception):
    """Raised on malformed specs or when inert (non-executable) dynamics run."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    """Recursively coerce to JSON-friendly types (tuples -> lists; fall back to
    str for anything exotic). State is symbolic, so this is normally a no-op."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _infer_schema(value: Any) -> str:
    """A compact type name for one state value (bool before int on purpose)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return f"list[{_infer_schema(value[0])}]" if value else "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _try_getsource(obj: Any) -> Optional[str]:
    try:
        return textwrap.dedent(inspect.getsource(obj))
    except (OSError, TypeError):
        return None


def _numeric_scalars(state: Dict[str, Any]) -> Dict[str, float]:
    """Top-level numeric state vars plus numeric aggregator outputs (for previews)."""
    out: Dict[str, float] = {}
    for k, v in state.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
        elif k == AGG_KEY and isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    out[kk] = float(vv)
    return out


def _first_agent(world: World) -> Optional[str]:
    agents = getattr(world, "agents", None)
    return next(iter(agents)) if agents else None


def _pick_preview_action(world: World) -> str:
    """The single action that moves the most numeric state in one step from the
    initial state --- so the card sparkline is lively, deterministically."""
    candidates = list(world.actions) or ["noop"]
    agent = _first_agent(world)
    base = WorldState(world.initial_state).copy()
    base_n = _numeric_scalars(base)
    best, best_delta = candidates[0], -1.0
    for a in candidates:
        try:
            nxt = world.transition.step(base.copy(), Action(a, agent=agent))
            n2 = _numeric_scalars(dict(nxt))
            delta = sum(abs(n2.get(k, 0.0) - base_n.get(k, 0.0))
                        for k in set(base_n) | set(n2))
        except Exception:
            delta = -1.0
        if delta > best_delta:
            best_delta, best = delta, a
    return best


def _rollout_preview(world: World, steps: int) -> Dict[str, Any]:
    """Roll the live world forward and record numeric series for the card.
    Best-effort: any failure yields an empty preview (never blocks to_spec)."""
    try:
        if world.transition is None or steps <= 0:
            return {}
        act = _pick_preview_action(world)
        agent = _first_agent(world)
        s = WorldState(world.initial_state).copy()
        series: Dict[str, List[float]] = {}

        def record(state: Dict[str, Any]) -> None:
            for k, v in _numeric_scalars(state).items():
                series.setdefault(k, []).append(round(v, 4))

        record(s)
        for _ in range(steps):
            s = world.transition.step(s, Action(act, agent=agent))
            record(dict(s))
        series = {k: v for k, v in series.items() if len(set(v)) > 1}
        return {"steps": steps, "action": act, "series": series}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# transitions
# --------------------------------------------------------------------------- #
def _transition_to_spec(t: Optional[Transition]) -> Dict[str, Any]:
    if t is None:
        return {"kind": "none"}
    if isinstance(t, CompositeTransition):
        return {"kind": "composite"}
    if isinstance(t, CodeTransition):
        return {"kind": "code", "func_name": t.func_name, "code": t.code}
    if isinstance(t, PhasedTransition):
        return {"kind": "phased", "record_key": t.record_key,
                "phases": [_phase_to_spec(p) for p in t.phases]}
    if isinstance(t, LLMTransition):
        return {"kind": "llm", "description": t.description, "rules": list(t.rules)}
    if isinstance(t, FunctionTransition):
        src = _try_getsource(t.fn)
        name = getattr(t.fn, "__name__", "")
        if src and name and name != "<lambda>":
            return {"kind": "code", "func_name": name, "code": src,
                    "from_function": True}
        return {"kind": "function", "lossy": True, "repr": repr(t.fn)}
    src = _try_getsource(type(t))
    return {"kind": "function", "lossy": True, "repr": repr(t),
            **({"class_source": src} if src else {})}


def _phase_to_spec(phase: Any) -> Dict[str, Any]:
    trigger, transition = phase
    out: Dict[str, Any] = {"transition": _transition_to_spec(transition)}
    if isinstance(trigger, int) and not isinstance(trigger, bool):
        out["trigger"] = trigger
    else:
        out["trigger"] = 0
        out["trigger_lossy"] = True
    return out


def _transition_from_spec(d: Optional[Dict[str, Any]], allow_code: bool,
                          llm: Any) -> Optional[Transition]:
    if not d or d.get("kind") in (None, "none"):
        return None
    kind = d["kind"]
    if kind == "composite":
        return None  # rebuilt by the composite path
    if kind == "code":
        if allow_code:
            return CodeTransition(d["code"], func_name=d.get("func_name", "transition"))
        return _InertTransition("code transition not loaded (allow_code=False)")
    if kind == "function":
        return _InertTransition("function transition is not serializable (lossy)")
    if kind == "phased":
        if not allow_code:
            return _InertTransition("phased transition not loaded (allow_code=False)")
        phases = [(p.get("trigger", 0),
                   _transition_from_spec(p["transition"], allow_code, llm))
                  for p in d["phases"]]
        return PhasedTransition(phases, record_key=d.get("record_key", "_phase"))
    if kind == "llm":
        if llm is None:
            return _InertTransition("llm transition needs an llm= to load")
        return LLMTransition(llm, description=d.get("description", ""),
                             rules=d.get("rules", []))
    return _InertTransition(f"unknown transition kind {kind!r}")


class _InertTransition(Transition):
    """A placeholder for dynamics not loaded for safety. Describes, never runs."""

    def __init__(self, reason: str):
        self.reason = reason

    def step(self, state: WorldState, action: Action) -> WorldState:
        raise SpecError(
            f"world loaded without executable dynamics: {self.reason}; "
            "reload with from_spec(..., allow_code=True) to run it")


# --------------------------------------------------------------------------- #
# composite pieces
# --------------------------------------------------------------------------- #
def _bridge_to_spec(b: Bridge) -> Dict[str, Any]:
    d = {"name": b.name, "a": b.a, "b": b.b, "description": b.description,
         "rules": list(b.rules), "transition": _transition_to_spec(b.transition)}
    if isinstance(b, Route):
        d["kind"] = "route"
        d["on_cross"] = _transition_to_spec(b.on_cross)
    else:
        d["kind"] = "bridge"
    return d


def _bridge_from_spec(d: Dict[str, Any], allow_code: bool, llm: Any) -> Bridge:
    trans = _transition_from_spec(d.get("transition"), allow_code, llm)
    common = dict(name=d["name"], a=d["a"], b=d["b"], transition=trans,
                  description=d.get("description", ""), rules=tuple(d.get("rules", ())))
    if d.get("kind") == "route":
        return Route(on_cross=_transition_from_spec(d.get("on_cross"), allow_code, llm),
                     **common)
    return Bridge(**common)


def _aggregator_to_spec(a: Aggregator) -> Dict[str, Any]:
    src = _try_getsource(a.fn)
    name = getattr(a.fn, "__name__", "")
    if src and name and name != "<lambda>":
        return {"name": a.name, "func_name": name, "source": src}
    return {"name": a.name, "lossy": True, "repr": repr(a.fn)}


def _stub_agg(_children: Dict[str, dict]) -> None:
    return None


def _aggregator_from_spec(d: Dict[str, Any], allow_code: bool) -> Aggregator:
    src, name = d.get("source"), d.get("name")
    if allow_code and src and not d.get("lossy"):
        try:
            fn = load_transition_code(src, d.get("func_name", name))
            return Aggregator(name=name, fn=fn)
        except SandboxError:
            pass
    return Aggregator(name=name, fn=_stub_agg)


def _composite_to_spec(world: CompositeWorld, preview_steps: int) -> Dict[str, Any]:
    return {
        "children": {ns: to_spec(child, preview_steps=preview_steps)
                     for ns, child in world.children.items()},
        "bridges": [_bridge_to_spec(b) for b in world.bridges],
        "aggregators": [_aggregator_to_spec(a) for a in world.aggregators],
        "bindings": [{"source_path": list(b.source_path), "child": b.child,
                      "key": b.key} for b in world.bindings],
        "timescales": dict(world.timescales),
        "default_actions": dict(world.default_actions),
        "agents": {k: dict(v) for k, v in world.agents.items()},
    }


def _composite_from_spec(spec: Dict[str, Any], allow_code: bool, llm: Any) -> CompositeWorld:
    c = spec["composite"]
    children = {ns: from_spec(child, allow_code=allow_code, llm=llm)
                for ns, child in c["children"].items()}
    return CompositeWorld(
        name=spec["name"],
        children=children,
        bridges=[_bridge_from_spec(b, allow_code, llm) for b in c.get("bridges", [])],
        aggregators=[_aggregator_from_spec(a, allow_code) for a in c.get("aggregators", [])],
        bindings=[Binding(tuple(b["source_path"]), b["child"], b["key"])
                  for b in c.get("bindings", [])],
        timescales=c.get("timescales") or None,
        default_actions=c.get("default_actions") or None,
        description=spec.get("description", ""),
        rules=spec.get("rules") or None,
        agents=c.get("agents") or None,
    )


# --------------------------------------------------------------------------- #
# card metadata
# --------------------------------------------------------------------------- #
def _default_card(world: World, card: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = {"version": "0.1", "license": None, "authors": [], "tags": [],
            "lineage": None, "metrics": {}, "description": world.description}
    if card:
        base.update(card)
    return base


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def to_spec(world: World, *, card: Optional[Dict[str, Any]] = None,
            preview_steps: int = 12) -> Dict[str, Any]:
    """Serialize a world model to a portable JSON-friendly spec dict."""
    spec: Dict[str, Any] = {
        "openworld_spec_version": SPEC_VERSION,
        "name": world.name,
        "description": world.description,
        "card": _default_card(world, card),
        "state_schema": {k: _infer_schema(v)
                         for k, v in dict(world.initial_state).items()},
        "initial_state": _jsonable(dict(world.initial_state)),
        "actions": list(world.actions),
        "rules": list(world.rules),
        "transition": _transition_to_spec(world.transition),
    }
    if isinstance(world, CompositeWorld):
        spec["composite"] = _composite_to_spec(world, preview_steps)
    spec["preview"] = _rollout_preview(world, preview_steps)
    return spec


def from_spec(spec: Dict[str, Any], *, allow_code: bool = False,
              llm: Any = None) -> World:
    """Reconstruct a world from a spec. Embedded code is inert unless
    ``allow_code=True`` (the trust gate for downloaded specs)."""
    if not isinstance(spec, dict) or "name" not in spec:
        raise SpecError("spec must be a dict with at least a 'name'")
    if spec.get("composite"):
        return _composite_from_spec(spec, allow_code, llm)
    transition = _transition_from_spec(spec.get("transition"), allow_code, llm)
    return World(
        name=spec["name"],
        description=spec.get("description", ""),
        initial_state=spec.get("initial_state", {}),
        actions=list(spec.get("actions", [])),
        rules=list(spec.get("rules", [])),
        transition=transition,
        llm=llm,
    )


def validate_spec(spec: Any) -> List[str]:
    """Return a list of human-readable problems with a spec (empty == valid).
    The marketplace publish gate."""
    problems: List[str] = []
    if not isinstance(spec, dict):
        return ["spec is not a JSON object"]
    if spec.get("openworld_spec_version") != SPEC_VERSION:
        problems.append(
            f"openworld_spec_version must be {SPEC_VERSION!r}, "
            f"got {spec.get('openworld_spec_version')!r}")
    for key in ("name", "actions", "transition"):
        if key not in spec:
            problems.append(f"missing required field {key!r}")
    if not isinstance(spec.get("name", ""), str) or not spec.get("name"):
        problems.append("'name' must be a non-empty string")
    if "actions" in spec and not isinstance(spec["actions"], list):
        problems.append("'actions' must be a list")
    schema, init = spec.get("state_schema"), spec.get("initial_state")
    if isinstance(schema, dict) and isinstance(init, dict):
        extra = set(schema) - set(init)
        if extra:
            problems.append(f"state_schema keys not in initial_state: {sorted(extra)}")
    t = spec.get("transition")
    if isinstance(t, dict):
        if t.get("kind") not in _TRANSITION_KINDS:
            problems.append(f"transition.kind {t.get('kind')!r} not in {sorted(_TRANSITION_KINDS)}")
    elif "transition" in spec:
        problems.append("'transition' must be an object with a 'kind'")
    comp = spec.get("composite")
    if comp is not None:
        if not isinstance(comp, dict) or "children" not in comp:
            problems.append("'composite' must be an object with 'children'")
        else:
            for ns, child in comp["children"].items():
                problems.extend(f"child {ns!r}: {p}" for p in validate_spec(child))
            for b in comp.get("bridges", []):
                for need in ("name", "a", "b"):
                    if need not in b:
                        problems.append(f"bridge missing {need!r}")
    return problems


def spec_to_json(spec: Dict[str, Any], indent: Optional[int] = 2) -> str:
    return json.dumps(spec, indent=indent, sort_keys=False, default=str)


def spec_from_json(text: str) -> Dict[str, Any]:
    return json.loads(text)
