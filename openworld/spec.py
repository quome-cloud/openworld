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


def _state_key(state: Dict[str, Any]) -> str:
    return json.dumps(dict(state), sort_keys=True, default=str)


def _transition_graph(world: World, max_nodes: int = 6) -> Dict[str, Any]:
    """The world's actual state-transition graph, by BFS from the initial state:
    nodes are distinct reachable states, edges are actions. Computed once at
    serialize time (the card renderer cannot execute downloaded code). Bounded to
    a handful of nodes so the picture stays legible. Leaf worlds only."""
    try:
        if world.transition is None or isinstance(world, CompositeWorld):
            return {}
        agent = _first_agent(world)
        start = dict(WorldState(world.initial_state).copy())
        k0 = _state_key(start)
        states: Dict[str, Dict[str, Any]] = {k0: start}
        order: List[str] = [k0]
        edges: List[Dict[str, Any]] = []
        queue = [k0]
        while queue and len(order) < max_nodes:
            k = queue.pop(0)
            idem: List[str] = []
            for a in world.actions:
                try:
                    nxt = dict(world.transition.step(
                        WorldState(states[k]).copy(), Action(a, agent=agent)))
                except Exception:
                    continue
                nk = _state_key(nxt)
                if nk == k:
                    idem.append(a)
                    continue
                if nk not in states:
                    if len(order) >= max_nodes:
                        continue
                    states[nk] = nxt
                    order.append(nk)
                    queue.append(nk)
                edges.append({"src": k, "dst": nk, "action": a})
            if idem:
                edges.append({"src": k, "dst": k, "action": ", ".join(idem[:3])})
        # choose up to 3 scalar vars that vary across nodes for compact labels
        scal = {k: _numeric_scalars(s) for k, s in states.items()}
        keys = sorted({kk for sc in scal.values() for kk in sc})
        varying = [kk for kk in keys
                   if len({scal[k].get(kk) for k in order}) > 1][:3] or keys[:3]
        idx = {k: i for i, k in enumerate(order)}
        nodes = []
        for k in order:
            label = [f"{kk} {scal[k][kk]:g}" for kk in varying if kk in scal[k]]
            nodes.append({"id": idx[k], "label": label or [f"s{idx[k]}"],
                          "initial": k == k0})
        graph_edges = [{"src": idx[e["src"]], "dst": idx[e["dst"]],
                        "action": e["action"]} for e in edges]
        return {"kind": "state", "nodes": nodes, "edges": graph_edges,
                "truncated": len(order) >= max_nodes}
    except Exception:
        return {}


def reachable_state_ids(world: World, max_nodes: int = 6) -> Dict[str, int]:
    """Map each reachable state-key to its node index in the transition graph
    (same BFS/ordering as the card). Lets the inference server highlight the
    'current' node as a leaf world is stepped. Empty for composites."""
    try:
        if world.transition is None or isinstance(world, CompositeWorld):
            return {}
        agent = _first_agent(world)
        start = dict(WorldState(world.initial_state).copy())
        k0 = _state_key(start)
        states, order, queue = {k0: start}, [k0], [k0]
        while queue and len(order) < max_nodes:
            k = queue.pop(0)
            for a in world.actions:
                try:
                    nxt = dict(world.transition.step(
                        WorldState(states[k]).copy(), Action(a, agent=agent)))
                except Exception:
                    continue
                nk = _state_key(nxt)
                if nk == k or nk in states:
                    continue
                if len(order) >= max_nodes:
                    continue
                states[nk] = nxt
                order.append(nk)
                queue.append(nk)
        return {k: i for i, k in enumerate(order)}
    except Exception:
        return {}


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
        return {"steps": steps, "action": act, "series": series,
                "graph": _transition_graph(world)}
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
# --------------------------------------------------------------------------- #
# perception (the perceive -> world boundary)
# --------------------------------------------------------------------------- #
def _schema_field_to_spec(v: Any) -> Any:
    """A perceptor schema value is a type, or a (type, (lo, hi)) range pair."""
    if isinstance(v, tuple) and v:
        typ = v[0]
        out: Dict[str, Any] = {"type": getattr(typ, "__name__", str(typ))}
        if len(v) > 1 and v[1] is not None:
            out["range"] = list(v[1])
        return out
    return getattr(v, "__name__", str(v))


_SCHEMA_TYPES = {"int": int, "float": float, "str": str, "bool": bool}


def _schema_field_from_spec(v: Any) -> Any:
    if isinstance(v, dict):
        typ = _SCHEMA_TYPES.get(v.get("type"))
        if typ is None:
            return None
        rng = v.get("range")
        return (typ, tuple(rng)) if rng is not None else typ
    return _SCHEMA_TYPES.get(v, v)


def _perceptor_to_spec(p: Any) -> Dict[str, Any]:
    d = {"kind": type(p).__name__,
         "modality": getattr(p, "modality", "text"),
         "produces": list(getattr(p, "produces", [])),
         "schema": {k: _schema_field_to_spec(v)
                    for k, v in getattr(p, "schema", {}).items()}}
    code = getattr(p, "code", None)
    if code is not None:                                   # CodePerceptor: runnable
        d["code"] = code
        d["func_name"] = getattr(p, "func_name", "perceive")
    return d


def _perceptor_from_spec(d: Dict[str, Any], allow_code: bool) -> Any:
    """Reconstruct a runnable perceptor when possible (CodePerceptor under
    allow_code). Descriptor-only perceptors (Mock/Text/Vision) cannot be rebuilt
    from a spec and return None."""
    if d.get("kind") == "CodePerceptor" and allow_code and d.get("code"):
        from .perceive import CodePerceptor
        schema = {k: _schema_field_from_spec(v) for k, v in d.get("schema", {}).items()}
        schema = {k: v for k, v in schema.items() if v is not None}
        return CodePerceptor(code=d["code"], produces=d.get("produces", []),
                             schema=schema, modality=d.get("modality", "text"),
                             func_name=d.get("func_name", "perceive"))
    return None


def _emit_to_spec(e: Any) -> Dict[str, Any]:
    """An emitter (the world -> output boundary): a modality + the state fields it
    reads out, and an optional `report` template/code for a human-readable
    artifact. Accepts a dict or an object with .modality/.fields(/.reads)."""
    if isinstance(e, dict):
        out = {"modality": e.get("modality", "data"),
               "fields": list(e.get("fields", []))}
        for k in ("report", "kind", "template"):          # incl. LLM-emit channel
            if e.get(k):
                out[k] = e[k]
        return out
    # an object emitter (e.g. LLMEmitter): a template-driven LLM text-out channel
    out = {"modality": getattr(e, "modality", "data"),
           "fields": list(getattr(e, "fields", None) or getattr(e, "reads", []))}
    if getattr(e, "template", None):
        out["kind"] = "llm"
        out["template"] = e.template
    if getattr(e, "report", None):
        out["report"] = e.report
    return out


def _objective_to_spec(o: Any) -> Dict[str, Any]:
    """A goal the world is evaluated against (name + direction + optional weight)."""
    if isinstance(o, dict):
        d = {"name": o.get("name"), "goal": o.get("goal") or o.get("direction"),
             "weight": o.get("weight")}
    else:
        d = {"name": getattr(o, "name", None),
             "goal": getattr(o, "direction", None) or getattr(o, "goal", None),
             "weight": getattr(o, "weight", None)}
    return {k: v for k, v in d.items() if v is not None}


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
            preview_steps: int = 12, perceptors: Optional[list] = None,
            emit: Optional[list] = None,
            objectives: Optional[list] = None) -> Dict[str, Any]:
    """Serialize a world model to a portable JSON-friendly spec dict.

    Perception (the perceive->world boundary) is captured when the world carries
    perceptors -- either passed as ``perceptors=`` or set on ``world.perceptors``.
    Each is described by its modality, the state fields it ``produces``, and its
    typed schema."""
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
    percs = perceptors if perceptors is not None else getattr(world, "perceptors", None)
    if percs:
        spec["perception"] = [_perceptor_to_spec(p) for p in percs]
    ems = emit if emit is not None else getattr(world, "emit", None)
    if ems:
        spec["emit"] = [_emit_to_spec(e) for e in ems]
    objs = objectives if objectives is not None else getattr(world, "objectives", None)
    if objs:
        spec["objectives"] = [_objective_to_spec(o) for o in objs]
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
        return _attach_io(_composite_from_spec(spec, allow_code, llm), spec, allow_code)
    transition = _transition_from_spec(spec.get("transition"), allow_code, llm)
    world = World(
        name=spec["name"],
        description=spec.get("description", ""),
        initial_state=spec.get("initial_state", {}),
        actions=list(spec.get("actions", [])),
        rules=list(spec.get("rules", [])),
        transition=transition,
        llm=llm,
    )
    return _attach_io(world, spec, allow_code)


def _attach_io(world: World, spec: Dict[str, Any], allow_code: bool) -> World:
    """Attach the reconstructable I/O components to a rebuilt world: runnable
    perceptors (CodePerceptor under allow_code), and the emit/objectives
    descriptors (used by the inference server)."""
    percs = [p for p in (_perceptor_from_spec(d, allow_code)
                         for d in spec.get("perception", [])) if p is not None]
    if percs:
        world.perceptors = percs
    if spec.get("emit"):
        world.emit = spec["emit"]
    if spec.get("objectives"):
        world.objectives = spec["objectives"]
    return world


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
    for fld in ("perception", "emit", "objectives"):
        if fld in spec and not isinstance(spec[fld], list):
            problems.append(f"{fld!r} must be a list")
    card = spec.get("card")
    if isinstance(card, dict) and "metrics" in card and not isinstance(card["metrics"], dict):
        problems.append("card.metrics must be an object")
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


def _mm_id(prefix: str, key: Any) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(key))
    return f"{prefix}_{safe}"


def to_mermaid(spec: Dict[str, Any]) -> str:
    """Render the world's graph as Mermaid flowchart text (GitHub/markdown render
    it natively): a composition dataflow for composites, the state-transition
    automaton for leaves."""
    comp = spec.get("composite")
    lines: List[str]
    if comp:
        lines = ["flowchart TD"]
        for ns in comp.get("children", {}):
            lines.append(f'  {_mm_id("c", ns)}["{ns}"]')
        agg_ids = set()
        for a in comp.get("aggregators", []):
            nm = a.get("name", "agg")
            aid = _mm_id("a", nm)
            agg_ids.add(nm)
            lines.append(f'  {aid}(["Σ {nm}"])')
            for ns in comp.get("children", {}):
                lines.append(f'  {_mm_id("c", ns)} --> {aid}')
        for b in comp.get("bridges", []):
            arrow = "-.->" if b.get("kind") == "route" else "-->"
            label = b.get("name", "")
            mid = f"|{label}|" if label else ""
            lines.append(f'  {_mm_id("c", b.get("a"))} {arrow}{mid} {_mm_id("c", b.get("b"))}')
        for bd in comp.get("bindings", []):
            sp = bd.get("source_path", [])
            src = (_mm_id("a", sp[1]) if len(sp) >= 2 and sp[0] == "_agg"
                   and sp[1] in agg_ids else _mm_id("c", bd.get("child")))
            lines.append(f'  {src} -.{bd.get("key", "")}.-> {_mm_id("c", bd.get("child"))}')
        return "\n".join(lines)

    g = (spec.get("preview", {}) or {}).get("graph", {}) or {}
    lines = ["flowchart LR"]
    gnodes = g.get("nodes", [])
    init_id = next((n["id"] for n in gnodes if n.get("initial")),
                   gnodes[0]["id"] if gnodes else None)
    # perception boundary: sensors -> initial state
    for i, p in enumerate(spec.get("perception", [])):
        lab = p.get("modality", "text") + ": " + ", ".join(p.get("produces", []))
        lines.append(f'  perc{i}[/"{lab}"/]')
        if init_id is not None:
            lines.append(f'  perc{i} -. perceive .-> n{init_id}')
    for n in gnodes:
        label = "<br/>".join(n.get("label", [])) or f's{n["id"]}'
        if n.get("initial"):
            lines.append(f'  n{n["id"]}["{label}"]:::start')
        else:
            lines.append(f'  n{n["id"]}["{label}"]')
    for e in g.get("edges", []):
        act = e.get("action", "")
        mid = f"|{act}|" if act else ""
        lines.append(f'  n{e["src"]} -->{mid} n{e["dst"]}')
    # emit boundary: initial state -> outputs
    for i, em in enumerate(spec.get("emit", [])):
        lab = em.get("modality", "data") + ": " + ", ".join(em.get("fields", []))
        lines.append(f'  emit{i}[\\"{lab}"/]')
        if init_id is not None:
            lines.append(f'  n{init_id} -. emit .-> emit{i}')
    if any(n.get("initial") for n in gnodes):
        lines.append("  classDef start stroke-width:2px;")
    return "\n".join(lines)


def spec_to_json(spec: Dict[str, Any], indent: Optional[int] = 2) -> str:
    return json.dumps(spec, indent=indent, sort_keys=False, default=str)


def spec_from_json(text: str) -> Dict[str, Any]:
    return json.loads(text)
