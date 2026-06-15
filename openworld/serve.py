"""FastAPI inference server for world models — the *deploy* phase.

Loads world models from specs and serves a stateless "forward pass": given a
state and an action, return the next state. Like serving a trained model, but the
artifact is a verified symbolic world. Handles composites (namespaced actions,
``tick``, ``travel``), bridges (applied inside the composite transition), and
perception (run a perceptor on raw input, gated, then step). A browser view
renders the world's graph and animates it as you step.

This module imports FastAPI/uvicorn, which are NOT part of the zero-dependency
core: ``import openworld`` never imports this. Install with the serving extras.

Security: running dynamics executes the spec's code, so a runnable registry needs
``allow_code=True``. That is a trust gate for local, trusted specs — not a sandbox
against adversarial code.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, Response
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "openworld.serve needs FastAPI/uvicorn. Install the serving deps: "
        "pip install fastapi uvicorn") from exc

from .card import render_card, to_reactflow
from .perceive import Observation, PerceptionGate
from .spec import (from_spec, reachable_state_ids, spec_to_json, to_mermaid,
                   validate_spec)
from .state import Action, WorldState

_GATE = PerceptionGate()


# --------------------------------------------------------------------------- #
# request models (pydantic v1 compatible)
# --------------------------------------------------------------------------- #
class ActionIn(BaseModel):
    name: str
    params: Dict[str, Any] = {}
    agent: Optional[str] = None


class StepIn(BaseModel):
    state: Dict[str, Any]
    action: ActionIn


class PredictIn(BaseModel):
    inputs: List[StepIn]


class RolloutIn(BaseModel):
    state: Optional[Dict[str, Any]] = None
    actions: List[ActionIn] = []


class ObserveIn(BaseModel):
    state: Optional[Dict[str, Any]] = None
    observation: Optional[Dict[str, Any]] = None      # {modality, data}
    delta: Dict[str, Any] = {}                         # pre-extracted, gated+merged
    step: bool = False


class RunIn(BaseModel):
    input: Dict[str, Any]                              # {modality, data}
    state: Optional[Dict[str, Any]] = None
    steps: int = 8


class LoadIn(BaseModel):
    spec: Optional[Dict[str, Any]] = None
    path: Optional[str] = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _changed(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def _depth(spec: Dict[str, Any]) -> int:
    comp = spec.get("composite")
    if not comp:
        return 0
    return 1 + max((_depth(c) for c in comp.get("children", {}).values()), default=0)


def _metrics(spec: Dict[str, Any]) -> Dict[str, Any]:
    schema = {k: v for k, v in spec.get("state_schema", {}).items()
              if not k.startswith("_")}
    graph = (spec.get("preview", {}) or {}).get("graph", {}) or {}
    m: Dict[str, Any] = {
        "state_fields": len(schema),
        "actions": len(spec.get("actions", [])),
        "dynamics": (spec.get("transition", {}) or {}).get("kind"),
        "reachable_states": len(graph.get("nodes", [])) or None,
        "perception_channels": len(spec.get("perception", [])),
        "emit_channels": len(spec.get("emit", [])),
        "objectives": len(spec.get("objectives", [])),
        "spec_bytes": len(spec_to_json(spec)),
    }
    comp = spec.get("composite")
    if comp:
        m.update({"children": len(comp.get("children", {})),
                  "bridges": len(comp.get("bridges", [])),
                  "aggregators": len(comp.get("aggregators", [])),
                  "bindings": len(comp.get("bindings", [])),
                  "agents": len(comp.get("agents", {})),
                  "depth": _depth(spec)})
    m.update((spec.get("card", {}) or {}).get("metrics", {}))
    return m


def _roll_action(spec: Dict[str, Any]) -> str:
    actions = spec.get("actions", [])
    if "tick" in actions:
        return "tick"
    for a in actions:
        if a not in ("noop", "wait"):
            return a
    return actions[0] if actions else "noop"


def _emit_output(spec: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Build the output artifact from the world's emit channels + final state."""
    outs = []
    for e in spec.get("emit", []):
        fields = {f: state.get(f) for f in e.get("fields", [])}
        item = {"modality": e.get("modality", "data"), "fields": fields}
        report = e.get("report")
        if report:
            try:
                item["report"] = report.format(**state)
            except Exception:
                item["report"] = report
        outs.append(item)
    return {"emitted": outs}


def load_specs(source) -> List[Dict[str, Any]]:
    """Load specs from a directory, a list of paths, or a list of spec dicts."""
    if isinstance(source, dict):
        return [source]
    if isinstance(source, (str, Path)):
        p = Path(source)
        paths = sorted(p.glob("*.json")) if p.is_dir() else [p]
        return [json.loads(Path(x).read_text(encoding="utf-8")) for x in paths]
    specs: List[Dict[str, Any]] = []
    for item in source:
        if isinstance(item, dict):
            specs.append(item)
        else:
            specs.append(json.loads(Path(item).read_text(encoding="utf-8")))
    return specs


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
class WorldRegistry:
    def __init__(self, allow_code: bool = False):
        self.allow_code = allow_code
        self._specs: Dict[str, Dict[str, Any]] = {}
        self._worlds: Dict[str, Any] = {}
        self._index: Dict[str, Dict[str, int]] = {}

    def add(self, spec: Dict[str, Any]) -> str:
        name = spec["name"]
        self._specs[name] = spec
        world = from_spec(spec, allow_code=self.allow_code)
        self._worlds[name] = world
        self._index[name] = reachable_state_ids(world) if self.allow_code else {}
        return name

    def __contains__(self, name):
        return name in self._specs

    def names(self):
        return list(self._specs)

    def spec(self, name):
        return self._specs[name]

    def world(self, name):
        return self._worlds[name]

    def current_node(self, name, state):
        from .spec import _state_key
        idx = self._index.get(name, {}).get(_state_key(state))
        return f"n{idx}" if idx is not None else None


# --------------------------------------------------------------------------- #
# app factory
# --------------------------------------------------------------------------- #
def serve_app(specs, allow_code: bool = False,
              title: str = "OpenWorld inference server") -> "FastAPI":
    registry = WorldRegistry(allow_code=allow_code)
    for spec in load_specs(specs):
        registry.add(spec)

    app = FastAPI(title=title)
    app.state.registry = registry

    def _spec(name):
        if name not in registry:
            raise HTTPException(404, f"no world named {name!r}")
        return registry.spec(name)

    def _runnable(name):
        if not allow_code:
            raise HTTPException(
                403, "world dynamics are not loaded (server started without "
                "--allow-code); only metadata is served")
        return registry.world(name)

    def _step(name, state, action: ActionIn):
        world = _runnable(name)
        nxt = world.transition.step(
            WorldState(state), Action(action.name, params=action.params,
                                      agent=action.agent))
        nxt = dict(nxt)
        return {"next_state": nxt, "changed": _changed(state, nxt),
                "current_node": registry.current_node(name, nxt)}

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "worlds": len(registry.names()), "runnable": allow_code}

    @app.get("/worlds")
    def list_worlds():
        out = []
        for n in registry.names():
            s = registry.spec(n)
            comp = s.get("composite")
            out.append({"name": n, "kind": "composite" if comp else "leaf",
                        "description": s.get("description", ""),
                        "actions": len(s.get("actions", [])),
                        "bridges": len(comp.get("bridges", [])) if comp else 0})
        return {"worlds": out, "runnable": allow_code}

    @app.get("/worlds/{name}")
    def world_info(name: str):
        s = _spec(name)
        comp = s.get("composite") or {}
        return {
            "name": name, "kind": "composite" if comp else "leaf",
            "description": s.get("description", ""),
            "state_schema": s.get("state_schema", {}),
            "initial_state": s.get("initial_state", {}),
            "actions": s.get("actions", []),
            "bridges": [{"name": b.get("name"), "a": b.get("a"), "b": b.get("b"),
                         "kind": b.get("kind", "bridge")} for b in comp.get("bridges", [])],
            "aggregators": [a.get("name") for a in comp.get("aggregators", [])],
            "perception": s.get("perception", []),
            "emit": s.get("emit", []),
            "objectives": s.get("objectives", []),
            "runnable": allow_code,
        }

    @app.get("/worlds/{name}/spec")
    def world_spec(name: str):
        return _spec(name)

    @app.get("/worlds/{name}/state")
    def world_state(name: str):
        return _spec(name).get("initial_state", {})

    @app.get("/worlds/{name}/actions")
    def world_actions(name: str):
        s = _spec(name)
        comp = s.get("composite") or {}
        return {"actions": s.get("actions", []),
                "default": _roll_action(s),
                "bridges": [b.get("name") for b in comp.get("bridges", [])]}

    @app.get("/worlds/{name}/metrics")
    def world_metrics(name: str):
        return _metrics(_spec(name))

    @app.get("/worlds/{name}/card.svg")
    def world_card(name: str):
        return Response(render_card(_spec(name)), media_type="image/svg+xml")

    @app.get("/worlds/{name}/mermaid")
    def world_mermaid(name: str):
        return Response(to_mermaid(_spec(name)), media_type="text/plain")

    @app.get("/worlds/{name}/reactflow")
    def world_reactflow(name: str):
        return to_reactflow(_spec(name))

    @app.post("/worlds/{name}/step")
    def step(name: str, body: StepIn):
        _spec(name)
        return _step(name, body.state, body.action)

    @app.post("/worlds/{name}/predict")
    def predict(name: str, body: PredictIn):
        _spec(name)
        return {"outputs": [_step(name, i.state, i.action) for i in body.inputs]}

    @app.post("/worlds/{name}/rollout")
    def rollout(name: str, body: RolloutIn):
        s = _spec(name)
        world = _runnable(name)
        state = dict(body.state if body.state is not None else s.get("initial_state", {}))
        acts = body.actions or [ActionIn(name=_roll_action(s))]
        traj = []
        for a in acts:
            state = dict(world.transition.step(
                WorldState(state), Action(a.name, params=a.params, agent=a.agent)))
            traj.append({"action": a.name, "state": state,
                         "current_node": registry.current_node(name, state)})
        return {"trajectory": traj, "final_state": state}

    @app.post("/worlds/{name}/observe")
    def observe(name: str, body: ObserveIn):
        s = _spec(name)
        world = _runnable(name)
        state = dict(body.state if body.state is not None else s.get("initial_state", {}))
        delta = dict(body.delta)
        perceptors = getattr(world, "perceptors", None) or []
        if body.observation is not None and perceptors:
            p = perceptors[0]
            obs = Observation(modality=body.observation.get("modality", "text"),
                              data=body.observation.get("data"))
            try:
                delta = _GATE.check(p, p.perceive(obs))
            except Exception as e:
                raise HTTPException(422, f"perception failed the gate: {e}")
        elif delta and perceptors:
            try:
                delta = _GATE.check(perceptors[0], delta)
            except Exception as e:
                raise HTTPException(422, f"delta failed the perception gate: {e}")
        state.update(delta)
        out = {"perceived_delta": delta, "state": state}
        if body.step:
            nxt = dict(world.transition.step(
                WorldState(state), Action(_roll_action(s))))
            out["next_state"] = nxt
            out["current_node"] = registry.current_node(name, nxt)
        return out

    @app.post("/worlds/{name}/run")
    def run(name: str, body: RunIn):
        s = _spec(name)
        world = _runnable(name)
        perceptors = getattr(world, "perceptors", None) or []
        if not perceptors:
            raise HTTPException(400, f"{name} has no runnable perceptor to take input")
        state = dict(body.state if body.state is not None else s.get("initial_state", {}))
        p = perceptors[0]
        obs = Observation(modality=body.input.get("modality", "text"),
                          data=body.input.get("data"))
        try:
            delta = _GATE.check(p, p.perceive(obs))
        except Exception as e:
            raise HTTPException(422, f"perception failed the gate: {e}")
        state.update(delta)
        act = _roll_action(s)
        traj = [{"action": "perceive", "state": dict(state),
                 "current_node": registry.current_node(name, state)}]
        for _ in range(max(0, body.steps)):
            state = dict(world.transition.step(WorldState(state), Action(act)))
            traj.append({"action": act, "state": dict(state),
                         "current_node": registry.current_node(name, state)})
        return {"perceived_delta": delta, "trajectory": traj,
                "output": _emit_output(s, state)}

    @app.post("/worlds")
    def load_world(body: LoadIn):
        if not allow_code:
            raise HTTPException(403, "hot-loading requires the server to run with "
                                "--allow-code")
        spec = body.spec
        if spec is None and body.path:
            spec = json.loads(Path(body.path).read_text(encoding="utf-8"))
        if spec is None:
            raise HTTPException(422, "provide a 'spec' object or a 'path'")
        problems = validate_spec(spec)
        if problems:
            raise HTTPException(422, {"validation_errors": problems})
        return {"loaded": registry.add(spec)}

    @app.websocket("/worlds/{name}/live")
    async def live(ws: WebSocket, name: str):
        await ws.accept()
        try:
            while True:
                msg = await ws.receive_json()
                if name not in registry or not allow_code:
                    await ws.send_json({"error": "world not runnable"})
                    continue
                s = registry.spec(name)
                if "input" in msg:                         # input mode: stream a run
                    res = run(name, RunIn(**{"input": msg["input"],
                                             "state": msg.get("state"),
                                             "steps": msg.get("steps", 8)}))
                    for fr in res["trajectory"]:
                        await ws.send_json({"frame": fr})
                    await ws.send_json({"done": True, "output": res["output"]})
                else:                                      # action mode: one step
                    state = msg.get("state") or s.get("initial_state", {})
                    act = msg.get("action") or {"name": _roll_action(s)}
                    await ws.send_json(_step(name, state, ActionIn(**act)))
        except WebSocketDisconnect:
            return

    @app.get("/", response_class=HTMLResponse)
    def index():
        badge = ('<span class="badge on">runnable</span>' if allow_code
                 else '<span class="badge off">read-only</span>')
        tiles = []
        for n in registry.names():
            s = registry.spec(n)
            kind = "composite" if s.get("composite") else "leaf"
            tags = "".join(f'<span class="tag">{html.escape(str(t))}</span>'
                           for t in (s.get("card", {}) or {}).get("tags", [])[:3])
            tiles.append(
                f'<div class="tile" onclick="location.href=\'/worlds/{n}/view\'">'
                f'<div class="thumb"><img src="/worlds/{n}/card.svg" loading="lazy" '
                f'alt="{html.escape(n)} card"></div>'
                f'<div class="body"><div><span class="nm">{html.escape(n)}</span>'
                f'<span class="kind">{kind}</span></div>'
                f'<div class="tags">{tags}</div>'
                f'<div class="links">'
                f'<a class="cta" href="/worlds/{n}/view" onclick="event.stopPropagation()">▸ Open in React Flow</a>'
                f'<a href="/worlds/{n}/card.svg" onclick="event.stopPropagation()">card</a>'
                f'<a href="/worlds/{n}/spec" onclick="event.stopPropagation()">spec</a>'
                f'<a href="/worlds/{n}/metrics" onclick="event.stopPropagation()">metrics</a>'
                f'</div></div></div>')
        return (
            f'<!doctype html><html><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{_INDEX_CSS}</style></head><body>'
            f'<div class="wrap"><header class="top"><div class="brand">{_MARK_SVG}'
            f'<div><div class="kick">OPENWORLD · INFERENCE SERVER</div>'
            f'<div class="wm">OpenWorld</div></div></div>'
            f'<div class="meta">{len(registry.names())} worlds {badge}'
            f'<a href="/docs">API docs</a></div></header>'
            f'<div class="rule"></div>'
            f'<div class="grid">{"".join(tiles)}</div>'
            f'<footer>build · optimize · deploy — verified symbolic world models</footer>'
            f'</div></body></html>')

    @app.get("/worlds/{name}/view", response_class=HTMLResponse)
    def view(name: str):
        _spec(name)
        return _VIEW_HTML.replace("__NAME__", name).replace("__TITLE__", title)

    return app


# Brand mark + index stylesheet (the "atlas" look, matching the SVG cards).
_MARK_SVG = (
    '<svg width="34" height="34" viewBox="-40 -40 80 80" aria-hidden="true">'
    '<rect x="-34" y="-34" width="68" height="68" rx="13" fill="#fff" '
    'stroke="#1d4ed8" stroke-width="4.5"/>'
    '<rect x="-20" y="-20" width="40" height="40" rx="8" fill="none" '
    'stroke="#b45309" stroke-width="3.6"/>'
    '<rect x="-8" y="-8" width="16" height="16" rx="4" fill="#0f766e"/></svg>')

_INDEX_CSS = """
:root{--bg:#fcfbf8;--bg2:#eef0ec;--ink:#16202e;--muted:#5b6675;--accent:#1d4ed8;
--accent2:#b45309;--teal:#0f766e;--line:#dde2ea;--card:#fff;--mono:ui-monospace,
SFMono-Regular,Menlo,Consolas,monospace;--serif:'Iowan Old Style',Palatino,Georgia,serif}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;color:var(--ink);
background:linear-gradient(180deg,var(--bg),var(--bg2));
font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1160px;margin:0 auto;padding:34px 22px 60px}
header.top{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:12px}
.brand .wm{font-family:var(--serif);font-size:30px;font-weight:800;letter-spacing:-.5px;line-height:1}
.kick{font-family:var(--mono);font-size:10.5px;letter-spacing:2.2px;color:var(--accent);font-weight:700}
.meta{margin-left:auto;display:flex;gap:14px;align-items:center;font-size:13px;
color:var(--muted);font-family:var(--mono)}
.meta a{color:var(--accent);text-decoration:none}.meta a:hover{text-decoration:underline}
.badge{padding:3px 10px;border-radius:999px;font-weight:700;font-size:12px;font-family:var(--mono)}
.badge.on{background:#e7f6f1;color:var(--teal)}.badge.off{background:#fdeee0;color:var(--accent2)}
.rule{height:4px;border-radius:2px;background:linear-gradient(90deg,var(--accent),#0891b2);margin:14px 0 26px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:22px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:16px;overflow:hidden;
cursor:pointer;display:flex;flex-direction:column;transition:transform .1s ease,box-shadow .1s ease}
.tile:hover{transform:translateY(-3px);box-shadow:0 16px 40px rgba(22,32,46,.13)}
.thumb{height:228px;overflow:hidden;background:var(--bg);border-bottom:1px solid var(--line)}
.thumb img{width:100%;display:block}
.body{padding:14px 16px 16px}
.nm{font-family:var(--serif);font-size:19px;font-weight:800}
.kind{font-family:var(--mono);font-size:11px;color:var(--muted);margin-left:8px}
.tags{margin:9px 0 13px;display:flex;gap:6px;flex-wrap:wrap;min-height:8px}
.tag{font-size:11px;font-weight:600;background:#eef2fb;color:#1e3a8a;padding:3px 9px;
border-radius:999px;font-family:var(--mono)}
.links{display:flex;align-items:center;gap:13px;font-size:12px;font-family:var(--mono)}
.cta{background:linear-gradient(90deg,var(--accent),#0891b2);color:#fff !important;font-weight:700;
padding:7px 12px;border-radius:9px;margin-right:auto;text-decoration:none}
.links a{color:var(--muted);text-decoration:none}.links a:hover{color:var(--accent)}
footer{margin-top:42px;color:var(--muted);font-size:12px;font-family:var(--mono);text-align:center}
"""


# The interactive React Flow view (input -> traverse -> output, looping).
_VIEW_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>__NAME__ · OpenWorld</title>
<link rel="stylesheet" href="https://esm.sh/reactflow@11/dist/style.css">
<style>
 :root{--ink:#16202e;--accent:#1d4ed8;--ochre:#b45309;--teal:#0f766e;--line:#dde2ea}
 body{margin:0;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;color:var(--ink);background:#fcfbf8}
 header{padding:12px 18px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:baseline}
 header b{font-size:18px} header span{color:#5b6675;font-size:13px}
 .wrap{display:grid;grid-template-columns:1fr 340px;height:calc(100vh - 52px)}
 .graph{height:100%} .side{border-left:1px solid var(--line);padding:16px;overflow:auto}
 h3{font-size:11px;letter-spacing:1.5px;color:#5b6675;margin:16px 0 8px}
 textarea,select{width:100%;font-family:ui-monospace,Menlo,monospace;font-size:12px;padding:8px;border:1px solid var(--line);border-radius:8px;box-sizing:border-box}
 button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:9px 14px;font-weight:700;cursor:pointer;margin-top:8px}
 button.alt{background:#fff;color:var(--accent);border:1px solid var(--accent)}
 pre{background:#eef1f6;border-radius:8px;padding:10px;font-size:11px;overflow:auto;max-height:220px}
 .out{border:1px solid var(--teal);border-radius:8px;padding:10px;background:#f0faf8;font-size:12px}
</style></head><body>
<header><b>🌐 __NAME__</b><span id="meta">loading…</span><span style="margin-left:auto"><a href="/worlds/__NAME__/card.svg">card.svg</a> · <a href="/docs">api</a></span></header>
<div class="wrap"><div class="graph" id="graph"></div>
<div class="side" id="side"></div></div>
<script type="module">
import React,{useState,useEffect,useCallback} from 'https://esm.sh/react@18';
import {createRoot} from 'https://esm.sh/react-dom@18/client';
import ReactFlow,{Background,Controls} from 'https://esm.sh/reactflow@11?deps=react@18,react-dom@18';
import htm from 'https://esm.sh/htm';
const html=htm.bind(React.createElement);
const NAME="__NAME__";
const api=(p,b)=>fetch(p,b?{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(b)}:undefined).then(r=>r.json());

function App(){
  const [nodes,setNodes]=useState([]); const [edges,setEdges]=useState([]);
  useEffect(()=>{(async()=>{
    const rf=await api(`/worlds/${NAME}/reactflow`);
    setNodes((rf.nodes||[]).map(n=>({...n,data:{label:(n.data&&n.data.label||'').replace(/\n/g,' · ')}})));
    setEdges(rf.edges||[]);
  })()},[]);
  return html`<${ReactFlow} nodes=${nodes} edges=${edges} fitView=${true}
      onInit=${(i)=>setTimeout(()=>i.fitView(),80)}>
      <${Background}/><${Controls}/></${ReactFlow}>`;
}
function Side(){
  const [info,setInfo]=useState(null);const [text,setText]=useState("");const [act,setAct]=useState("");
  const [state,setState]=useState(null);const [out,setOut]=useState(null);const [busy,setBusy]=useState(false);
  useEffect(()=>{(async()=>{const i=await api(`/worlds/${NAME}`);setInfo(i);setState(i.initial_state);
    document.getElementById('meta').textContent=`${i.kind} · ${i.actions.length} actions · ${i.bridges.length} bridges${i.perception&&i.perception.length?' · perceives '+i.perception[0].modality:''}`;
    const a=await api(`/worlds/${NAME}/actions`);setAct(a.default);})()},[]);
  const hi=(cur)=>{document.querySelectorAll('.react-flow__node').forEach(el=>{el.style.boxShadow=(el.getAttribute('data-id')===cur)?'0 0 0 3px #1d4ed8':'';});};
  const animate=async(frames,output)=>{for(const f of frames){setState(f.state);hi(f.current_node);await new Promise(r=>setTimeout(r,420));}setOut(output||null);};
  const run=async()=>{setBusy(true);setOut(null);try{const r=await api(`/worlds/${NAME}/run`,{input:{modality:'text',data:text},steps:8});await animate(r.trajectory,r.output);}finally{setBusy(false);}};
  const step=async()=>{const r=await api(`/worlds/${NAME}/step`,{state,action:{name:act}});setState(r.next_state);hi(r.current_node);};
  if(!info)return html`<p>loading…</p>`;
  const hasPerc=info.perception&&info.perception.length>0;
  return html`<div>
    ${hasPerc?html`<${React.Fragment}>
      <h3>INPUT (${info.perception[0].modality})</h3>
      <textarea rows=5 placeholder="paste input, e.g.\npriority: 7\nload: 40" value=${text} onChange=${e=>setText(e.target.value)}></textarea>
      <button disabled=${busy} onClick=${run}>${busy?'running…':'▸ Run'}</button>
    </>`:html`<${React.Fragment}>
      <h3>ACTION</h3>
      <select value=${act} onChange=${e=>setAct(e.target.value)}>${info.actions.map(a=>html`<option key=${a}>${a}</option>`)}</select>
      <button onClick=${step}>Step ▸</button>
    </>`}
    <h3>STATE</h3><pre>${JSON.stringify(state,null,1)}</pre>
    ${out?html`<h3>OUTPUT</h3>${out.emitted.map((e,i)=>html`<div class=out key=${i}><b>${e.modality}</b>${e.report?html`<div>${e.report}</div>`:''}<pre>${JSON.stringify(e.fields,null,1)}</pre></div>`)}`:''}
  </div>`;
}
createRoot(document.getElementById('graph')).render(html`<${App}/>`);
createRoot(document.getElementById('side')).render(html`<${Side}/>`);
</script></body></html>
"""
