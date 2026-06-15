# OpenWorld: build → optimize → deploy — Design

**Goal:** Make OpenWorld concrete like an AutoML framework with three phases —
**build**, **optimize**, **deploy** — fronted by a CLI-first experience. Build and
optimize are Claude-Code-driven (frontier LLM in a tmux pane). Deploy is a
FastAPI inference server that loads world models from specs and serves a
stateless "forward pass" (step / batch predict / rollout / observe), handling
composites, bridges, and perception.

**Decisions (approved):** CLI-first; add `fastapi`, `uvicorn`, `click`, `rich`
as real dependencies (core `import openworld` stays import-clean — these live in
serve/cli modules imported only on use; CLAUDE.md's zero-dep rule is amended to
carve out the serve/CLI layer); Claude-Code-in-tmux for build/optimize;
multi-world registry for deploy; server binds `localhost:8080` by default;
**stateless** (no server-side sessions); add **batch predict** and **per-world
metrics** endpoints; expose actions/bridges as first-class.

## Architecture

A new `openworld` console script (click + rich) with subcommands per phase, plus
a FastAPI app factory.

| File | Responsibility |
|------|----------------|
| `openworld/serve.py` | `serve_app(specs, allow_code) -> FastAPI`; `WorldRegistry`; Pydantic request/response models; metrics computation. |
| `openworld/cli.py` | click CLI: `build`, `optimize`, `serve`, `ls`, `card`; rich output. Entry point `openworld`. |
| `openworld/_tmux.py` | Launch/drive a `claude` session in tmux (build/optimize); detect availability; tail pane; wait-for-file. |
| `pyproject.toml` | Project metadata, dependencies, `openworld` console script. |
| `tests/test_serve.py` | FastAPI `TestClient`: step/predict/rollout/observe/metrics/actions, composite + bridge behavior, code-gating. |
| `tests/test_cli.py` | click `CliRunner`: `ls`/`card`/`serve --help`; build/optimize availability messaging. |
| `QUICKSTART.md` | End-to-end build → optimize → deploy walkthrough. |

Core `openworld/__init__.py` does NOT import serve/cli, so `import openworld`
works without fastapi/click installed.

## Phase 1 — BUILD (`openworld build "<description>" --name foo`)

Natural language → a validated world spec, authored by Claude Code.

1. Scaffold: write `build/foo.py` (framework cheatsheet + a `to_spec(world)` →
   `specs/foo.json` write-stub) and `build/BUILD.md` (instructions: author a
   `World`/`CompositeWorld` with verified `CodeTransition` dynamics, optional
   perceptors/emit/objectives, then run the scaffold to emit the spec).
2. `_tmux.py` opens a tmux session running `claude` in the repo and sends an
   opening message pointing at `BUILD.md`. `rich` tails the pane and shows a
   spinner.
3. The CLI waits for `specs/foo.json`; on appearance runs `validate_spec` +
   round-trip check and prints the card path. If tmux/`claude` are missing, it
   prints the scaffold location and the exact prompt to paste into Claude
   manually (graceful degradation).

## Phase 2 — OPTIMIZE (`openworld optimize specs/foo.json --goal "..."`)

Improve a spec against a goal, using the framework's existing optimizers
(`openworld.tune` Study/Tuner, `openworld.optimize.sweep`, `objectives`).

- Spawns Claude in tmux with the current spec, the goal, and the tuning tools;
  it iterates propose → run world → measure → keep best, then writes a versioned
  improved spec (`specs/foo.v2.json`). `rich` shows iteration/metric progress.
- Same graceful degradation as build.

## Phase 3 — DEPLOY (`openworld serve specs/ --port 8080`)

FastAPI multi-world inference server — a stateless forward pass for worlds.

- **Registry:** loads every spec in a dir/list at startup via `from_spec`.
  Running dynamics needs executable code, so loading runnable worlds requires
  `--allow-code` (specs are trusted local files; a warning is printed). Without
  it, worlds register metadata-only and `step`/`predict`/`rollout` return 403.
- **Stateless:** the client always passes `state`; the server never holds session
  state. (Mirrors a model `forward(x)`.)
- **Endpoints** (Pydantic models; auto OpenAPI at `/docs`):
  - `GET /` — HTML index (worlds + links to cards/docs).
  - `GET /healthz` — liveness.
  - `GET /worlds` — list (name, kind, description, #actions, #bridges).
  - `GET /worlds/{name}` — info: description, kind, state_schema, initial_state,
    actions, bridges, aggregators, perception, emit, objectives.
  - `GET /worlds/{name}/spec` — full spec JSON.
  - `GET /worlds/{name}/card.svg` — the SVG model card (`image/svg+xml`).
  - `GET /worlds/{name}/state` — initial state.
  - `GET /worlds/{name}/actions` — available actions (for composites: namespaced
    child actions + `tick`, and `travel` when routes exist) + bridge list.
  - `GET /worlds/{name}/metrics` — per-world metrics (below).
  - `POST /worlds/{name}/step` — `{state, action:{name,params,agent}}` →
    `{next_state, changed}`. Composites/bridges handled by `world.transition.step`.
  - `POST /worlds/{name}/predict` — **batch**: `{inputs:[{state, action}]}` →
    `{outputs:[{next_state, changed}]}` (the batch forward pass).
  - `POST /worlds/{name}/rollout` — `{state?, actions:[…]}` → `{trajectory:[…]}`.
  - `POST /worlds/{name}/observe` — **perception**: validate a supplied perceptual
    `delta` against the declared perceptor schema via `PerceptionGate` (and run
    live perceptors when a world is loaded from Python with them), merge, then
    optionally step.
  - `POST /worlds` — hot-load a spec at runtime (`{spec}` or `{path}`; guarded by
    `--allow-code`).
  - `GET /worlds/{name}/reactflow` — `to_reactflow(spec)` JSON (nodes/edges/
    playground), for the live view.
  - `GET /worlds/{name}/view` — an interactive **React Flow** page (below).
  - `WS /worlds/{name}/live` — step over a WebSocket (below).

### Live interactive view — input → traverse → output, looping

The deploy frontend is **input-centric**, not API-centric: the most tangible demo
is to paste an input, watch the world perceive it and traverse its rules live in
React Flow, then read the output artifact — and loop.

- **`GET /worlds/{name}/view`** returns a single HTML page (React + React Flow
  from esm.sh CDN — this page is not self-contained, unlike `/card.svg`, which is
  still offered) rendering the graph from `GET /worlds/{name}/reactflow`. Two
  modes, chosen by whether the world has perception:
  - **Input mode (perception worlds):** an **input box** (text for a `text`
    perceptor). Pressing **Run** sends the raw input; the page then animates the
    pipeline through the graph — the `⌖ sensor` node fires → `perceive` edge →
    the perceived state lights up → the dynamics auto-roll (each `step` animating
    the active node + the action edge) → the `▸ output` node fires → the **output
    artifact** appears (the emitted fields, plus a report string if the emitter
    declares one). Clear and paste again to **loop**.
  - **Action mode (no perception):** a JSON **state panel** + **action picker**
    (from `/actions`) + **Step**/**Play**, as a manual fallback.
- **`POST /worlds/{name}/run`** powers input mode: body `{input:{modality,data},
  steps?}` → `{perceived_delta, state, trajectory, output}`. The server
  perceives the input (live perceptor, or a reconstructed `CodePerceptor` under
  `--allow-code`), gates the delta, merges it, rolls the dynamics `steps` times,
  and emits the output artifact. The frontend animates from this single response.
- **WebSocket (`WS /worlds/{name}/live`):** streams the run/rollout step-by-step
  so the graph animates in real time (and multiple viewers can watch). Each
  message carries `{state, action}` (action mode) or `{input}` (input mode); the
  server replies per step with `{next_state, changed, current_node, emitted?}`.
  Stateless — state rides on the message.
- `current_node`: the server keys each precomputed graph node by its state (the
  same BFS as the card); a matching stepped state returns that id to highlight,
  else `null` (the state panel/edge still animate).

### Runnable perception/emit (so demos work from a spec)

For "paste text → see it traverse" to work from a spec with no LLM at serve time,
perception/emit must be reconstructable:

- **`CodePerceptor`** (new, in `openworld/perceive.py`): a `Perceptor` whose
  `perceive(observation) -> delta` is verified Python code (run in the sandbox),
  declaring `modality`/`produces`/`schema` like any perceptor. Serialized into
  the spec (`perception[i].kind = "CodePerceptor"`, `code = "..."`) and rebuilt by
  `from_spec(..., allow_code=True)` — symmetric to `CodeTransition`. Live
  perceptors (e.g., `TextPerceptor` with an LLM) still work when a world is
  registered from Python.
- **Emit** may optionally carry a `report` code/template that turns the emitted
  fields into a human-readable artifact string; otherwise the output is just the
  emitted field values. (LLM-backed perception remains possible but is opt-in and
  out of scope for the default serve path.)
- **Per-world metrics** (`/metrics`): state fields, action count, dynamics kind
  (verified/code/llm), reachable states (from the preview graph), composite depth,
  #children / #bridges / #aggregators / #bindings / #agents, perception & emit
  channels, objectives count, spec size (bytes), plus any `card.metrics`.
- CLI `serve` runs `uvicorn` on `127.0.0.1:8080` (configurable `--host/--port`).

## Error handling
- Unknown world → 404; unknown action on a world → handled by the transition
  (unchanged state) but `/actions` advertises the valid set; malformed
  state/action → 422 (Pydantic); stepping a code-gated world → 403 with a clear
  message; perception delta violating the gate → 422 with the gate's reason.

## Testing
- `test_serve.py` uses `TestClient` over `serve_app([...], allow_code=True)`:
  step on a leaf, batch predict, rollout, composite `tick` + a bridge effect,
  `/actions`, `/metrics`, `/reactflow`, `/view` (HTML), the `/live` WebSocket
  (send `{state, action}` → receive `{next_state, current_node}`), `/observe`
  gate accept+reject, and a 403 when `allow_code=False`.
- `test_cli.py` uses `CliRunner`: `ls` over a spec dir, `card` writes an SVG,
  `serve --help`; build/optimize print the graceful-degradation message when
  tmux/claude are absent.

## Out of scope (YAGNI)
- Auth/rate-limiting/multi-tenant (deploy behind a gateway if needed).
- A hosted web UI beyond the index + `/docs` + SVG cards.
- Sandboxed execution of untrusted specs (the `--allow-code` flag is a trust
  gate, documented as such, not a sandbox).

## Dependency / branch notes
- First third-party deps in the repo; core import path stays clean. CLAUDE.md
  updated accordingly.
- Depends on the E57 spec/card layer (PR #40). Branch is based on
  `e57-world-specs`; the PR targets `main` and collapses to just this work once
  #40 merges (so it won't strand).
