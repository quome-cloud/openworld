# OpenWorld in three phases: build → optimize → deploy

OpenWorld works like an AutoML framework for *verified symbolic world models*.
You **build** a world (Claude Code authors it), **optimize** it toward a goal, and
**deploy** it as an inference server you can call or demo in the browser.

```bash
pip install -e .          # installs the `openworld` CLI (fastapi/uvicorn/click/rich)
```

The core library stays zero-dependency; the CLI/server are the one batteries-
included layer.

## 1. Build — Claude Code authors a world from a description

```bash
openworld build "a support-ticket queue you can paste a ticket into" --name intake
```

This scaffolds `build/intake.py` + `build/BUILD.md` and opens an interactive
**Claude Code** session in tmux that authors and verifies the world, then writes
`specs/intake.json`. (No tmux/claude? It prints the scaffold + the exact prompt to
run manually.) The result is a portable [world spec](openworld/spec.py): state,
actions, rules, verified dynamics, and — for input — a `CodePerceptor` and an
`emit` report.

```bash
openworld ls specs/                 # inspect specs
openworld card specs/intake.json --open   # render the SVG model card
```

## 2. Optimize — tune it toward a goal

```bash
openworld optimize specs/intake.json --goal "clear high-priority load fastest"
```

Claude Code iterates with the framework's tuners (`Study`/`Tuner`,
`optimize.sweep`, `objectives`) — propose → run → measure → keep best — and writes
an improved `specs/intake.v2.json`.

## 3. Deploy — a stateless inference server

```bash
openworld serve specs/ --allow-code        # http://127.0.0.1:8080
```

`--allow-code` runs the worlds' verified dynamics (only for specs you trust).
Open `http://127.0.0.1:8080/` for the index, `/docs` for the OpenAPI explorer, or
a world's **live view**:

```
http://127.0.0.1:8080/worlds/intake/view
```

Paste an input (e.g. `priority: 7` / `load: 4`) and watch the world **perceive →
traverse its rules → emit a report**, animated in React Flow. Loop with new input.

### Call it like a model

```bash
# single forward pass: state + action -> next state
curl -s localhost:8080/worlds/intake/step -H 'content-type: application/json' \
  -d '{"state":{"priority":7,"load":4,"done":0},"action":{"name":"work"}}'

# perception pipeline: raw input -> perceive -> roll -> emit
curl -s localhost:8080/worlds/intake/run -H 'content-type: application/json' \
  -d '{"input":{"modality":"text","data":"priority: 7\nload: 4"},"steps":8}'

# batch predict
curl -s localhost:8080/worlds/intake/predict -H 'content-type: application/json' \
  -d '{"inputs":[{"state":{"priority":7,"load":4,"done":0},"action":{"name":"work"}}]}'
```

Endpoints per world: `GET /worlds/{name}` (info), `/spec`, `/state`, `/actions`,
`/metrics`, `/card.svg`, `/reactflow`, `/mermaid`, `/view`; `POST /step`,
`/predict` (batch), `/rollout`, `/observe` (perception), `/run` (input pipeline);
`WS /live` (streamed, animated rollout). **Composites, bridges, and perception**
are served transparently — a composite's `tick`/namespaced actions and bridge
flow run inside the world's transition.

### Deploy a whole catalog

```bash
python examples/intake_world.py        # writes specs/intake.json
openworld serve specs/ gallery/ --allow-code   # serve a list/dir of specs
```
