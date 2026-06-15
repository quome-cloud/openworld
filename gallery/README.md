# OpenWorld model gallery

A small **marketplace of world models** — the kind of thing a "HuggingFace for
world models" would host. Each world is published as two artifacts:

- **`<name>.json`** — a portable, lossless [world-model spec](../openworld/spec.py):
  name, description, state schema, concrete initial state, actions, rules, and
  dynamics (recursively, for composites). This is the unit you'd upload/download.
- **`<name>.svg`** — a self-contained model card rendered from the spec: a graph
  of the world (leaf → its real state-transition automaton; composite → its
  dataflow of child worlds, bridges, and aggregators), plus state schema, action
  set, and a sample-rollout sparkline. One file, no external fetches — embeds
  straight into a README with `<img src="...">`.

`index.svg` is the browsable contact sheet (tiles link to each card).

Regenerate everything with:

```bash
python experiments/e57_world_specs.py
```

## Using a spec

```python
import json
from openworld import from_spec, render_card, to_mermaid, to_reactflow

spec = json.load(open("gallery/sprint.json"))

# Inspect safely — code stays inert (a downloaded spec is not executed):
world = from_spec(spec)

# Opt in to run the verified dynamics:
world = from_spec(spec, allow_code=True)

# Re-render the card, or emit a Mermaid diagram (GitHub renders it natively):
render_card(spec, path="sprint.svg")
print(to_mermaid(spec))

# Export for an interactive React Flow canvas (nested composites become
# parentNode group nodes). The returned dict includes a `playground` URL
# (https://play.reactflow.dev) — paste the nodes/edges there to explore.
flow = to_reactflow(spec)          # {"nodes": [...], "edges": [...], "playground": ...}
```

Every rendered card also carries a **▸ open in React Flow** link in its footer.

**Safety.** `from_spec` does **not** execute a spec's embedded transition code
unless you pass `allow_code=True`. Without it, a spec loads fully described
(schema, metadata, graph, preview) but its transition raises rather than runs.
This is a trust gate for untrusted downloads, not a sandbox against adversarial
code — only opt in for specs you trust.

## In this gallery

| World | Kind | Notes |
|-------|------|-------|
| `sprint` | leaf | Engineering sprint backlog (ship / fix / refactor). |
| `triage` | leaf | ICU triage queue; untreated critical patients deteriorate. |
| `orchard` | leaf | Shared apple orchard, multi-agent. |
| `hospital-network` | composite | Two triage clinics that transfer critical load and roll up treated counts. |

All four validate against the spec format and round-trip losslessly
(`from_spec(to_spec(w), allow_code=True)` reproduces the original rollout) — see
experiment **E57**.
