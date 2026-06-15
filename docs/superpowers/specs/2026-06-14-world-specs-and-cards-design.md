# World-Model Specs & Model Cards — Design

**Goal:** Represent any OpenWorld world model as a portable JSON spec (publishable
to a "HuggingFace for world models" marketplace) and render beautiful,
self-contained HTML model cards from those specs.

**Decisions (locked):** self-contained HTML+SVG visualizations (stdlib only,
keeps the framework zero-dependency); a *full model card* as the headline layout;
a *lossless round-trip* spec (`world → JSON → world` preserves behavior), with
code execution gated off by default when loading untrusted specs.

## Architecture

Two new stdlib-only modules in the core package, plus a demo experiment that
seeds a real marketplace gallery in the repo.

| File | Responsibility |
|------|----------------|
| `openworld/spec.py` | Spec format: `to_spec`, `from_spec`, `validate_spec`, JSON I/O. Dict-based; hand-written validator (no `jsonschema` dep). |
| `openworld/card.py` | `render_card(world_or_spec, path=None, theme="light")` → self-contained HTML string; internal SVG builders for the composition diagram and rollout sparkline; `render_gallery(specs, path)` → grid index. |
| `openworld/__init__.py` | Export `to_spec`, `from_spec`, `validate_spec`, `render_card`, `render_gallery`, `SPEC_VERSION`. |
| `tests/test_spec.py` | Round-trip fidelity (leaf + composite), validation catches malformed specs, code-gating keeps downloaded specs inert by default, lossy flags are honest. |
| `tests/test_card.py` | Card renders valid self-contained HTML (no external refs), expected sections present, SVG well-formed; gallery renders. |
| `experiments/e57_world_specs.py` | Serialize several real worlds → specs + cards → `gallery/index.html`; assert byte-stable round-trips and zero validation errors. |
| `gallery/` | Generated marketplace seed: per-world `*.json` specs, `*.html` cards, `index.html`. |

## The JSON spec (v1.0)

```json
{
  "openworld_spec_version": "1.0",
  "name": "orchard-economy",
  "card": {
    "version": "1.2", "license": "MIT", "authors": ["..."],
    "tags": ["economy", "multi-agent", "verified"],
    "description": "...", "lineage": "derived from E44", "metrics": {}
  },
  "state_schema": { "apples": "int", "harvested": "dict", "prices": "list[float]" },
  "initial_state": { "...": "concrete values, for exact reconstruction" },
  "actions": ["pick", "wait", "trade"],
  "rules": ["'pick' moves one apple ..."],
  "transition": { "kind": "code", "func_name": "transition", "code": "def transition(...)..." },
  "composite": {
    "children": { "farm": { "<nested spec>": "..." } },
    "bridges": [ { "name": "...", "a": "farm", "b": "market", "kind": "bridge",
                   "transition": { "...": "..." }, "description": "", "rules": [] } ],
    "aggregators": [ { "name": "total_wealth", "source": "def total_wealth(children): ...",
                       "lossy": false } ],
    "bindings": [ { "source_path": ["_agg", "rate"], "child": "farm", "key": "rate" } ],
    "timescales": { "farm": 1 }, "default_actions": { "farm": "grow" },
    "agents": { "alice": { "loc": "farm" } }
  },
  "preview": { "steps": 12, "series": { "apples": [10, 9, 7, "..."], "wealth": ["..."] } }
}
```

- **`state_schema`** is inferred from `initial_state` types recursively
  (`int`/`float`/`str`/`bool`/`list[...]`/`dict`/`null`); `initial_state` keeps
  concrete values for exact reconstruction.
- **`transition.kind`** ∈ `{code, phased, llm, function}`:
  - `code` → `CodeTransition`: full code text (round-trippable, gated on load).
  - `phased` → `PhasedTransition`: recursive list of phase transition specs.
  - `llm` → `LLMTransition`: `{description, rules}`; needs an `llm=` on load,
    else loads as a descriptive stub.
  - `function` → `FunctionTransition`: `inspect.getsource` when available
    (serialized as `code`), else `{kind:"function", lossy:true, repr:"..."}`.
- **`composite`** present only for `CompositeWorld`; fully recursive. Aggregator
  `fn` serializes via `inspect.getsource` (lossy-flagged if unavailable, like
  function transitions). `Route` rows carry `kind:"route"` + optional `on_cross`.
- **`preview.series`** is computed once at `to_spec` time by rolling the *live*
  world forward `preview_steps` and recording top-level numeric state vars — so
  the card sparkline needs no code execution at render time.

## Round-trip & marketplace safety

- `to_spec(world, *, card=None, preview_steps=12) -> dict`.
- `from_spec(spec, *, allow_code=False, llm=None) -> World`. Rebuilds an
  identical runnable world. **Code is inert by default:** `kind:"code"` /
  `kind:"function"` transitions are *not* compiled unless `allow_code=True` (the
  trust gate for downloaded specs). Without it the world loads fully described —
  structure, metadata, schema, preview intact — but its transition raises a clear
  `SpecError` if stepped. With `allow_code=True` the round-trip is behavioral:
  the test steps original vs reloaded through an action sequence and asserts the
  resulting states match.
- `validate_spec(spec) -> list[str]` is the publish gate: returns a list of
  human-readable problems (empty = valid). Checks version, required fields,
  schema/initial-state agreement, transition kind, and recurses into composite
  children and bridges.

## The card (self-contained `.html`)

Inline `<style>` + inline `<svg>`, **no external references** (so a card is a
single droppable file). Layout matches the approved mock:

- **Header band:** 🌍 name · version · license; one-line description; tag chips.
- **Left column — composition:** an SVG diagram. Leaf world → a single titled box
  with a couple of state keys. Composite → nested boxes (recursively), bridge
  arrows between siblings, aggregator roll-up boxes, and counts (worlds / bridges
  / agents).
- **Right column:** state-schema list, action chips, and an SVG rollout sparkline
  (polyline per `preview.series` var, labeled with trend arrows).
- **Theme:** `light` / `dark` via a small CSS variable block.
- **`render_gallery(specs, path)`** writes a responsive grid of compact tiles
  (mini composition glyph + tags + badges) linking to each full card —
  `gallery/index.html`.

## Paper integration (E57)

`experiments/e57_world_specs.py` serializes a handful of real worlds already in
the repo (a leaf world, the sprint world, and a composite), round-trips each,
and emits the gallery. Self-checks: every spec validates with zero errors, and
each `allow_code=True` reload reproduces the original rollout exactly. Paper gets
a short subsection + one figure (a rendered card screenshot or an SVG of the
composition) and a small table (worlds serialized, round-trip fidelity,
spec size). Follows the standard `make_paper_assets.py` pipeline and
`\NumExperiments` 55→56.

## Out of scope (YAGNI)
- A hosted backend / upload API for the marketplace (this is the on-disk format
  and renderer; hosting is a separate project).
- Sandboxed execution of untrusted code (we gate it off by default and document
  the risk; real sandboxing is future work).
- Editing worlds through the card (cards are read-only artifacts).
