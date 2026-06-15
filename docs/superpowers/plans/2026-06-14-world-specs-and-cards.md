# World-Model Specs & Model Cards — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans
> (inline). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Portable JSON specs for any world model + self-contained HTML model
cards, with a demo gallery wired in as experiment E57.

**Architecture:** Two stdlib-only modules — `openworld/spec.py`
(serialize/deserialize/validate) and `openworld/card.py` (HTML+SVG renderer) —
exported from the package, covered by `tests/test_spec.py` and
`tests/test_card.py`, demonstrated by `experiments/e57_world_specs.py` which
seeds `gallery/`, then integrated into the paper.

**Tech Stack:** Python 3.9, stdlib only (`json`, `inspect`, `html`, `math`,
`dataclasses`). No new runtime deps.

---

### Task 1: Spec serialization (`to_spec`) for leaf worlds

**Files:** Create `openworld/spec.py`; Test `tests/test_spec.py`.

- [ ] Write failing test: build a `World` with a `CodeTransition`; `to_spec`
  returns a dict with `openworld_spec_version == SPEC_VERSION`, `name`,
  `state_schema` matching `initial_state` types, `initial_state`, `actions`,
  `rules`, and `transition == {"kind":"code","func_name":...,"code":...}`.
- [ ] Implement `SPEC_VERSION="1.0"`, `to_spec(world, *, card=None, preview_steps=12)`,
  `_infer_schema(value)` (recursive type names: int/float/bool/str/null,
  `list[<elt>]`, `dict`), `_transition_to_spec(t)` (handles `CodeTransition`,
  `PhasedTransition`, `LLMTransition`, `FunctionTransition` via `inspect.getsource`
  → code, else `{"kind":"function","lossy":true,"repr":repr(t.fn)}`), and
  `_rollout_preview(world, steps)` (try/except: step a copy with noop/first
  action, collect top-level numeric vars; on any error return `{}`).
- [ ] Run: `pytest tests/test_spec.py -k leaf -v` → PASS.
- [ ] Commit.

### Task 2: Spec validation (`validate_spec`)

**Files:** Modify `openworld/spec.py`; Test `tests/test_spec.py`.

- [ ] Write failing test: a good spec → `validate_spec(spec) == []`; specs with a
  missing `name`, wrong version, or bad `transition.kind` each return a non-empty
  list naming the problem.
- [ ] Implement `validate_spec(spec) -> list[str]` (checks version, required keys,
  `transition.kind` ∈ allowed, schema keys ⊆ initial_state keys; recurse into
  `composite.children` and `composite.bridges`). Add `SpecError(Exception)`.
- [ ] Run: `pytest tests/test_spec.py -k validate -v` → PASS.
- [ ] Commit.

### Task 3: Deserialization (`from_spec`) + code gating + round-trip

**Files:** Modify `openworld/spec.py`; Test `tests/test_spec.py`.

- [ ] Write failing test (round-trip): `w2 = from_spec(to_spec(w), allow_code=True)`;
  step `w` and `w2` through the same action list and assert resulting states equal.
- [ ] Write failing test (gating): `from_spec(spec)` (default `allow_code=False`)
  returns a world whose `.step(...)` raises `SpecError`; `state_schema`/metadata
  still present via `to_spec(reloaded)` or attributes.
- [ ] Implement `from_spec(spec, *, allow_code=False, llm=None)` and
  `_transition_from_spec(d, allow_code, llm)`: `code`/`function`→ compile only if
  `allow_code` else an `_InertTransition` that raises `SpecError` on `step`;
  `phased`→ recurse; `llm`→ `LLMTransition(llm, description, rules)` if `llm` else
  inert stub. Add `spec_to_json(spec, indent=2)`, `spec_from_json(text)`.
- [ ] Run: `pytest tests/test_spec.py -v` → PASS.
- [ ] Commit.

### Task 4: Composite serialization round-trip

**Files:** Modify `openworld/spec.py`; Test `tests/test_spec.py`.

- [ ] Write failing test: build a `CompositeWorld` (two children + a `Bridge` with
  a `CodeTransition` + an `Aggregator` + a `Binding`); round-trip with
  `allow_code=True`; assert children, bridge names, binding, timescales,
  default_actions, agents survive and a `tick` step matches the original.
- [ ] Extend `to_spec`/`from_spec` with a `composite` block: recurse children;
  serialize `Bridge`/`Route` (incl. `transition`/`on_cross` via
  `_transition_to_spec`, `kind` "bridge"/"route"); `Aggregator.fn` via
  `inspect.getsource` (lossy-flag + drop fn on reload if unavailable);
  `Binding` (`source_path` list↔tuple); pass through `timescales`,
  `default_actions`, `agents`.
- [ ] Run: `pytest tests/test_spec.py -k composite -v` → PASS.
- [ ] Commit.

### Task 5: HTML+SVG model card (`render_card`)

**Files:** Create `openworld/card.py`; Test `tests/test_card.py`.

- [ ] Write failing test: `html = render_card(world)` is a `str` starting with
  `<!DOCTYPE html>`, contains the world name, an `<svg`, the word "ACTIONS", every
  action label, and has **no** `http://`/`https://`/`src=`/`<script` external refs
  (self-contained); `render_card(world, path=tmp)` writes the file.
- [ ] Implement `render_card(world_or_spec, path=None, theme="light")`: accept a
  world (call `to_spec`) or a spec dict; build header band, left composition SVG
  (`_composition_svg`: recursive nested boxes + bridge arrows + counts), right
  column (schema list, action chips, `_sparkline_svg` from `preview.series`),
  inline `<style>` with light/dark CSS vars. All values `html.escape`d.
- [ ] Run: `pytest tests/test_card.py -v` → PASS.
- [ ] Commit.

### Task 6: Gallery index (`render_gallery`) + exports

**Files:** Modify `openworld/card.py`, `openworld/__init__.py`; Test `tests/test_card.py`.

- [ ] Write failing test: `render_gallery([spec_a, spec_b], path=tmp)` writes an
  `index.html` containing both names and links to `<name>.html`; symbols importable
  from `openworld` top level.
- [ ] Implement `render_gallery(specs, path, theme="light")` (responsive grid of
  compact tiles). Add exports to `__init__.py`: `to_spec`, `from_spec`,
  `validate_spec`, `spec_to_json`, `spec_from_json`, `render_card`,
  `render_gallery`, `SPEC_VERSION`, `SpecError`.
- [ ] Run: `pytest tests/test_card.py tests/test_spec.py -v` → PASS.
- [ ] Commit.

### Task 7: E57 demo experiment + gallery seed

**Files:** Create `experiments/e57_world_specs.py`, `experiments/results/e57_world_specs.json`, `gallery/*`.

- [ ] Implement E57: import a leaf world, the sprint world (`experiments/common.py`),
  and a small composite; for each: `to_spec`, `validate_spec` (assert `[]`),
  reload with `allow_code=True`, assert rollout reproduces; write each
  `gallery/<name>.json` + `<name>.html` and a `gallery/index.html`. `save_results`
  BEFORE asserts (CLAUDE.md). Record per-world: validated, round_trip_exact,
  spec_bytes, n_children.
- [ ] Run: `python experiments/e57_world_specs.py` → checks pass, gallery written.
- [ ] Commit.

### Task 8: Paper integration

**Files:** Modify `scripts/make_paper_assets.py`, `paper/main.tex`.

- [ ] Add `"e57_world_specs"` to `EXPERIMENTS`; add `fig_world_specs` (composition
  SVG→PNG or a card screenshot rendered via matplotlib table) + `table_world_specs`
  (per-world round-trip fidelity + spec size); register calls in `main()`; add
  macros (e.g. `\SpecsNumWorlds`, `\SpecsRoundTrip`); bump `\NumExperiments` 55→56.
- [ ] Add a `\subsubsection{...E57}` to `paper/main.tex` with `\label{sec:specs}`,
  figure, and table.
- [ ] Run: `python scripts/make_paper_assets.py`; `cd paper && tectonic main.tex`;
  confirm 0 undefined refs.
- [ ] Commit.

### Task 9: Full suite + ship

- [ ] Run: `pytest -q` → all pass.
- [ ] Push branch `e57-world-specs`; open PR with base `main`.

## Self-review notes
- Spec covers every constructor field of `World` and `CompositeWorld` (verified
  against `world.py`/`compose.py` signatures): name, description, initial_state,
  actions, rules, transition; children, bridges, aggregators, bindings,
  timescales, default_actions, agents.
- Lossy paths (`FunctionTransition` without source, `Aggregator.fn` without
  source) are flagged, never silently dropped.
- Naming consistency: `to_spec`/`from_spec`/`validate_spec`/`render_card`/
  `render_gallery` used identically across plan, tests, and exports.
