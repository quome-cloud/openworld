# Multimodal Phase 1 (perception foundation) - Plan

> Inline execution. ADDITIVE-ONLY: the full existing test suite must stay green at every step; no existing public signature changes.

**Goal:** The perception boundary made real and testable, offline, with text as the first modality. Spec: docs/superpowers/specs/2026-06-13-multimodal-perception-design.md

### Task 1: `openworld/perceive.py`
- `MODALITIES = ("text","audio","image","video_frame","video_segment")`.
- `Observation(modality, data, t=None)`: validates modality; `.sha256` content hash (bytes or str).
- `PerceptionError(ValueError)`.
- `Perceptor` base: attrs `modality`, `produces: list`, `schema: dict` ({field: type} or {field: (type,(lo,hi))}); `perceive(obs)->dict` raises NotImplementedError.
- `PerceptionGate.check(perceptor, delta)`: reject fields not in `produces`, wrong types, out-of-range; return the cleaned delta. Boundary enforcement.
- `MockPerceptor(produces, deltas, schema=None, modality="text")`: returns scripted deltas in order (last repeats) - offline/deterministic.
- `TextPerceptor(llm, produces, schema=None, system=None)`: prompts the LLM to extract `produces` as JSON, parses via `extract_json`, keeps only owned fields. Deterministic with MockLLM.
- Tests: `tests/test_perceive.py` (Observation validation+sha; gate happy + 3 rejections; MockPerceptor; TextPerceptor with MockLLM JSON).

### Task 2: additive `World.observe()` + exports
- New method `World.observe(observations, perceptors)` in world.py (NEW method, no edit to __init__/step/reset). Lazily init `self.__dict__.setdefault("perceptions", [])` so __init__ is untouched. `perceptors` may be a single `Perceptor` (all obs), a list aligned 1:1, or a dict modality->Perceptor. For each obs: gate.check(perceptor.perceive(obs)) -> `self.state.update(delta)` -> append provenance record {modality,produces,delta,input_sha256,perceptor}. Returns `self.state`.
- Export `Observation, Perceptor, PerceptionGate, PerceptionError, MockPerceptor, TextPerceptor` from `openworld/__init__.py` (+ `__all__`).
- Tests: observe commits delta then `step()` runs normally; provenance recorded; gate rejection surfaces; a world that never calls observe() is unchanged.
- **Run full `python -m pytest tests/ -q` — must be all-green (additive guarantee).**

### Task 3: tutorial `tutorials/multimodal_perception.md` + `.py`
- A small symbolic world; free-text observations perceived into state via TextPerceptor (offline MockLLM) / MockPerceptor; show the gate rejecting an out-of-schema percept; then dynamics `step()`. Runs offline, deterministic, self-asserts. README row + pointer.

### Task 4: verify + PR
- Full suite green; tutorial exits 0; commit; update PR #19 (spec+Phase1) or new PR.
