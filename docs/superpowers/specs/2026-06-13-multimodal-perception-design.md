# Multimodal inputs via a perception boundary

**Date:** 2026-06-13
**Status:** approved scope (design only; no implementation this round)

## Goal

Let OpenWorld ingest **text, audio, and video/image** as inputs without
touching the verified symbolic-state + code-dynamics core. Raw media is
resolved to symbolic state at a typed, *untrusted* boundary (a Perceptor)
*before* any transition runs. Dynamics stay symbolic and bit-exact; perception
is a new, explicitly fallible component with its own evaluation. This
operationalizes the "hybrid perception-to-symbol pipeline" the paper's
conclusion names as future work.

## Non-negotiable principle: additive, no change-failure

This feature must be **purely additive and extensibly modular**. Adding it
cannot alter or break any existing behavior. Concrete guarantees the design
holds itself to:

1. **No signature changes** to `World.__init__`, `World.step`, `WorldState`,
   `Transition`, `Simulation`, `Objective`, `Dial`, or the LLM classes' public
   methods. Perception is reached only through *new* surfaces.
2. **`WorldState` is unchanged** — still a plain JSON-serializable dict.
   Perceptors write into existing state fields through the existing dict
   interface; no new state type.
3. **Opt-in:** a world with no perceptors behaves byte-identically to today.
   `World.observe()` is a new method; never calling it changes nothing.
4. **No new required dependencies.** Text + mock perceptors need only the
   stdlib + existing LLM client. Audio (ASR) and video (vision model) live
   behind guarded optional imports; absent → those adapters are unavailable,
   the rest works.
5. **Open/closed extensibility:** new modalities/adapters are added by
   subclassing `Perceptor`, with no edits to the core, the gate, or
   `observe()` (all modality-agnostic).
6. **Regression guarantee:** the entire existing test suite passes unchanged
   after the foundation lands; existing tutorials/experiments produce
   identical output.

## Architecture (new module `openworld/perceive.py`)

- **`Observation(modality, data, t=None)`** — a typed wrapper for one raw
  input. `modality` ∈ {`text`, `audio`, `image`, `video_frame`,
  `video_segment`}. `data` is the payload (str for text; bytes/path for
  media). Content-hashed for provenance.
- **`Perceptor`** (base / `Protocol`) — `perceive(observation) -> dict`
  returns a *partial* symbolic state update; declares `produces: List[str]`
  (the state fields it owns) and `schema` (type/range per field). A Perceptor
  is a sensor: it may be wrong. Modality-agnostic interface.
- **`PerceptionGate`** — contract-checks a perceptor's output against the
  world's declared state schema / the perceptor's `schema`: rejects fields the
  perceptor doesn't own, wrong types, or out-of-range values, *at the
  boundary*. Bad perception is caught, never silently fed to dynamics.
- **`World.observe(observations, perceptors=None) -> WorldState`** — new,
  optional. Runs each observation through its perceptor → gate → commits the
  merged symbolic delta into `self.state`. Returns the updated state. Then the
  caller `step(action)`s as today. Perception and dynamics are separate calls,
  separately evaluated.
- **Trajectory recording:** `observe` appends a perception record to the
  trajectory: `{modality, produces, delta, input_sha256, perceptor}`. The raw
  media is **not** stored in state (preserves JSON/determinism); only the
  symbolic delta + a content hash for provenance.

## The seams (stated honestly)

- **Determinism / replay.** The symbolic state and dynamics remain bit-exact
  and JSON-serializable. Replay from recorded *deltas* is exact. Re-perception
  from cached raw inputs is best-effort: an LLM-backed perceptor may vary
  run-to-run. Perception is the acknowledged non-deterministic layer; the
  dynamics layer is not.
- **Two-layer verification, reported separately.** Dynamics verification is
  unchanged (and exact). Perception cannot be *proven* correct, so it is
  *measured*: (a) the typed contract gate; (b) a **perception-fidelity** score
  against labeled `raw → expected symbolic fields` examples; (c) optional
  ensemble / temporal-consistency cross-checks. Never reported as "verified."
- **Error decomposition (the payoff).** Because dynamics add zero error,
  end-to-end error equals perception error. The architecture makes this
  cleanly measurable — a natural future experiment and paper contribution.

## Modality handling (asymmetric, by necessity)

- **Text →** existing text LLM (or rules) maps free text → symbolic fields.
  Deterministic with `MockLLM`. No new deps. *The simplest, first real
  adapter.*
- **Audio →** ASR/transcript → text path → symbol. Ollama has no native
  audio; ASR (e.g. a Whisper-class model) is an optional guarded dependency,
  or the caller supplies a transcript (deterministic-friendly).
- **Video / image →** frame-sample → vision LLM → symbol. Requires extending
  `OllamaLLM` with image input (Ollama accepts base64 `images` per message)
  and a pulled vision model. Frame sampling policy is the perceptor's concern.

## Phasing (each phase is independently shippable & additive)

1. **Foundation (offline, no deps):** `Observation`, `Perceptor`,
   `PerceptionGate`, `MockPerceptor`, **`TextPerceptor`** (text LLM/rules →
   symbol, runs offline via `MockLLM`), `World.observe()`, trajectory
   recording, tests, exports. Proves the boundary end-to-end with text.
2. **Audio:** `TranscriptPerceptor` (+ optional guarded ASR adapter) and a
   tutorial — e.g. a triage world fed clinician voice notes → symbolic vitals.
3. **Video / image:** `OllamaLLM` image support + `VisionPerceptor` + frame
   sampling + a demo (frame → symbolic scene state).
4. **Eval + paper:** a perception-fidelity experiment and the
   error-decomposition result; a paper subsection turning the conclusion's
   "hybrid perception-to-symbol" promise into a measured contribution.

## Explicit non-goals

- Latent multimodal dynamics / pixel-native next-frame prediction (Dreamer's
  domain; intentionally out of scope).
- Storing raw media in `WorldState`.
- Claiming perception is verified-exact (it is measured, not proven).
- Any change to existing public APIs or default behavior.

## Open questions for review

- Should `schema` be declared on the Perceptor, inferred from the world's
  `initial_state`, or both (Perceptor declares, gate cross-checks against the
  world)? (Leaning: Perceptor declares; gate cross-checks field ownership
  against the world's state keys.)
- Phase 2 vs 3 ordering — audio (reuses the text path, lower lift) before
  video (needs vision-LLM client work) is the recommended order; confirm.
