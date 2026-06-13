# Multimodal inputs: the perception boundary

> Script: [`multimodal_perception.py`](multimodal_perception.py) — runs offline.

OpenWorld worlds are symbolic: their state is a JSON dict and their dynamics
are verified code. So how do you feed in a clinician's voice note, a camera
frame, or a free-text report? You **perceive** it into symbolic state at a
typed boundary, *before* any dynamics run. The core stays exact; perception is
a separate, explicitly fallible layer with its own checks.

## The shape

```
raw input ──▶ Perceptor ──▶ PerceptionGate ──▶ symbolic state ──▶ step() [verified dynamics]
 (text/audio/    (a sensor:     (contract-check:   (a plain JSON     (unchanged; exact
  image/video)   may be wrong)   owned fields,      delta committed)  as always)
                                 types, ranges)
```

Three objects, one new world method:

- **`Observation(modality, data)`** — one raw input. `modality` is `text`,
  `audio`, `image`, `video_frame`, or `video_segment`. Content-hashed for
  provenance; the raw payload never enters state.
- **`Perceptor`** — an *untrusted sensor*. `perceive(obs)` returns a partial
  symbolic delta and declares `produces` (the fields it may write) and a
  `schema` (types/ranges). `TextPerceptor` maps free text → fields via an LLM;
  `MockPerceptor` scripts deltas for offline/tests; audio and video perceptors
  are future subclasses that plug into the *same* boundary.
- **`PerceptionGate`** — rejects anything a perceptor shouldn't have produced
  (unowned field, wrong type, out-of-range) *at the boundary*, so bad
  perception never reaches the dynamics.
- **`world.observe(observations, perceptors)`** — runs perceptor → gate →
  commits the delta into the symbolic state, recording provenance on
  `world.perceptions`. Then you `step()` as always.

## Walkthrough (from the script)

An ICU bed with symbolic state `{hr, stable, minutes}`. A clinician note is
perceived into the `hr` vital:

```python
perceptor = TextPerceptor(note_llm, produces=["hr"], schema={"hr": (int, (20, 250))})
bed.observe(Observation("text", "Pt tachycardic, HR up to 118..."), perceptor)
# -> bed.state["hr"] == 118, recorded with a content hash in bed.perceptions
bed.step(Action("monitor"))   # verified dynamics: stable = hr < 100  ->  False
```

The gate is the trust layer. A perceptor that emits an impossible reading is
rejected before it can corrupt the world:

```
gate rejected  : field 'hr'=9000 out of range [20, 250]
```

## What this preserves (and what it doesn't)

- **The symbolic core is untouched.** State stays JSON-serializable, dynamics
  stay verified and bit-exact, replay stays deterministic. A world that never
  calls `observe()` behaves exactly as before — perception is purely additive.
- **Perception is the one fallible layer, and it's honest about it.** It can't
  be *proven* correct, only *measured* (against labeled raw→symbolic examples)
  and *contract-checked* (the gate). It is never reported as "verified." Replay
  is exact from the recorded symbolic deltas; re-running an LLM perceptor on
  the raw input may vary.
- **A clean error decomposition.** Because the dynamics add zero error,
  end-to-end error equals perception error — which the boundary makes
  measurable.

## Where this goes

This is Phase 1 (text, offline). Audio (ASR/transcript → text path) and
video/image (frame → vision model) are new `Perceptor` subclasses behind the
identical boundary — additive, no change to the world or its dynamics. See the
design spec at `docs/superpowers/specs/2026-06-13-multimodal-perception-design.md`.
