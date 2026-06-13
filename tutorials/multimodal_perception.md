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

## All three modalities, one boundary

The script extends the same ICU bed to audio and video — same `observe()`, same
gate, same dynamics:

- **`TranscriptPerceptor`** (audio) — the front of the pipe is transcription.
  Offline, the audio `Observation` carries a transcript string; live, you
  inject `transcribe=<asr_model>`. Ollama has no native audio, so ASR is the
  optional, guarded dependency; field extraction reuses the text path.
- **`VisionPerceptor`** (image / video frame) — a frame is base64-encoded and
  passed to a vision model through the additive `images=` channel on
  `OllamaLLM` (works with any `BaseLLM`, so `MockLLM` drives it in tests). For a
  clip, `sample_frames(frames, k)` picks evenly-spaced frames to perceive.

```python
bed.observe(Observation("audio", "heart rate ninety-six"), transcript_perceptor)
bed.observe(Observation("video_frame", frame_bytes), vision_perceptor)
```

Every modality is a `Perceptor` subclass behind the identical, gated boundary —
purely additive, no change to the world or its verified dynamics.

## Why this stays honest at scale

Perception is *measured, not proven*. The architecture makes the measurement
clean: because the dynamics add zero error, **end-to-end error equals
perception error**. The deterministic `experiments/e39_perception_fidelity.py`
demonstrates exactly this decomposition — a perceptor that is wrong some of the
time yields end-to-end accuracy equal to its perception accuracy, while the
dynamics layer stays exact throughout. See the design spec at
`docs/superpowers/specs/2026-06-13-multimodal-perception-design.md`.
