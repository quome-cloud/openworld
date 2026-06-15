# E59 — Optimizing the brain architecture for a task, LLM held constant

**Goal:** Show, in a composite world model, that **the brain architecture matters
even with a frozen backbone**, and that we can **optimize the architecture for the
problem while keeping the LLM constant** — a neuromorphic architecture search over
composable worlds. Plus the genuine text-in → reason → **text-out** loop missing
today: a real **LLM emitter** for a live demo.

**Decisions (approved):** real LLM emitter for a live demo; the
"answer-needs-the-right-recalled-fact + verify" task suite; a new experiment (E59),
not a rewrite of E58.

## Framework addition: `LLMEmitter` (the missing text-out)
Symmetric to `TextPerceptor`: `LLMEmitter(llm, template, reads)` with
`.emit(state) -> str` — fills `template` from the named `reads` state fields and
asks the LLM to write the output text. Deterministic with `MockLLM`; serializes as
an emit channel `{kind:"llm", reads, template}`. This completes
perceive → world → **emit** with a real LLM at both boundaries.

## The task suite — recalled-fact + verify
Each task is a question whose answer requires a specific fact stored in long-term
memory among `D` distractors; some tasks are **hard** (a 2-step compose) where a
single draft often errs but a verifier can check a candidate.

- **Backbone (held constant):** a modeled LLM answerer with a fixed capability
  profile — it answers a simple task correctly **iff the needed fact is in its
  context**, and a hard task correctly with a fixed per-draft probability that a
  reliable **verifier** can filter. The *same* backbone is used by every
  architecture (the experiment's whole point). The live demo swaps in a real
  `OllamaLLM` via `TextPerceptor` + `LLMEmitter`.

## The architecture space (composite configs)
- `memory ∈ {none, longterm}` — retrieve the needed fact into context, or not.
- `tree_width ∈ {1, 3, 5}` — best-of-N drafts (tree-of-thoughts).
- `verify ∈ {off, on}` — self-check that filters drafts.

Each config is a brain `CompositeWorld` (conscious + unconscious + task), built by
`brain_world(arch)`; the best one serializes to a spec + card (asserted round-trip).

## What it proves (deterministic, seeded, self-checking)
- **Architecture matters, LLM constant:** the **bare LLM** (`none, 1, off`) scores
  low; **+retrieval** lifts simple-fact accuracy to ~1.0; **+best-of-N+verify**
  lifts the hard tasks; the **optimized** architecture beats the bare LLM by a real
  margin — all with the *identical* backbone.
- **The optimizer finds it:** a `Study`/`Tuner` searches the architecture space and
  recovers the top config; report the **leaderboard** + the **search**.
- **Ablation:** memory vs. best-of-N+verify contributions (which structure carries
  which gain).
- **Asserts:** `optimized > bare` by ≥ margin; retrieval ⇒ simple≈1.0;
  verify+width ⇒ hard improves; `Study.best == optimized config`; backbone
  capability identical across arms (by construction); brain spec round-trips.

## Live demo (real LLM)
`require_ollama` (fail-soft): build a brain with `TextPerceptor(llm)` (question →
conscious), retrieve the relevant fact, optional best-of-N, `LLMEmitter(llm)` writes
the answer; run on a couple of real Q&A and print the generated text — text-in →
reason → text-out, for real. Not part of the deterministic asserts.

## Files & integration
- `openworld/perceive.py`: `LLMEmitter` (+ export); optional emit serialization.
- `experiments/e59_brain_arch.py` (+ results JSON), self-contained (no E58 import).
- `tests/test_e59_brain_arch.py`: backbone-constant, retrieval/verify effects,
  Study recovers best, brain round-trip, `LLMEmitter` with `MockLLM`.
- Paper: `fig_brain_arch` (accuracy by architecture + search/leaderboard + ablation)
  + `table_brain_arch` + macros, `EXPERIMENTS += e59_brain_arch`,
  `\NumExperiments` 57→58, `sec:brainarch` subsection.
- One branch off `main` (`e59-brain-arch`), one PR. (3-spot count collision with any
  other open experiment PR resolved at merge per CLAUDE.md.)

## Honesty
The deterministic claim uses a *modeled* backbone with a fixed capability — a
controlled demonstration that architecture matters with constant capability, not a
benchmark of a specific LLM. The live Ollama run shows the real text-in/text-out
brain works end to end.
