# E142 — Generalization-critique subagent (anti-overfitting world-model review)

A harness step that makes the source-free coding agent **adversarially review its own world model for
overfitting before banking a new deepest level**, directly targeting the binding wall: a world model
that reproduces levels 1..N but does not GENERALIZE to level N+1 (E136's H1).

**Provenance / integrity.** The *workflow* is inspired by the public baseline1 run
(arXiv 2605.05138), whose harness ran a "generalization-critique subagent" as adversarial review. We
adopt only the **methodology** — a generic skeptic that hunts for hard-coding, ontology drift, and
next-level fragility. We do **NOT** use baseline1's per-game world-model content or any inferred
mechanics (that would be banked solutions). The critique prompt (`critique.md`) is our own wording,
contains zero game-specific content, and explicitly forbids reading source or banked solutions, so the
source-free + solution-free attestation is preserved.

**How it wires in.** `scripts/run_arc_agent_ewm_toolkit.sh` injects a `CRITIQUE PROTOCOL` step into the
agent's TASK.md: after the agent forms or refines a level's world model / win hypothesis and *before*
it banks a new deepest level, it must run the generalization critique (spawn a subagent with
`critique.md`, or self-apply it), then address every serious finding (replace a hard-coded constant
with an inferred rule, unify a drifted ontology, remove a known-layout dependency) before proceeding.

**Why it should help.** baseline1's critique loop is part of why its models transferred across the
curriculum. Our agent currently rebuilds more ad-hoc per level; forcing an explicit "what breaks on the
next level?" pass should raise N→N+1 transfer (the metric E136 measures) and convert
reproduces-the-level into generalizes-to-the-next.

**Scope (YAGNI).** A prompt/protocol addition + an integrity test. No new dependencies. The effect is
measured by the next source-free sweep (does the critique step raise deepest-level reach vs the
no-critique control), reported honestly either way.
