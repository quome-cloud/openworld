# E89: ARC-AGI-2 via LLM-Written Python — Burst-1 Results

**Branch**: `exp-e89-arc-agi2-python-synth`
**Date**: 2026-06-26
**Experiment**: E89 — verified-code synthesis with arbitrary Python on ARC-AGI-2

---

## Setup

**Pipeline**:
1. Load ARC-AGI-2 evaluation task (demo pairs + test input)
2. Send to synthesis agent via DM protocol (arc_synthesizer_harness.py + arc_synthesizer_protocol.md)
3. Agent writes `transform(grid) -> grid` as arbitrary Python (no DSL constraints)
4. Exact-match verify gate: accept only programs passing ALL demo pairs
5. Vote across verified candidates; apply to test input

**Synthesis agent**: claude-sonnet-4-6 (routed via DM protocol, M56495–M56557)
**Candidates per task**: 3
**Tasks evaluated**: 30 (ARC-AGI-2 evaluation set)
**Verify gate**: strict exact-match on all demo pairs (no partial credit)

---

## Results

| Metric | Value |
|--------|-------|
| Tasks evaluated | 30 |
| Candidates per task | 3 (90 total) |
| Candidates passing verify gate | 1 (1.1%) |
| Tasks solved (ground truth match) | 0 / 30 (0%) |

**Key finding**: 89/90 candidate programs failed to exactly reproduce all demo pairs. The one candidate that passed the verify gate did not match the ground-truth test output.

---

## Comparison: E84 vs E89

| | E84 (Fixed DSL) | E89 (Arbitrary Python) |
|--|--|--|
| Representation | 17-primitive geometric DSL | Unrestricted Python |
| Synthesis approach | DSL program search | LLM-written arbitrary code |
| ARC-AGI-2 score | 0% | 0% |
| Bottleneck | Coverage gap (missing primitives) | LLM program correctness |

**Conclusion**: Removing the DSL constraint (E84's hypothesized bottleneck) does not improve ARC-AGI-2 accuracy. Both approaches score 0% on this sample.

This **refines E84's analysis**: E84 attributed its 0% to DSL coverage gaps (no connected-component, object-identity, or counting primitives). E89 removes those constraints entirely — yet still scores 0%. This suggests the binding bottleneck is not the DSL straitjacket but the LLM's ability to derive correct transformation rules from a small number of examples, possibly within a single prompt.

---

## Failure Mode Analysis

Out of 90 candidates, only 1 passed the exact-match verify gate (1.1% gate-pass rate). This is the primary failure mode: the synthesized programs do not correctly reproduce the demo-pair transformations, let alone generalize to the test input.

Candidate programs showed plausible but incorrect approaches — typically:
- Correctly identifying the structural pattern type (fill gaps, connect boxes, color-key lookup)
- Incorrectly implementing the edge cases, directionality, or selection logic

The exact-match gate is unforgiving: any cell wrong = reject. ARC-AGI-2 tasks are designed to require precise rule inference, and programs that "approximately" capture the pattern still fail.

---

## Honest Scope

- **Sample size**: 30 tasks from the ARC-AGI-2 evaluation set. Results may not generalize to the full set.
- **Candidates**: 3 per task. Higher N might recover some tasks (see: stochastic synthesis).
- **Model**: claude-sonnet-4-6. A stronger model or multi-turn refinement might improve gate-pass rate.
- **No held-out set**: All 30 tasks are from the public evaluation set. No contamination control applied.

The purpose of this experiment is a clean comparison data point vs E84, not a state-of-the-art score.

---

## Next Steps (if continuing)

1. **More candidates**: Run N=32 (vs N=3) to increase gate-pass probability per task. Theory: if gate-pass rate is ~1%, N=32 gives ~28% expected coverage.
2. **Multi-turn refinement**: Feed verify-gate failures back to the LLM with error context.
3. **Stronger model**: Try claude-opus-4-8 on the tasks where 3 candidates all failed immediately.
4. **Held-out evaluation**: To claim a defensible score, use the ARC-AGI-2 private evaluation set.

---

## Summary

E89 establishes that replacing a fixed DSL with arbitrary LLM-written Python does not unlock ARC-AGI-2 accuracy at claude-sonnet-4-6 with N=3 candidates. The binding constraint appears to be rule-inference correctness under the few-shot regime, not representation coverage. This closes the E84 loop: the problem is harder than "just give it more primitives."
