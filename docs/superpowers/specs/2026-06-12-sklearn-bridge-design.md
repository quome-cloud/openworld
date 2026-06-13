# Dev-rel: OpenWorld for scikit-learn users

**Date:** 2026-06-12
**Status:** approved

## Goal

Lower the conceptual on-ramp for ML/scikit-learn practitioners who struggle
with "worlds" and "oracles." A dedicated bridge that maps OpenWorld concepts
to ML concepts they already own, attacks the three hardest confusions, and
proves the mapping with runnable side-by-side code.

## Deliverables

1. **`tutorials/from_scikit_learn.md`** (~140 lines, tutorial house voice):
   - *Lead*: "you know fit/predict/score; here's the same workflow in a new
     shape."
   - *Rosetta table*: World ~ environment / data-generating process;
     `transition` ~ the function you would fit; **oracle ~ held-out
     ground-truth labels**; `world.compile()` ~ `fit` (LLM writes code, not
     weights; no gradient descent); `world.step()` ~ `predict`;
     verification (sandbox + invariants + critic) ~ property-based tests /
     assertions, NOT a held-out score; `Dial` ~ an inference-time loss
     weight / hyperparameter; `Objective` ~ a scoring function / metric.
   - *Three myth-busters* (the chosen pain points), each "you'd expect X
     (sklearn) -> here it's Y -> because":
     1. **A world is not a dataset** -- a world generates transitions on
        demand from state+actions+rules; there is no fixed (X, y) table.
     2. **What an oracle is and why** -- ground truth you compare against to
        verify the synthesized dynamics (like checking predictions against
        held-out labels); you write one when you can, and when you cannot,
        verification leans on invariants + your own probes. (Hardest concept;
        give it the most space.)
     3. **Verification is not test accuracy** -- gates assert the *program*
        is correct-by-construction (runs, respects invariants, a critic
        agrees); closer to `assert`/property tests than a number on a test
        set.
   - *Closing*: when to reach for each paradigm (declarable dynamics ->
     OpenWorld; pattern-from-data with no declarable rule -> stay in sklearn),
     and a pointer to the five domain tutorials.
2. **`tutorials/from_scikit_learn.py`** (runnable offline, ~120 lines):
   - Left: an sklearn-style `fit`/`predict` attempt to *model an environment's
     dynamics from sampled transitions* (a small regressor), shown missing
     out-of-range inputs -- the failure ML folks recognize.
   - Right: the **same** task the OpenWorld way -- declare the world, get
     dynamics (`MockLLM`/hand-written `FunctionTransition` offline; live
     `compile()` when an Ollama model name is passed), `step()`, and **verify
     exact against the oracle** including out-of-range inputs.
   - Prints a side-by-side summary so the contrast is felt. Guards the
     sklearn import (skips that arm with a note if absent). Asserts the
     OpenWorld path matches the oracle (the script is its own test).
3. **Glue:**
   - `tutorials/README.md`: a "New to worlds? **Start here ->**
     [OpenWorld for scikit-learn users](from_scikit_learn.md)" banner above
     the table; a table row; the run command in the block.
   - A one-line pointer at the top of each of the five existing `.md`
     tutorials: "New to worlds/oracles? See [OpenWorld for scikit-learn
     users](from_scikit_learn.md) first."

## Constraints

- Offline, deterministic, no new framework dependency (sklearn guarded,
  experiment/tutorial-only); runs with no Ollama via MockLLM/FunctionTransition.
- Public APIs only (`World`, `FunctionTransition`, `Action`, `MockLLM`,
  `OllamaLLM`, `Objective`, `Dial`). No pytest addition (tutorials aren't in
  the suite; the script self-checks via asserts).
- Accurate to the framework: every Rosetta mapping must be technically
  correct, not a marketing analogy.

## Out of scope

Rewriting the five existing tutorials; video/interactive content; changing
framework APIs.
