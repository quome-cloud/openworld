# OpenWorld for scikit-learn users

> Script: [`from_scikit_learn.py`](from_scikit_learn.py) — runs offline; shows
> the scikit-learn way and the OpenWorld way side by side.

If you come from scikit-learn, OpenWorld can feel like it's using familiar
words (model, predict, score) for unfamiliar things — and two words in
particular, **world** and **oracle**, tend to trip people up. This guide maps
every concept to one you already own, then clears up the three confusions that
cause the most friction. You already know the loop: `fit` a model to data,
`predict`, check a `score`. Hold onto that — most of it transfers, with one
twist that changes everything.

## The one-sentence version

In scikit-learn you **learn an unknown function from data**. In OpenWorld you
**declare a known function as rules and have an LLM write it as verified code**.
Same goal — a thing that predicts what happens next — reached from the opposite
direction.

## Rosetta table

| You know (scikit-learn) | OpenWorld | The mapping |
|---|---|---|
| environment / data-generating process | **`World`** | state + actions + rules. It *generates* transitions on demand; it is **not** a dataset (see below). |
| the function you `fit` | **`transition`** | maps `(state, action) → next_state`. In sklearn you learn it; here you declare it and the LLM writes the code. |
| held-out ground-truth labels | **oracle** | the correct answers you score against to know your model is right. |
| `model.fit(X, y)` | **`world.compile()`** | produces the transition. But the LLM writes a *program* from the rule text — no weights, no gradient descent, no epochs. The "training data" is your description. |
| `model.predict(X)` | **`world.step(action)`** | advances the state one tick. Exact and instant — it's running code. |
| `cross_val_score` / test accuracy | **verification** (sandbox + invariants + critic) | correctness **gates** on the program, like property tests / `assert`s — not a number on a held-out set (see below). |
| a hyperparameter / loss weight | **`Dial`** | a knob, except you can turn it at *inference* time to move along a trade-off, with no retraining. |
| a metric / scorer | **`Objective`** | a scoring function over trajectories. |
| `Pipeline` of estimators | **`CompositeWorld`** | small worlds composed into a big one, coupled by explicit bridges. |

## Three things that surprise scikit-learn users

### 1. A world is not a dataset

Your instinct is to look for `X, y`. There isn't one. A `World` is closer to an
OpenAI-Gym environment than to a CSV: it holds the current `state`, the legal
`actions`, and the `rules`, and it *produces* the next state when you `step`.
There is no fixed table of rows you fit on — transitions are generated on
demand, as many as you want, exactly, for free. If you've used a simulator or
an RL environment, that's the right mental model; if you've only used
`fit(X, y)`, the shift is: **the data source is a program, not a file.**

### 2. What an oracle is, and why you keep hearing about it

This is the concept that confuses people most, and it's simpler than it sounds.

An **oracle is ground truth** — the same role as the held-out labels you score a
classifier against. When you write `accuracy_score(y_test, y_pred)`, `y_test`
*is* an oracle: the correct answers you compare your model's answers to. In
OpenWorld the oracle is a hand-written reference implementation of the
dynamics — the `transition` you *know* is correct — and you use it the same
way: to check that the LLM-synthesized code agrees with it.

Two honest clarifications:

- **You don't always have one, and that's fine.** When you can write the
  reference dynamics, you get the strongest check (bit-exact agreement on
  probes you choose — exactly what [`from_scikit_learn.py`](from_scikit_learn.py)
  does). When you can't, verification leans on the other gates — invariants you
  *can* state ("inventory never negative") and your own spot-check probes — the
  way you'd sanity-check predictions without a full labeled test set.
- **The oracle is for evaluation, never for serving.** Just as you'd never ship
  `y_test` as your model, the oracle is the yardstick that tells you the
  synthesized program is trustworthy; the *program* is what runs in production.

### 3. Verification is not a test score

When sklearn says "92% accuracy," you accept some errors and move on.
OpenWorld's verification is categorically different: it asks "is this program
*correct*?" and answers with gates, not a number. Sandboxed smoke-runs prove it
executes; invariants prove it never violates a stated property; an optional
second-model critic reads the code against the rules. This is much closer to a
**property-based test suite** or a wall of `assert`s than to `cross_val_score`.
The payoff is the headline result you can see in the script: because the
accepted artifact is *code that encodes the rule*, it is exact at any input
scale — where a fitted regressor, having learned only the magnitudes it saw,
collapses out of distribution.

## See it run

[`from_scikit_learn.py`](from_scikit_learn.py) models one tiny environment — a
water tank with the rule *fill +2, drain −3 (never below empty)* — both ways
and scores each against the oracle:

```
approach                           in-distribution   10x out-of-dist
scikit-learn (fit a regressor)               100%                0%
OpenWorld (declare + verify code)            100%              100%
```

The regressor memorizes the levels it trained on and is helpless at 10× scale;
the declared-and-verified dynamics are the *rule*, so they're exact everywhere —
and verification proved that before a single step was trusted.

## When to reach for which

- **A pattern you can only learn from data**, with no rule you could write down
  (image classes, demand from messy history) → that's scikit-learn's job; keep
  using it.
- **Dynamics you can describe** ("the queue grows by arrivals minus throughput,
  clamped at zero") → declare them and let OpenWorld write and verify the code.
  You get exactness, OOD robustness, speed, and an auditable artifact, with no
  training run.

Ready for a real one? The five domain tutorials build from here — start with
[healthcare triage](healthcare_triage.md) (the gentlest synthesis walkthrough)
or [software engineering sprint](software_engineering_sprint.md) (which shows
`compile()` synthesizing dynamics and *verifying them against an oracle*, the
full version of what you just saw).
