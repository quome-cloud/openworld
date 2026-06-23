# OpenWorld → ARC-AGI-3: the recipe to move the needle

**Goal.** Beat the near-zero frontier baseline on ARC-AGI-3 by treating each game as a *world*
whose dynamics OpenWorld can **synthesize as verified code** and **plan** through — the same wager
that makes symbolic code world models exact elsewhere, now on an interactive benchmark.

## Why this should work (preconditions, measured in the PoC)
- **Deterministic dynamics.** On `ls20`, every repeated `(state, action)` reproduced the next
  frame exactly (determinism = 1.00 over 100+ repeats) → the dynamics are *writable as exact code*.
- **Dense state, sparse change.** ~1,500 / 4,096 cells are non-background, but only ~50 change per
  step. A code model handles this natively (local update rules); a token-level next-frame learner
  chokes on ~10k-token prompts.
- **Headroom.** Random and change-seeking heuristics complete 0/7 `ls20` levels in 400 steps.

## The recipe (one agent, four stages)
1. **Explore.** Budgeted policy (random → change-seeking → model-guided) collects verified
   `(s, a, s')` transitions. The env is ground truth, so every label is exact.
2. **Synthesize verified code.** Prompt an LLM with a sample of transitions (compact: background
   color + changed-cell deltas + action) to emit `predict(frame, action) -> next_frame`.
   **Accept only if it exact-matches held-out transitions** (per-frame verification gate). On
   failure, feed back counterexamples and retry (plan–generate–verify relay). Track the
   *branch/transition verification rate*.
3. **Plan.** With the verified model, search action sequences (BFS / beam / CEM lookahead) for
   trajectories that raise `levels_completed` or reach the win state — planning in imagination
   through exact code at native speed.
4. **Act + re-sync.** Execute the planned actions; if a real transition ever disagrees with the
   model (verification miss in a new region), add it and re-synthesize.

## Experiments
- **E86 — verified-model fidelity (foundation).** Per game: synthesized-code exact-match rate on
  held-out transitions vs. learned baselines (copy-previous-frame; 1-NN over transitions; small
  MLP). Claim: code is exact / near-exact where learned models compound error. *(determinism makes
  exactness attainable, not just better.)*
- **E87 — planning / level completion (payoff).** Levels completed under: random, change-seeking,
  learned-model planner, and the verified-code planner. Across the 25 public games.
- **E88 — generalization (the landmark).** Does a synthesizer prompted/fine-tuned on transitions
  from *held-in* games write correct code faster / for *held-out* games (cross-game transfer)? And
  the stretch: do ARC-1/2 verified transformations as extra held-in worlds help (cross-generation)?

## Baselines to report (honesty)
- random, change-seeking (done: 0/7 on ls20),
- learned next-frame (copy / 1-NN / MLP) + its planner (expected to compound error),
- frontier LLM agent acting directly (if budget allows), for context.

## Metrics
- transition verification rate (exact next-frame match, held-out),
- levels completed / win rate per game (the scorecard),
- synthesis cost (LLM calls to reach a verified model),
- planning depth vs. success.

## Scope / risks (state them)
- ARC-3 games are *designed to be individually novel* → cross-game generalization (E88) may be a
  null; a clean null is still a finding.
- Some games may have stochastic or perceptual elements that break determinism → report per-game.
- Full dynamics may be too complex for one-shot synthesis → partial models + planning under
  uncertainty is the fallback.

## Status
- ✅ Integration (local play, gym-like, no key), determinism precondition, baselines (PoC).
- ▶ E86 harness (this folder's experiments): env + transition logs + verification + synthesis loop.
- ☐ E87 planner, E88 generalization.
