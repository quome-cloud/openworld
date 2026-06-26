# dc22 Coverage Analysis: Why 1.000 Held-Out Accuracy Doesn't Guarantee Solvability

**Game**: dc22 (ARC-AGI-3 public benchmark)
**Date**: 2026-06-26
**Experiment**: E87 verified-code world model track

## Finding

We synthesized a `predict(frame, action) -> frame` model for dc22 via Prism (A004, M56643).
The model achieves **held-out accuracy = 1.000** on 50 test transitions.
Despite this, BFS planning over the model finds **no winning action sequence** — the scorecard
score remains 0.

## Root Cause: Physical Disconnection in the Maze

Random exploration (300 steps, seed=0) visited only a **9-position patch** of the playfield:
rows 38–43, cols 8–13. The 2×2 controllable entity (value=14) can reach every cell in this
patch via value-2 corridors. No value-2 corridor connects this patch to the goal cell.

The **goal** (value=11) sits at rows 20–21, cols 24–25. The only plausible path between the
start zone and the goal runs through teleporter sprites ("tewfut" cells), which appeared in
the frame but were never activated during random exploration.

Because no training transition ever touched the teleporter mechanic, the synthesized model
has zero representation of it. A BFS search over the model cannot plan through a transition
type that does not exist in the model.

## Why 1.000 Doesn't Mean "Complete"

Held-out accuracy measures **in-distribution** performance: how well the model predicts
transitions similar to what was collected. It does not measure **coverage** of the full
game mechanic space.

In this case:
- Training transitions = {movement in accessible zone, step counter increment}
- Missing mechanics = {teleporter activation, goal-state trigger, cross-zone movement}

A model with 1.000 accuracy on the collected distribution is a *correct* model of what it
saw, not a *complete* model of the game. These are only equal when exploration was exhaustive.

## Quantified Coverage Gap

| Metric | Value |
|--------|-------|
| Total playfield area | 64×64 = 4096 cells |
| Entity-accessible cells (visited) | ~9 positions × 4 cells = 36 cells |
| Goal cell reachable by corridors | No |
| Teleporter cells in frame | Present (visually) |
| Teleporter activations in training | 0 |
| Transitions where levels_completed went 0→1 | 0 |

## Implication: The Next Architectural Step

Random exploration is fundamentally limited for games with sparse rewards and gated mechanics.
The dc22 result implies the next architectural step is **exploration guided by the synthesizer**:

1. Synthesize a partial world model from whatever random play collected.
2. Ask the synthesizer to *identify* underexplored mechanic signatures in the current model
   (sprites with value ranges never seen in transitions, cells never visited by the entity).
3. Generate targeted exploration actions that activate those mechanics.
4. Augment training data, re-synthesize, repeat.

This is closer to Jim's ARC-AGI-3 agent pattern (LLM-in-the-loop planning), where the
LLM is reasoning interactively rather than one-shot from static data. Jim's 19/25 solutions
suggest this loop works end-to-end; our dc22 result explains mechanically *why* one-shot
synthesis hits a ceiling even with perfect in-distribution accuracy.

## Broader Claim

> **1.000 held-out accuracy on a verified-code world model is necessary but not sufficient
> for game solvability.** The binding constraint is whether random exploration visited
> all mechanically-distinct regions of the state space.

This is a publishable observation for the ARC-AGI-3 track. It also explains the
disconnect between our scorecard (0) and Prism's synthesis quality (perfect), without
implying any deficiency in either the synthesizer or the verifier.
