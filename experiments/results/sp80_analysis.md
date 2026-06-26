# sp80 Synthesis Analysis: Multi-Zone Painting Mechanic

**Game**: sp80 (ARC-AGI-3 public benchmark)
**Date**: 2026-06-26
**Experiment**: E87 verified-code world model track

## Finding

Two synthesis rounds (Prism via DM wire, M56677 + M56696) converged to 0.64 held-out
accuracy, down from the original 0.81. The regression isn't a worse model — it's the
held-out set crossing a level boundary that the hardcoded zone assumption can't handle.

## Game Mechanics (Inferred)

sp80 has three interacting mechanics:

1. **Entity movement** (value=9, 4×N block): actions 1–4 move the entity by 4 cells;
   action 5 is a no-move/erase trigger; action 6 is pass.

2. **Row-0 trail**: row 0 tracks the entity's column history. Each step removes 2 (usually)
   or 3 (at specific positions: C=40, C=57, C=25, C=8) cells from the trail's right edge.

3. **Painting zone**: actions 3/4/5 trigger a "paint/erase" mechanic on a horizontal band
   of rows. The zone changes by level:
   - **Level 1**: rows 20–23 (4 rows × entity-width cols)
   - **Level 2+**: rows 8–15 (8 rows × entity-width cols) after level completion + reset

## Root Cause of Synthesis Failure

Random exploration from seed=0 completed level 1 mid-collection (step ~200–225, best_levels=1).
After the win, the env reset and the entity appeared in a new level with the painting zone
at rows 8–15 instead of rows 20–23. The held-out transitions (steps 225–300) are entirely
from this second-level zone.

Both Prism candidates hardcode `rows 20-23` because all 12 demo examples shown were from
level 1. The second-level zone was never shown to the synthesizer.

## Quantified Impact

| Failure type | Count | Root cause |
|---|---|---|
| Row 20-23 (partially wrong) | 6 | Column range ambiguity (new vs old entity cols) |
| Row 0 off-by-one | 4 | 3-cell positions not fully enumerated |
| Rows 8-15 (zone mismatch) | 19 | Level 2 painting zone never shown in demos |

## Implication

Same pattern as dc22: **transitions must span all mechanically-distinct states for
the synthesizer to model them.** For sp80, the painting zone is a level-dependent
parameter. A synthesizer shown only level-1 data cannot infer level-2 behavior.

Fix options:
- Collect transitions from multiple level cycles (1000+ steps, or explicitly trigger resets)
- Show the synthesizer transitions from BOTH zones and let it generalize the level-detection rule
- Expose `levels_completed` signal in the transition record so the synthesizer can condition on level

## Connection to dc22

Both sp80 and dc22 hit the same failure mode: **correct in-distribution accuracy,
incorrect out-of-distribution behavior** due to coverage gaps from random exploration.
dc22's gap was unreachable mechanics (teleporters); sp80's gap is unseen level states.
These are the same architectural failure expressed differently.
