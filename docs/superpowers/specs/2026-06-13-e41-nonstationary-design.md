# E41 - Non-stationary dynamics: detecting and adapting to sudden rule changes

**Date:** 2026-06-13
**Status:** approved

## Goal

Test how fast different algorithms pick up on an UNANNOUNCED, sudden change in
a world's rules, fed through the perception boundary. E32 showed a *known*
regime switch is handled exactly by a pre-verified PhasedTransition; E41 asks
the harder question: when the change time is hidden, who detects it and how
fast do they recover?

## World

A parametric reservoir/queue with a hidden regime parameter that governs its
one-step update. The episode is a long horizon (T=120 steps) of one-step-ahead
prediction, with STABLE stretches punctuated by 2 SUDDEN unannounced changes
(near step 40 and step 80) to a different in-family rule. Dynamics are exact
declarable code; the regime parameter is what jumps.

## Perception

Each step's true state is observed through the perception boundary (a
deterministic perceptor; optional perception noise). All predictors consume
the perceived symbolic state, so the setup composes with E39/E40 (under noise
the E39 decomposition adds on top).

## Algorithms (one-step-ahead prediction each step)

1. **static-frozen** - fit on the first regime, never updates.
2. **online sliding-window (W)** - refit on the last W transitions each step.
3. **symbolic monitor + refit (ours)** - predict with the current rule; on a
   prediction-error spike sustained k steps, declare a changepoint, re-identify
   the rule by deterministic search over the candidate family, verify it
   reproduces the recent window, switch. Snaps back to exact once identified.
4. **oracle-switch** - knows the change times, switches instantly (0-lag
   ceiling).

## Metrics

- per-step exact-match error timeline (with change markers);
- recovery lag per change (steps from change to error returning to its
  post-recovery floor);
- cumulative regret (total wrong predictions over the episode);
- post-change steady error per method.

Hypothesis: symbolic recovers to EXACT within a few steps; sliding-window
recovers slowly to approximate; frozen never recovers; oracle = 0 lag.

## Deliverables

- `experiments/e41_nonstationary.py` (deterministic, offline, self-checking)
  + results JSON.
- Figure: error-over-time timeline (4 methods + change markers) and a
  recovery-lag / regret comparison; Results subsection; NumExperiments bump.

## Honest boundary

Symbolic snap-to-exact holds only when the changed rule remains within the
candidate family the refit searches (a declarable change). A change to an
undeclarable rule yields no exact recovery - the same paradigm boundary as the
rest of the framework. Stated in the writeup.

## Out of scope

Live-LLM re-synthesis (deterministic refit for reproducibility; a --model
option may be added later); gradual/drift changes (this is sudden-change);
multi-rule simultaneous changes.
