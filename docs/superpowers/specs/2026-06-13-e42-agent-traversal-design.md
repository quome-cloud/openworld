# E42 - Agents traversing connected worlds with changing rules (capstone)

**Date:** 2026-06-13
**Status:** approved

## Goal

Combine every thread - composite worlds + toll routes (E31), non-stationary
rules (E41), per-world perception boundaries (E39/E40), and the interference
result (E36) - at the AGENT-BELIEF level. Two worlds connected by a toll
route; each has its own changing rules and its own perception boundary; an
agent hops back and forth and tries to track each world's rule. Question: can
an agent generalize/track each world's rule across hops as the rules change,
including changes that happen while it is AWAY in the other world?

## World

A `CompositeWorld` with two reservoir children (A, B) joined by a `Route` with
a toll. Each child has its own non-stationary dynamics: a hidden (target, rate)
regime that jumps over the horizon (per E41). Each child has its own
`Perceptor` (its own perception boundary). The agent dwells D steps in a world,
then `travel`s across the route (paying the toll) to the other, on a fixed
schedule over T steps. Rule changes are scheduled BOTH while the agent is
present in that world (it sees the transient) AND while it is away (it returns
to a silently-changed world). Deterministic ground-truth trajectories per
world advance every step regardless of where the agent is (worlds evolve
whether observed or not).

## Agent-belief models (all consume the same perceived observations)

1. **symbolic per-world monitor (ours)** - a separate symbolic rule-belief per
   world; on arrival predicts with the retained belief, detects a change from
   the transient and re-identifies (the E41 monitor, kept per world). Per-world
   separation => no cross-world interference; away-changes caught on return.
2. **single shared online learner** - one model updated from whichever world
   the agent is in; learning B overwrites its A-belief (E36 interference, now
   driven by traversal).
3. **per-world sliding-window learner** - a separate windowed predictor per
   world; no interference but refills its window each visit / after each change.
4. **oracle** - knows the agent's location and each world's current rule (0
   error ceiling).

## Metrics

- prediction accuracy in the current world over the hop schedule;
- post-arrival recovery lag, split into:
  - return-to-unchanged (world's rule did not change while away),
  - return-to-silently-changed (it did - the hard case);
- cumulative regret;
- interference measure: the shared learner's accuracy on a world immediately
  after a stint in the other vs a per-world model's.

Hypothesis: the symbolic per-world monitor retains each world's rule through
absence and recovers fast from both present- and away-changes; the shared
learner thrashes from interference; the windowed learner lags.

## Deliverables

- `experiments/e42_agent_traversal.py` (deterministic, offline, self-checking)
  + results JSON. Reuses CompositeWorld/Route/travel, MockPerceptor/observe,
  and the E41-style transient monitor.
- Figure: per-world accuracy timeline with hop + change markers, and a
  recovery-lag breakdown (unchanged vs silently-changed return); Results
  subsection; NumExperiments bump.

## Honest boundary

Exact recovery holds only when the changed rule stays in the identifiable
family the monitor searches. "No interference" is a property of separating
beliefs per world (the E36 lesson applied to an agent's memory), not magic.
Two worlds, one route (pairwise composition); more worlds/routes are future
work.

## Out of scope

Agent action/planning/optimization (the agent's task here is rule tracking,
not control); live-LLM belief models; >2 worlds.
