# E43 - Active world-model induction (warm-up)

**Date:** 2026-06-13
**Status:** approved

## Goal

Turn E38's identifiability ceiling into a positive result: an agent that ACTS
to disambiguate unknown dynamics identifies the exact rule in far fewer
transitions than passive random observation. Acting, not scaling, closes the
gap. Warm-up before the emergent-economy capstone (E44).

## Setup

The E37/E38 sprint world, with its dynamics drawn from a known **candidate
family** parameterizing the unknowns: the ship effects (debt delta), the
subtle interaction `bugs += debt // k` (the divisor `k` is the hard one - it
only manifests when debt is high enough and a ship occurs), and the
fix/refactor magnitudes. The true rule is one point in this grid; the agent
must identify it by gathering transitions. The candidate family + the
consistency check (a candidate is eliminated when its predicted next-state
disagrees with an observed transition) make identification exact and
deterministic.

## Conditions (all eliminate candidates by consistency)

1. **passive_random** - acts under a random policy (the E37/E38 setting). The
   disambiguating states (high debt + ship, to reveal `k`) are rare, so it
   plateaus - reproducing the identifiability ceiling.
2. **active_versionspace (ours)** - maintains the set of still-consistent
   candidate rules and, at each step, picks the action that most splits the
   remaining set at the current state, tie-breaking toward state-building
   actions (ship) so it drives toward the rare disambiguating states. Optimal-
   experiment-design / candidate-elimination.
3. **oracle_floor** - a hand-designed minimal probe sequence that uniquely
   pins the rule; the information-theoretic lower bound on steps.

## Metric

- **transitions-to-unique-identification**: steps until exactly one candidate
  remains (and probe accuracy on held-out states -> 1.0);
- remaining-candidate-count vs steps (the elimination curve);
- swept over several hidden true rules so it is not one lucky target.

Hypothesis: active identifies in far fewer transitions than passive (often
near the oracle floor); passive plateaus, frequently never resolving `k`
within the budget.

## Honest boundary

Active identification requires (a) the true rule lies in the searched family,
and (b) the agent can REACH the disambiguating states (controllability). If a
distinguishing state is unreachable, even active cannot resolve it. Stated in
the writeup. The active policy is a greedy/elimination heuristic, not provably
optimal; it is compared against the oracle floor to show how close it gets.

## Deliverables

- `experiments/e43_active_induction.py` (deterministic, offline, self-checking)
  + results JSON. Reuses the sprint ground truth.
- Figure: candidates-remaining vs transitions (active vs passive vs oracle
  floor) and mean steps-to-identify; Results subsection extending E37/E38;
  NumExperiments bump.

## Out of scope

LLM-proposed probes (deterministic version-space for the headline; a future
arm); continuous/parametric-real rule families (discrete candidate grid here);
multi-rule simultaneous identification beyond the sprint family.
