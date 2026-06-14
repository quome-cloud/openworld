# E49 - Path integrals over composite-world learning trajectories

**Date:** 2026-06-14
**Status:** approved (design)

## Goal

Bridge E46 (many-worlds store) with path integrals: given an agent spec (initial
capabilities) and a new, out-of-distribution target problem, compute the most
direct trajectory through the space of composite worlds/skills to master in order
to solve it. There are combinatorially (or infinitely) many trajectories; the
path integral sums over them weighted by `exp(-beta * action)`, dominated by the
least-action path (the optimal curriculum), computed WITHOUT enumeration as a
semiring DP - reusing the E46 `Semiring` abstraction with the value ring swapped.

## The bridge

A path integral is a sum over paths = a semiring sum-over-paths. So:
- **tropical** `(min,+)` -> least-action path (optimal curriculum).
- **log** `(logsumexp,+)` -> full path integral / log-partition / free energy.
- **counting** `(+,*)` -> number of trajectories.
- **forward x backward / Z** -> each world's path-integral marginal (what to learn).
Cycles / infinitely many paths are handled by semiring closure; the finite
combinatorial blow-up (orderings) is summed by the DP over capability-states.

## Model

`openworld/pathintegral.py`: `Skill(name, prereqs, compose_cost, scratch_cost)`
(cheap to compose once prereqs mastered, dear from scratch) and `TrajectorySpace`
(forward/backward DP over capability-states; least_action_path, node_marginals,
partition, count_trajectories). Additive; reuses `openworld.Semiring`.

## Experiment (`experiments/e49_path_integral.py`)

Abstract skill library with a corporate OOD target (turn around a stalling
division - cf. E48); agent specs (senior SWE / director / CEO) as initial
capabilities. Results: (1) agent spec -> least-action curriculum + cost per role;
(2) least action beats unplanned baselines (random / eager / greedy); (3)
compositional transfer (path cost vs from-scratch); (4) path-integral marginals +
free energy approaching least-action as beta grows; (5) trajectory count vs DP
states (tractability). Deterministic/offline/self-checking.

## Self-checks

least-action <= greedy and < random/eager for every agent; transfer cost <
from-scratch; free energy -> least-action as beta grows; marginals normalized and
=1 for mandatory worlds; trajectory count > DP states.

## Boundaries

- Stylized cost model (compose vs scratch); the claim is about the structure
  (optimal curriculum, transfer, the semiring bridge), not calibrated learning
  times.
- For a single-route dependency chain the per-world marginals are all 1 (every
  world mandatory); the beta-dependence then lives in the ordering entropy / free
  energy, not in node marginals. Alternative-route (OR-prereq) spaces would make
  node marginals beta-sensitive too; out of scope here.

## Deliverables

`openworld/pathintegral.py` + exports + `tests/test_pathintegral.py`;
`experiments/e49_path_integral.py` (+ results); figure + table + paper subsection
`sec:path-integral`; `\NumExperiments` -> 47. PR based on and targeting `main`.
