# E46 - A database for many worlds: factored, semiring-annotated world store

**Date:** 2026-06-13
**Status:** approved (design); pending spec review

## Goal

E43 maintained a version space by holding an EXPLICIT list of candidate worlds
and eliminating the inconsistent ones. That works for hundreds of worlds; it
cannot scale as parameter counts grow and ranges widen, when the world space is
combinatorial-to-effectively-infinite. E46 builds the data structure that does:
a **factored, semiring-annotated store** over the world-parameter space that
maintains an exact version space / posterior and answers traversal queries
("is this world possible?", "how many worlds remain?", "what is the expected
next state across all consistent worlds?") in time **linear in the sum of the
parameter-domain sizes, not their product** - i.e., over world spaces far too
large to enumerate.

This is the database design from the many-worlds discussion: provenance
semirings (one engine, many query types) over conditional-table / factor-graph
structure (finite symbolic representation of an infinite world set).

## The representation

- **Parameters** `P_i`, each a variable with a finite domain `D_i` (a continuous
  parameter is discretized). A *world* is a full assignment; the world space is
  `prod_i |D_i|` - astronomically large as parameters/ranges grow.
- **Mechanisms**: a world model's transition decomposes into named mechanisms,
  each computing one observable of the next state from the current (observed)
  state and a small **scope** of parameters. For the E43 sprint family:
  - `debt_on_ship` (scope `{ship_debt}`): on ship, `debt += ship_debt`
  - `bugs_on_ship` (scope `{k}`): on ship, `bugs += debt // k`  (uses observed debt)
  - `bugs_on_fix`  (scope `{fix}`): on fix, `bugs -= fix`
  - `debt_on_refactor` (scope `{refactor}`): on refactor, `debt -= refactor`
- **Factors**: the store keeps, per mechanism scope, a factor mapping a
  scope-assignment to a **semiring value**. A factor has size `prod` over its
  (small) scope domains, not the global product. Initially every value is the
  semiring one.

The key fact that makes this exact: because each transition is OBSERVED in full
(state, action, next_state), the likelihood factorizes over observables, and
each observable constrains only its mechanism's small scope. Updating on an
observation touches only the relevant small factor(s); the global posterior is
the product of the factors and is never materialized.

## Semirings (one engine, many queries)

The factor values live in a pluggable semiring; swapping it changes the question
with no change to the update logic:

- **Boolean** (`or`/`and`): version space - is a (scope-)assignment still
  possible? `count()` of survivors per factor.
- **Counting** (`+`/`*`, non-negative ints): exact number of globally consistent
  worlds = product over independent factors of their surviving counts.
- **Probability** (`+`/`*`, normalized): a posterior over parameters; marginals
  per factor; expected next-state per observable.

This is the provenance-semiring contribution made concrete: evaluate the
structure once, instantiate per query type.

## Operations and their cost

- `observe(state, action, next_state)`: for each observable, eliminate / reweight
  the scope-assignments that do not reproduce the observed observable. Cost:
  `sum` over touched scopes of their (small) sizes. Independent of the global
  world count.
- `count()` / `is_possible(world)` / `marginal(param)`: read from factors.
  `count()` is a product of per-factor survivor counts.
- `predict(state, action)`: distribution (or expectation) over next_state,
  obtained by marginalizing each observable over its scope factor only - cost
  independent of the global world count (**sub-linear in worlds**).

## Conditions / comparison

- **factored (ours)** - the store above.
- **enumerated (E43 baseline)** - the explicit candidate-list version space.
  Feasible only up to ~`1e6`-`1e7` worlds; used to certify correctness and to
  show the wall it hits.

## Claims and self-checks

1. **Correctness**: on small world spaces where enumeration is feasible, the
   factored store's answers (consistent-count, per-parameter marginals,
   `predict` expectation, `is_possible`) match brute-force enumeration **exactly**
   (asserted, several hidden true worlds).
2. **Scale**: the factored store maintains an exact version space + posterior over
   a world space of size `>= 1e18` in milliseconds and small memory, where
   enumeration is infeasible (we report the largest N enumeration can reach
   before blowing up, and that factored is flat past it).
3. **Sub-linear query**: `predict` / `count` time is ~flat as the global world
   count grows by many orders of magnitude; enumeration grows linearly then
   fails. Reported as a time-vs-N curve.
4. **Semiring generality**: the same store answers Boolean (possible?),
   counting (#worlds), and probabilistic (posterior) queries; demonstrated and
   cross-checked against enumeration on small N.
5. **Honest boundary (coupling / tree-width)**: when a mechanism's scope couples
   many parameters non-separably, its factor grows as the product over that
   scope; as coupling width `w` increases, the factored advantage degrades toward
   enumeration. We include a coupled variant and report the degradation curve -
   the exact analogue of #P-hardness for probabilistic databases. Not hidden.

## Deliverables

- `openworld/manyworlds.py` - the semiring abstraction + factored `WorldStore`
  (strictly additive new module; exported from `openworld/__init__.py`).
- `tests/test_manyworlds.py` - unit tests incl. factored-vs-enumerated equality.
- `experiments/e46_many_worlds.py` (+ `results/e46_many_worlds.json`),
  deterministic/offline/self-checking.
- Figure (scale: query/update time + worlds-remaining vs N, factored flat vs
  enumeration exploding; plus the coupling-width degradation) + table; paper
  subsection generalizing E43 to non-enumerable world spaces; `\NumExperiments`
  bump (reconcile with any concurrently-landing experiment at merge time).
- PR (based on the E44 branch).

## Honest boundaries

- The exact factorization requires the transition to decompose into
  small-scope mechanisms given the observed state; densely coupled dynamics
  (claim 5) cost more, up to enumeration. We quantify rather than assume.
- Discretizing a continuous parameter is an approximation of the continuum; the
  store is exact over the chosen discretization, and we report resolution.
- This is a representation/efficiency result (how to *hold and query* many
  worlds), complementary to E43's policy result (how to *choose actions* to
  shrink them); together they are "active inference over a non-enumerable world
  space."
