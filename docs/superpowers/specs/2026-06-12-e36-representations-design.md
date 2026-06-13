# E36 — Composition yields better representations than monolithic learners

**Date:** 2026-06-12
**Status:** approved (design)

## Goal

Show, with a controlled offline experiment, that representing a factored
world as a **composite of small worlds** generalizes, resists interference,
and learns from far less data than representing the same world as one
monolithic learner. Extends the learned-baseline narrative (E12/E19) from
single-world *fidelity* to multi-part *representation quality*, and gives the
CompositeWorld machinery a head-to-head against neural networks on the three
properties practitioners actually care about.

## The world: K parametric sectors (reuses the E30 substrate)

A factored economy of `K` sectors. Each sector has local integer state
`(stock, output, waste)` and a deterministic, branchy per-sector rule
(produce: `stock-=cost; output+=gain` when `stock>0`; recycle:
`waste->stock`; decay: `waste+=amt` when `output>thresh`; clamp at 0) with
per-sector coefficients — structurally identical to E30's sectors so the
substrate is shared, but here we *sample transitions and learn*, where E30
*synthesized*. Each sector's local state is bounded to a small grid (stock,
output, waste in `0..G`), so a single sector's transition table is small and
fully coverable; the joint space is `~G^{3K}` — combinatorial.

Action = advance one sector (`sector_i:tick`); the joint transition updates
that sector's slice from its local state and passes the other slices through.
Ground truth = hand-written per-sector update composed across sectors (the
oracle; shared by every condition's scoring).

## Conditions (identical learning machinery where applicable)

- **monolith-mlp** — one 2-hidden-layer numpy MLP (the E12 baseline) on the
  *joint* `(state, action) -> next_state`. Hidden width sized so total params
  ≥ the composite's, so any failure is representational, not capacity.
- **knn1** — 1-nearest-neighbor memorizer on joint transitions (E12 baseline;
  the pure-lookup control — exact on seen joint configs, undefined off them).
- **composite-learned (ours)** — `K` *small* MLPs, one per sector, each
  trained only on its own sector's local transitions, wired through a
  `CompositeWorld` (each child a learned `Transition`). Same MLP code as the
  monolith, just factored input.
- **composite-symbolic (ours, ceiling)** — exact per-sector code composed via
  `CompositeWorld`; zero training transitions. The framework's actual product.

## Three legs (each its own metric, all exact-next-state match vs the oracle)

1. **Compositional generalization.** Training transitions are drawn only from
   a "seen" sub-region (each sector's stock restricted to a training subset);
   test on the full joint grid including sector-value combinations that never
   co-occurred in training. Swept over `K = 2,3,4,5`. Hypothesis: composite
   accuracy is flat-high in `K` (each child saw its full small local space);
   monolith and knn1 accuracy fall as the unseen fraction of the joint grows.
   This is the headline curve.
2. **Interference.** Train the monolith *sequentially* on sector 1's
   transitions, then sector 2's, … then measure accuracy recovered on sector
   1 (catastrophic forgetting in shared weights). The composite stores each
   child separately → measured zero forgetting. Report retained accuracy on
   the first sector after training through all `K`.
3. **Sample efficiency.** Accuracy on the full joint test set vs total
   training transitions `K_tx in {100, 1000, 10000}` (the E12 ladder),
   composite-learned vs monolith. Hypothesis: the composite reaches high
   accuracy with far fewer transitions because each child is a tiny,
   coverable learning problem; the symbolic composite needs zero.

## Implementation

- `experiments/e36_representations.py` — self-contained: parametric sector
  generator + oracle, a small numpy MLP and 1-NN (sized generically by
  in/out dims; same algorithm as E12, not sprint-shaped), the four
  conditions, three legs, fixed seeds. `composite-learned`/`-symbolic` build
  a real `CompositeWorld` (children = learned/exact `Transition`s) so the
  experiment exercises the actual machinery, not a stand-in.
- Results -> `experiments/results/e36_representations.json` with per-leg,
  per-K, per-condition detail (regeneration of any table/figure).
- Offline, deterministic, pure numpy + stdlib — no Ollama. Fast.

## Paper

- `scripts/make_paper_assets.py`: `fig_representations` (3-panel: gen-vs-K,
  interference bars, sample-efficiency curve) + macros; `NumExperiments`
  bumped (reconcile with E34/E35 numbering at integration).
- New Results subsection extending E12/E19: "Composition yields better
  representations." Honest-results rule: whatever the numbers say ships.

## Out of scope

GPU / deep nets (the comparison is laptop numpy, matching E12); cross-sector
coupling under learning (sectors independent here so the generalization claim
is clean — bridges were E30's subject); LLM synthesis arms (E30 covered
synthesis; this is about learned representations).
