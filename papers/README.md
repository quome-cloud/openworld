# Papers

The OpenWorld manuscript is split into two papers so neither is overcrowded. Both share the
same authors, template, and the centrally-generated assets in `assets/` (figures, tables,
`numbers.tex`, `refs.bib`, and shared `sections/`), reused here via symlinks so the single asset
pipeline (`scripts/make_paper_assets.py` → `papers/assets/`) stays the source of truth — never
edit numbers by hand, and never duplicate the assets.

- **`world-time-compute/`** — the scientific spine: verified code world models are exact (no
  compounding error), and *world-time compute* (fine-tuning on trajectories from many verified
  worlds) lifts generalization to held-out worlds, with a predictive regularity and its
  boundaries. Contains the head-to-head/exactness results, the E74–E83 world-time-compute
  family, the predictive regularity, the trained-vs-verified bake-off, planning, and the
  self-training positioning.
- **`framework/`** — the system, framed as **world computing**: authoring and the
  plan–generate–verify relay, the formal composition algebra, the perceive→world→emit boundary,
  objectives/dials, agents-as-a-judge, portable specs + model cards, the Gymnasium adapter, and
  the breadth of domain demonstrations (economy, corporate, trading, relativity, brain, optimal
  transport, …). Benchmarks the substrate on cost, latency, and performance.

Build either with `tectonic main.tex` from its directory (the symlinks make `figs/`,
`tables/`, `numbers.tex`, `refs.bib`, and `sections/` resolve as if local).

`assets/` holds the generated artifacts only; there is no longer a separate combined
manuscript. It was the union of the two papers above, so it was removed to de-duplicate the
repository — the two carved papers plus the shared `assets/` carry all of its content with
nothing lost. Each abstract points to the other as a companion.
