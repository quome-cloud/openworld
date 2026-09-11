# Papers

The OpenWorld manuscripts share the same authors, template, and the centrally-generated assets in `assets/` (figures, tables,
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
- **`arc-3/`** — the ARC-AGI-3 technical report: verified code world models built by acting,
  the source-free audited protocol, the goal-as-procedure diagnosis, and the atlas of serveable
  per-game world models.
- **`agentworld/`** — an essay for the *Antikythera* **Agentworld** special issue (MIT Press,
  Fall 2026) that reframes the ARC-3 report's goal-as-procedure diagnosis for a preemptive
  anthropology of agents: modeling is cheap, intent is expensive. Reuses the ARC-3 figures and
  macros; adds no new assets.

Build any of them with `tectonic main.tex` from its directory (the symlinks make `figs/`,
`tables/`, `numbers.tex`, `refs.bib`, and `sections/` resolve as if local).

`assets/` holds the generated artifacts only; there is no longer a separate combined
manuscript. It was the union of the first two papers above, so it was removed to de-duplicate the
repository — the carved papers plus the shared `assets/` carry all of its content with
nothing lost. Each abstract points to its companions.
