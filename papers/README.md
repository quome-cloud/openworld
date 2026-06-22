# Papers

The OpenWorld manuscript is split into two papers so neither is overcrowded. Both share the
same authors, template, and the centrally-generated assets in `../paper/` (figures, tables,
`numbers.tex`, `refs.bib`), reused here via symlinks so the single asset pipeline
(`scripts/make_paper_assets.py` → `paper/`) stays the source of truth — never edit numbers by
hand, and never duplicate the assets.

- **`world-time-compute/`** — the scientific spine: verified code world models are exact (no
  compounding error), and *world-time compute* (fine-tuning on trajectories from many verified
  worlds) lifts generalization to held-out worlds, with a predictive regularity and its
  boundaries. Contains the head-to-head/exactness results, the E74–E82 world-time-compute
  family, the predictive regularity, the trained-vs-verified bake-off, planning, and the
  self-training positioning.
- **`framework/`** — the system: authoring, the plan–generate–verify relay, the composition
  algebra, the perceive→world→emit boundary, objectives/dials, agents-as-a-judge, portable
  specs + model cards, and the breadth of domain demonstrations (economy, corporate, trading,
  relativity, brain, optimal transport, …).

Build either with `tectonic main.tex` from its directory (the symlinks make `figs/`,
`tables/`, `numbers.tex`, and `refs.bib` resolve as if local).

`../paper/` remains the combined manuscript and the asset source; the two papers above are
carved views of it. Each abstract points to the other as a companion.

> Status: v1 structural split — both compile cleanly with no undefined references. The
> per-paper Introductions and Conclusions are still largely shared from the combined manuscript
> and would benefit from a tailoring pass (each leading with its own thesis).
