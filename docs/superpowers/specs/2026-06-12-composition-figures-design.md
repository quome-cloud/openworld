# Composition figures: design

**Date:** 2026-06-12
**Status:** approved

## Goal

Three publication-quality figures that make compositional worlds, bridges,
and traversal intuitive at a glance, generated inside the deterministic
assets pipeline (no hand-drawn TikZ — the paper's figures all regenerate
from `experiments/results/*.json` via `scripts/make_paper_assets.py`).

## Figures

1. **`paper/figs/composition.png` — "Anatomy of a composite world."**
   E31's real structure: a `region` panel containing two `country` panels,
   each containing two `city` state cards (actual fields: output, treasury,
   rate). Color-coded coupling channels with a legend: bridges (teal curved
   arrows, "conserved" badge), one Route (amber dashed arc, agent dot,
   "toll −2"), aggregators (upward dashed arrows into `_agg` chips —
   derived, one-way), one binding (downward slate arrow). House palette
   (BLUE/ORANGE/TEAL/SLATE/PURPLE), soft shadows, rounded boxes.
2. **`paper/figs/composition_cliff.png` — "Composition defeats the cliff."**
   E20's accuracy-vs-R curve with CI band falling to R=16; at R=16 the E30
   monolithic point (0.31, on the curve, CI whiskers) and the compositional
   star (0.92, CI whiskers) high above, annotated "same 16 rules: 4×4
   children + verified bridges"; faint guide line linking the star back to
   R=4-level accuracy.
3. **`paper/figs/traversal.png` — "What an agent sees."** Agent's-eye view
   at a city: own card full-detail/full-saturation; ancestors as `_agg`
   summary chips; the route-adjacent city as a reduced summary card; all
   non-observable nodes greyed out; a `legal_actions` strip underneath
   (`c0:b:work · c0:b:trade · travel:c1:b`). Values sourced from the E31
   JSON where recorded, static labels otherwise.

## Wiring

- Three `fig_*` functions in `make_paper_assets.py` reading
  `e30_composition.json`, `e31_nested_fidelity.json`, `e20_complexity.json`.
- `paper/main.tex`: Figure 1 + 3 referenced from the composition Results
  subsection (Figure 1 may move to Experimental Setup if crowded); Figure 2
  placed beside the E30 discussion, cross-referenced with
  `Figure~\ref{fig:complexity}`.
- PDF rebuilt with tectonic; regeneration must stay deterministic
  (re-running the script produces byte-identical PNGs, matching the
  existing figures' behavior).

## Acceptance gate

Rendered PNGs are sent to Jim for the "stunning" judgment before the PR;
iterate on visuals until approved. Technical gates: deterministic
regeneration; no new LaTeX warnings; full test suite unaffected.

## Out of scope

Animating or interactive figures; TikZ; changes to experiment data.
