# Composition Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three publication-quality matplotlib figures (composite-world anatomy, composition-vs-cliff, agent's-eye traversal view) in the deterministic assets pipeline, plus their figure blocks in the paper.

**Architecture:** Three `fig_*` functions appended to `scripts/make_paper_assets.py`, drawing with `matplotlib.patches` (FancyBboxPatch, FancyArrowPatch) in the house palette; data sourced from `e30_composition.json`, `e31_nested_fidelity.json`, `e20_complexity.json`. Aesthetics are developed in a render→view→adjust loop (the executor views PNGs directly); the user gets the rendered PNGs for final judgment before the PR.

**Tech Stack:** matplotlib (already a script dependency), tectonic for the PDF. Spec: `docs/superpowers/specs/2026-06-12-composition-figures-design.md`. Branch: `composition-figures`.

---

### Task 1: The three figure functions

**Files:**
- Modify: `scripts/make_paper_assets.py`

- [ ] **Step 1: Implement `fig_composition(e31)`** → `paper/figs/composition.png` (~7×4.6 in, dpi 200).
  Layout (axes coords, axis off): outer `region` FancyBboxPatch (rounded, `SLATE` 6% fill, labeled top-left); two inner `country` panels (`BLUE` 8% fill) side by side; each holds two `city` cards (white fill, thin border, soft shadow via offset grey patch) showing 3-line state text (`output / treasury / rate` with values from the e31 JSON structure where present, else the documented initial values). Overlays, color-coded:
  - bridge: thick `TEAL` FancyArrowPatch (arc3 rad≈0.25, lw≈2.5, arrowstyle '<->') between one city in each country, label "trade · Σ conserved" with a small rounded badge;
  - route: `ORANGE` dashed arc between two other cities, agent dot (filled circle, `PURPLE`) at the arc midpoint with tiny label "agent · toll −2";
  - aggregators: thin dashed upward arrows from each city pair into a parent `_agg` chip (small rounded box, e.g. "gdp = Σ output") at each country's top edge, and from country chips to a region chip;
  - binding: one downward `SLATE` arrow labeled "rate ↓".
  Legend strip along the bottom: four short colored line/arrow samples + labels (bridge / route / aggregator (derived) / binding). Title inside the canvas, top-left, bold, small.
- [ ] **Step 2: Implement `fig_composition_cliff(e20, e30)`** → `paper/figs/composition_cliff.png` (~5.8×3.5 in, dpi 200).
  Replot e20 summary (xs = n_rules, ys = mean_probe_accuracy, pooled CI band, `BLUE` line fading: use `SLATE` for the line with `BLUE` band, or keep e20's style); at x=16: monolithic e30 point as a red-toned (`#B91C1C`) circle with CI whiskers sitting near the curve, compositional point as a large `TEAL` star (markersize≈16, zorder top) with CI whiskers at 0.92; curved annotation from star: "same 16 rules:\n4×4 children + verified bridges"; faint dotted horizontal from the star to the y-axis. Y label "probe accuracy", x label "interacting rules (R)", xticks from e20, ylim (0, 1.05), grid alpha 0.25.
- [ ] **Step 3: Implement `fig_traversal(e31)`** → `paper/figs/traversal.png` (~7×4.2 in, dpi 200).
  Same nested geometry as Step 1 but rendered from agent-at-`c0:b`'s perspective: `c0:b` card full saturation + bold border (`PURPLE` accent, "you are here" dot); ancestor chips (`c0`'s `_agg`, region `_agg`) full visibility; the route-adjacent city (`c1:b`) drawn as a reduced summary card (lighter, fewer lines, label "neighbor · via route"); non-observable cities (`c0:a`, `c1:a`) greyed (3% grey fill, 30% alpha text, label "not observable"); the route arc drawn to the neighbor. Bottom strip styled as a prompt fragment: `legal_actions: c0:b:work · c0:b:trade · travel:c1:b` in monospace on a light chip.
- [ ] **Step 4: Wire all three into `main()`** after the existing fig calls:
  `fig_composition(data["e31_nested_fidelity"])`, `fig_composition_cliff(data["e20_complexity"], data["e30_composition"])`, `fig_traversal(data["e31_nested_fidelity"])`.
- [ ] **Step 5: Render → view → adjust loop.** Run `python3 scripts/make_paper_assets.py`; VIEW each PNG (Read tool on the file); iterate on spacing, font sizes, overlap, arrow collisions until clean and striking. Re-run the script twice and `git status` to confirm byte-stable output.
- [ ] **Step 6: Commit** — `git add scripts/make_paper_assets.py paper/figs/composition.png paper/figs/composition_cliff.png paper/figs/traversal.png && git commit -m "Add composition/cliff/traversal figures to the assets pipeline"`

### Task 2: Paper wiring

**Files:**
- Modify: `paper/main.tex`

- [ ] **Step 1:** In the composition Results subsection (`sec:composition`): add `\begin{figure}[htb]` blocks for `figs/composition.png` (label `fig:composition`, caption explaining the four coupling channels with E31 grounding) and `figs/composition_cliff.png` (label `fig:compcliff`, caption tying E20's curve to E30's two points); reference both from the prose. Place `figs/traversal.png` (label `fig:traversal`, caption on scoped observation/legal actions) near the traversal/E31 paragraph; if the subsection crowds, move `fig:composition` up to the Experimental Setup paragraph.
- [ ] **Step 2:** `make -C paper paper`; warning count must stay at baseline (15 Overfull) ± explainable delta; no undefined references. Commit: `git add paper/main.tex paper/main.pdf && git commit -m "Paper: composition, cliff, and traversal figures"`.

### Task 3: Preview gate + PR

- [ ] **Step 1:** Send the three rendered PNGs to the user (SendUserMessage with attachments) for the "stunning" judgment. Iterate on feedback (back to Task 1 Step 5) until approved.
- [ ] **Step 2:** `python -m pytest tests/ -q` (untouched, expect pass); push branch; open PR referencing the spec.
