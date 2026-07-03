# Researchy team contribution to the ARC-3 paper

**Status:** additive / optional. Everything here sits under `contributions/` and can be pulled
into the paper, ignored, or cherry-picked without touching the main experimental pipeline.

**Authored by** Origin Aleph (A001), researchy team.
**On top of** Jim's `experiments/e148_strategy_space.py` + the `arc3_traces/` corpus.
**Branch:** `researchy-strategy-deepdive-contrib`, opened as a PR against `arc3-runner-fix`
so it stays adjacent to the paper branch and doesn't pollute main.

## What's in here

| file | what it is |
|---|---|
| `deeper_analysis.py` | Reproducible script. Reads `experiments/results/e148_strategy_space.json` + `experiments/results/arc3_traces/meta/*.json`; writes the PNGs + `summary.json` + `takeaways.txt` in this folder. |
| `strategy_radar_per_model.png` | 9-strategy spider chart, one polygon per solver arm. |
| `strategy_success_correlations.png` | Horizontal bar chart of Pearson r (each strategy vs levels reached). |
| `game_model_success_heatmap.png` | Games × models heatmap of max levels reached. |
| `model_tier_efficiency.png` | Twin-axis bars: mean wall-time vs mean levels per (model × tier). |
| `strategy_divergence.png` | Cosine distance of each arm's strategy profile from the 3-arm centroid. |
| `summary.json` | All numbers dumped, so anyone can cite them without re-running. |
| `takeaways.txt` | Bullet-point human-readable summary. |
| `PAPER_FEEDBACK.md` | Detailed reviewer-style notes on the current manuscript. |
| `CONCISENESS_PROPOSAL.md` | Specific "cut / merge / replace-with-figure" suggestions. |
| `figures_for_paper/` | Cleaned versions of the plots, sized for direct paper inclusion (optional). |

## Reproduce

```bash
python3 contributions/researchy_A001_strategy_deepdive/deeper_analysis.py
```

Deps: `numpy`, `matplotlib`. No new dependencies beyond what E148 already needed.

## Headline observations

- **Strategy-level story is sharp.** Pearson r with levels reached: `simulate +0.25`, `probe -0.20`,
  `state_graph -0.14`. The paper's main thesis ("model, don't just poke") is *directly visible in the
  fingerprints*.
- **Fable is the most divergent solver arm** (cosine distance 0.108 from the 3-arm centroid vs
  opus's 0.004). Fable leans memory + verify + mechanic; opus leans perceive; codex is stripped-down.
  Worth calling out because it justifies keeping 3 arms rather than collapsing.
- **Sample-size asymmetry is severe** — opus n=229, codex n=14, fable n=18. The paper should show
  the CI-annotated bars or at minimum mark small-N arms visually. `strategy_radar_per_model.png`
  puts N in the legend for exactly this reason.
- **Per-game success is a natural 25×3 heatmap** — much easier to reason about than the current
  per-game tables scattered through §8–§9. If Jim wants a paper-conciseness win, replacing 2–3 tables
  with one heatmap saves half a page.

Details in `PAPER_FEEDBACK.md` and `CONCISENESS_PROPOSAL.md`.
