# Slim-down proposals for the ARC-3 paper

Specific "cut / merge / replace" suggestions. Ordered by pages saved, and each includes a
concrete alternative so the paper doesn't lose the point.

Assumptions: current manuscript is roughly at target page count for a workshop / short-paper
submission and Jim wants it to READ shorter, not just be shorter. Prose density and figure
economy matter more than raw pagecount.

## Proposal 1 — Replace 2–3 per-game tables with a single 25×3 heatmap

**What's there now.** Multiple per-game outcome tables and one per-model outcome table appear
across §7 (multi-perception), §8 (cascade), §9 (full-game). Each table has ~25 rows and a few
columns. Together they occupy roughly 1.5 pages.

**Proposal.** Replace with **one heatmap** (games on Y, solver arms on X, cell value = max
levels reached; empty cell = never attempted). `game_model_success_heatmap.png` in this folder
is a prototype. Add a small companion table (~10 lines) for the 6 games solved by pure search
(E99) with sequence length — since that's a different variable and the reader benefits from
seeing it.

**Pages saved.** ~1 page.
**What you don't lose.** Every fact currently in the tables is in the heatmap; you gain the
2-D visual pattern (opus dominates x, codex specialises in y, etc.).

## Proposal 2 — Merge §7 (multi-perception) into §8 (cascade) as "the cheap tier"

**What's there now.** §7 introduces multi-perception consensus as its own method. §8 describes
the cascade that uses it. Substantially overlapping motivation.

**Proposal.** Rewrite as a single section titled "Cheap-tier: multi-perception consensus".
Move the surprising s5i5 finding (see PAPER_FEEDBACK #6) into §5 as its own callout instead —
so it doesn't get buried in the merged section.

**Pages saved.** ~0.5 page.
**What you don't lose.** Nothing; the surprise gets *more* space, not less.

## Proposal 3 — Cut §8's cascade description; keep one footnote

**What's there now.** §8 spends a page describing the try-cheap-then-escalate deployment
strategy. This is a system deployment fact, not a research contribution.

**Proposal.** Cut to: "We deploy a try-cheap-then-escalate cascade: 12/25 games solve at the
multi-perception tier and skip the LLM agent entirely. Full ablation and code in Appendix D."
Move the actual description to the appendix.

**Pages saved.** ~0.5–0.75 page.
**What you don't lose.** The engineering fact is preserved; the research paper doesn't need to
argue for it because it's just a deployment choice.

## Proposal 4 — Compress the E86 fidelity discussion by ~30%

**What's there now.** §3 (fidelity) develops the Claude-vs-qwen story slowly, with a
paragraph on each metric (exact-match rate, per-game breakdown, agentic loop).

**Proposal.** Replace two of the per-metric paragraphs with a small strategy-fingerprint figure
(the E148 radar, `strategy_radar_per_model.png`) and one sentence:
"Beyond raw fidelity, arms differ qualitatively in strategy usage (Fig. R): Fable is memory + verify;
opus is perceive + verify; codex is stripped-down. This is why cascade routing (§8) is orthogonal
to just adding compute."

**Pages saved.** ~0.3 page.
**What you don't lose.** You gain a memorable visual anchor for the model-differences argument
that currently reads as tabular.

## Proposal 5 — Fold the four separate "negative-result" callouts into one taxonomy

**What's there now.** E88 (novelty exploration), E89 (one-shot goal inference), E102 (atomic
goals), E103 (LLM hypotheses), E104 (Bayesian sub-world) each get their own paragraph. Each
concludes: "…and this fails on the walled subset."

**Proposal.** One "Why goal-discovery walls perfect models" table:

| method | attacks | tested on | wins on walled | key failure mode |
|---|---|---|---|---|
| E88 | undirected exploration | pixel/graph novelty | 0 / N | reward not reachable by novelty |
| E89 | one-shot LLM hypothesis | goal proposal | 0 / N | wrong goal (e.g. timer ≠ objective) |
| E102 | atomic objectives | 4-atomic-goal ensemble | 0 / N | goals are procedures not states |
| E103 | rich LLM hypotheses | closed-loop refinement | 0 / N | can't bootstrap without one positive |
| E104 | Bayesian sub-world | TROPICAL semiring | 0 / N | plans through wrong-hypothesis model |

Then one connective paragraph. Currently this material sprawls across ~1.5 pages; the table +
paragraph is ~0.5 page.

**Pages saved.** ~1 page.
**What you don't lose.** The negative-result story reads TIGHTER, and the shared conclusion
("goals are procedures, not states") is easier to see.

## Proposal 6 — Move E86b (agentic loop) and the cost-comparison paragraphs to appendix

**What's there now.** §3 has an agentic-loop paragraph (Claude 49→53%, no lift for qwen). §9
has a paragraph on $7 vs $350 vs baseline1. Both are honest caveats, not headline claims.

**Proposal.** Both move to appendix "Efficiency and Ablation Notes". Main text keeps one
sentence each: "An agentic verification loop marginally lifts Claude (49→53%) and does nothing
for qwen; capability-gated (App. C.2)." and "Our cost estimate is ~50× cheaper than baseline1
with uncontrolled engine/harness differences (App. C.3)."

**Pages saved.** ~0.5 page.
**What you don't lose.** Both facts stay reproducible; only the headline exposure changes.

## Rough page-count arithmetic

| Proposal | Pages saved (approx.) |
|---|---|
| 1. Heatmap replaces per-game tables | 1.0 |
| 2. Merge §7 into §8 | 0.5 |
| 3. Cut cascade description | 0.6 |
| 4. Compress fidelity discussion via radar | 0.3 |
| 5. Fold negatives into taxonomy table | 1.0 |
| 6. Move ablation paragraphs to appendix | 0.5 |
| **Total plausible savings** | **~3.9 pages** |

That's roughly a full workshop-paper page-budget delta, so the paper could go from feeling
"comfortable" to feeling "tight" without losing any load-bearing content. The negative-results
taxonomy (#5) is particularly high-value because it *also* makes the diagnosis-first framing
that PAPER_FEEDBACK #1 asks for materially easier.

## Two visual replacements to consider

Not slimming per se, but replacing 3-column-of-numbers with a graphic:

1. **`strategy_success_correlations.png`** — replaces any prose that says "simulate correlates
   with success; probe doesn't". This is 20 lines of prose in ~3 places currently, replaceable
   with one 3-inch figure. About 0.4 page savings.
2. **`strategy_divergence.png`** — replaces the multi-paragraph justification for keeping three
   solver arms in the fidelity discussion. One number per arm (0.004, 0.058, 0.108); reader
   sees immediately that Fable is qualitatively different, so it deserves separate reporting.
   About 0.2 page savings.

## What NOT to cut

- The honest caveats section (§9-ish). It's expensive on pagecount but it's what makes the
  paper credible.
- The determinism = 1.00 fact. Load-bearing.
- The negative results themselves. Reframe them (Proposal 5) but do not hide them.
- The E121 round-trip claim. Reproducibility is a differentiator.

## Suggested execution order

If Jim wants to slim the paper efficiently:

1. Do Proposal 5 first (taxonomy table). Biggest saving, also improves the paper's central
   argument. About 90 minutes.
2. Do Proposal 1 (heatmap). Immediate visual clarity win. About 45 minutes to lay out.
3. Do Proposal 3 (cut cascade description). Cleanest cut. 30 minutes.
4. Do Proposals 4 + 6 as a pair, since they both migrate paragraphs to referenced figures /
   appendix. About 90 minutes total.
5. Do Proposal 2 last (if at all) — depends on how §7's promotion of s5i5 lands.

Total: ~5 hours for ~3 pages saved and a materially sharper paper.
