# E140 arm comparison — head-to-head + strategy-lift proposals

Additive analysis of the E140 source-free full-budget archives (Opus / Codex / Fable),
with concrete experiments to lift Opus and Sonnet.

## Files

| file | what it is |
|---|---|
| `arm_comparison.py` | Reads the three E140 archive JSONs + `e140_budget_asymmetry.json`; writes 3 PNGs + `overlap_venn.txt` + `arm_comparison_summary.json`. Reproducible, deterministic, no new deps. |
| `head_to_head.png` | Per-game bar chart, opus vs codex (and fable when the archive lands), win_levels overlay. |
| `walls_shared.png` | The subset of games where every arm gets stuck at the same level (±1). Target set for goal-predicate experiments. |
| `budget_lift_before_after.png` | The E140 story: capped 45-min → uncapped 4-hr moves opus 11→16 and codex 2→7. |
| `overlap_venn.txt` | Text set-diff — every-arm-full / opus-only / codex-only / shared-walls. |
| `arm_comparison_summary.json` | All numbers dumped for direct citation. |
| `LIFT_PROPOSALS.md` | Six concrete experiments to lift Opus and check Sonnet, with cost estimates and falsifiable predictions. |

## Reproduce

```bash
python3 contributions/researchy_A001_strategy_deepdive/e140_arm_comparison/arm_comparison.py
```

## Key numbers as of this commit

- Opus @ max: 16/25 full games, 158/183 levels
- Codex @ xhigh: 7/25 full games, 108/183 levels
- Every-arm-full overlap: 7 games (ar25, dc22, ft09, ka59, lp85, su15, tu93)
- Opus-only full: 9 games (cd82, cn04, g50t, m0r0, re86, s5i5, sb26, sc25, tr87)
- Codex-only full: 0 games
- Shared walls: 6 games where every arm stuck within ±1 level (lf52, ls20, r11l, sp80, tn36, vc33)

## Headline finding for the paper

**Codex adds no unique wins** at E140 budget. The "3-arm ensemble" framing needs Fable to
justify itself with at least one unique win, otherwise the multi-model story collapses to
"Opus is best; Codex is cheaper on a subset of games it also solves." Whether Fable adds
unique wins is the *falsifiable* multi-arm hypothesis. See `LIFT_PROPOSALS.md#P2` for the
prediction and the experiment that would settle it.
