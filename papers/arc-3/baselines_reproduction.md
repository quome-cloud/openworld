# T368 Phase 1: Baseline Reproduction — Jim's ARC-AGI-3 Pipeline

Run: 2026-06-26 | Commit: e1d8c7f | Branch: exp-arc-agi3-e88-transfer

**Config**: `e86_arc3.py --steps 300 --seed 0` (no synthesis), all 25 public games.

## Key findings

- **25/25 games** ran to completion
- **24/24 playable** games have `replay_determinism = 1.000` ✓ (matches paper's "determinism 1.0" claim)
- **1 game (tn36)** has 0 valid transitions — arcengine `KeyError:'x'` bug fires on all ACTION6 calls; the environment is broken
- **baseline_levels**: random exploration scores 0 levels on 23/24 games, 1/6 on sp80 (luck)
- **copy_frame_exact**: 3 games (ft09, lp85, su15) have = 1.0 — the frame never changes; copy-frame baseline is trivially perfect on these; synthesis on them is uninteresting

## Per-game table

| game | transitions | baseline_levels/win | replay_det | mean_cells_changed | copy_frame_exact | note |
|------|-------------|---------------------|------------|--------------------|-----------------:|------|
| ar25 | 300 | 0/8 | 1.000 | 71.3 | 0.1767 | |
| bp35 | 237 | 0/9 | 1.000 | 92.5 | 0.2658 | ACTION6 partial fail |
| cd82 | 300 | 0/6 | 1.000 | 62.3 | 0.2400 | |
| cn04 | 245 | 0/6 | 1.000 | 164.8 | 0.0449 | ACTION6 partial fail |
| dc22 | 300 | 0/6 | 1.000 | 4.9 | 0.2133 | |
| ft09 | 300 | 0/6 | 1.000 | 0.0 | 1.0000 | static frame |
| g50t | 300 | 0/7 | 1.000 | 33.1 | 0.2233 | |
| ka59 | 300 | 0/7 | 1.000 | 13.4 | 0.1233 | |
| lf52 | 250 | 0/10 | 1.000 | 1.0 | 0.0000 | ACTION6 partial fail |
| lp85 | 300 | 0/8 | 1.000 | 0.0 | 1.0000 | static frame |
| ls20 | 300 | 0/7 | 1.000 | 37.6 | 0.0167 | synthesis target |
| m0r0 | 245 | 0/6 | 1.000 | 55.7 | 0.1714 | ACTION6 partial fail |
| r11l | 300 | 0/6 | 1.000 | 1.1 | 0.0000 | |
| re86 | 300 | 0/8 | 1.000 | 41.7 | 0.0033 | |
| s5i5 | 300 | 0/8 | 1.000 | 1.3 | 0.0000 | |
| sb26 | 300 | 0/8 | 1.000 | 0.3 | 0.6633 | |
| sc25 | 300 | 0/6 | 1.000 | 15.5 | 0.2667 | |
| sk48 | 300 | 0/8 | 1.000 | 40.4 | 0.2467 | |
| sp80 | 300 | 1/6 | 1.000 | 61.4 | 0.0000 | synthesis target |
| su15 | 300 | 0/9 | 1.000 | 0.0 | 1.0000 | static frame |
| tn36 | 0 | 0/7 | 1.000 | N/A | N/A | ALL actions fail (broken) |
| tr87 | 300 | 0/6 | 1.000 | 20.1 | 0.0000 | |
| tu93 | 300 | 0/9 | 1.000 | 7.3 | 0.0000 | |
| vc33 | 300 | 0/7 | 1.000 | 1.3 | 0.0000 | synthesis target |
| wa30 | 300 | 0/9 | 1.000 | 26.2 | 0.1400 | |

## Verdict

**Phase 1 CONFIRMED**: replay determinism = 1.0 for all playable games, matching the paper's claim. The 25-game baseline is a solid reproduction of Jim's pipeline.

**Next**: Phase 2 (synthesis) — awaiting valid Anthropic API key for `--anthropic claude-sonnet-4-6` run on the 6 hand-picked games (ls20, ft09, vc33, sp80, cn04, ar25). ft09/su15/lp85 are static-frame (copy_frame=1.0) — low-value synthesis targets; recommend substituting with non-trivial games (re86 or sc25) for the actual synthesis benchmark.
