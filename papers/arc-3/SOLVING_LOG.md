# ARC-AGI-3 solving log

**Goal:** actually *solve* (complete ≥1 level of) an ARC-AGI-3 game. Local only (Claude Code + CPU,
no cloud/GPU). Rethink strategies via the experiment frameworks, documenting each. Budget ≤12h.

## What we know (the wall, from E86–E90)
- Verified code models synthesizable (E86; near-perfect on s5i5, r11l, sb26).
- Planning works at high fidelity (E87 controllability); partial models mislead.
- **Solving = ~0.** Bottleneck is **goal inference**: undirected exploration never triggers a reward
  (E88 pixel-novelty, E90 graph-novelty both 0); one-shot goal guesses are wrong (E89, s5i5 timer
  confusion); closed-loop negative feedback can't bootstrap without one positive reward (E89b).
- **Key: we need ONE positive reward**, then induce the win condition.

## Strategy ledger

### E92 — Claude as an interactive player (NEW)
Hypothesis: one-shot goal inference (E89) fails because Claude commits to a guess. An *interactive*
player that sees consequences + reward and adapts (like a human) can discover the goal by playing.
Perception = object-graph view + history; Claude proposes a short plan + an evolving goal note;
execute in real env; feed back level deltas. Prioritize games where level-1 is reachable (sp80).
- Result: _(running)_

### arc3_world / E93 — repo-native code world model in the loop (per user direction)
Use code world models *as defined in the repo* (World + CodeTransition), not a raw function. Finding:
the repo sandbox is **pure-Python** (whitelist math/random, no imports/numpy), so we synthesize the
dynamics as a numpy-free `transition(state, action)` over the frame-as-list, **verify it in the
sandbox**, and wrap as an `openworld.World` (like `minigrid_world.py`). E93 solver then plans via
`w.transition.step` (lookahead in the verified model) + Claude reasoning + real-env reward.
- `arc3_world.py`: synthesize + sandbox-verify + build World. Running on s5i5.
- Result: _(running)_

### Candidate next strategies (if E92 stalls)
- E92b: vision — render frames as images, use Claude's multimodal reasoning for the goal.
- E92c: brute-force / systematic short-sequence search in the real env to grab any level-1 reward
  on the easiest game, then induce the win-rule (E88b piecewise) and plan the rest.
- E92d: subgoal decomposition over the object graph (key→door→exit style).
