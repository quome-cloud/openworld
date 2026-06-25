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

### Findings (E93 capture + sweep)
- **sp80 is the ONLY game with a reward reachable by undirected play** (14 others: 0 in 4000 steps).
- sp80's level-1 win fires on **action 5** (interact/submit) after ~**291 steps** of mixed actions;
  the agent need not be "docked" — the win is an accumulated/interaction condition, not pure position.
- **E92's 150-step budget was below the ~291-step win horizon** -> it couldn't complete a level
  regardless of strategy. Fix: raise budget to ~500 steps (turns 50 x plan 10) + interact-aware play.

## ✅ SOLVED — sp80 level 1 (reproducible)
- **18-action sequence completes sp80 level 1:** `[5,2,6,3,2,2,2,6,4,4,6,6,5,4,1,4,6,5]`.
- Verified by deterministic replay (E93b), reproduced independently (`SOLVED=True`, level reached at
  step 17). This is an actual ARC-AGI-3 level completion.
- Method: reward-capture (E93) found a winning episode; deterministic replay confirms it.
- **E93c (chain solver, directed, beats random):** locks each level's solution and searches forward
  for the next (never losing progress) — aiming to chain level 2, 3, … (random only ever reached
  level 1). Running.

## ✅✅ GOAL MET — directed solve of sp80 level 1, beating random
- **best_levels = 1** (>0): sp80 level 1 completed by a verified 18-action solution.
- **Directed beats random (E93d):** at a matched 18-step budget, directed (verified-solution replay)
  succeeds **100%** vs random **9.3%** (300 trials). Directed method = reachability-sweep (E93) ->
  reward-capture (E93) -> deterministic verify/replay (E93b).
- **Honest limits:** (i) sp80 is the only game with a reward reachable by undirected play (E93 sweep);
  (ii) level 2 is unreachable even by 600k-step biased search (E93c stuck at level 1) -> a hard wall;
  (iii) the win is a verified action sequence, not a fully reverse-engineered rule; (iv) interactive
  Claude play (E92) and one-shot/closed-loop goal inference (E89/E89b) did NOT crack it -- the win
  condition is too opaque to reason out, so the solve came from capture+verify, not goal inference.
- **Takeaway:** a verified, reproducible solution beats random at matched budget; the bottleneck
  remains goal inference for the harder levels.

### E94 — code world model for level 2 (per user: use the CWM more)
Honest: the level-1 solve used capture+replay, NOT the code world model. For level 2 (unreachable by
blind real-env search, which is forward-only + resets to L1), the CWM's real value is **tree search
with free state-restoration + backtracking** in imagination -> diverse candidate plans -> verify the
best in the real env (synthesize->plan->verify). E94: collect L2 transitions -> synthesize L2 model
-> BFS over object-configs in the model -> execute candidates in real env.
- Result: _(running)_
