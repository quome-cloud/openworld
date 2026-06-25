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

### E94 result (code world model for level 2) — honest negative
Synthesized a level-2 model (fidelity 0.44), tree-searched **1374 distinct-config candidate plans**
in imagination, executed in real env: **best 1/6, level 2 NOT reached.** The CWM made candidate
generation efficient (far more configs than blind search), but level 2's win is opaque/rare and not
recoverable this way -> level 2 is a hard wall (consistent with E93c). Next: ensemble (E95) to raise
model fidelity, which should improve planning where fidelity (not goal-opacity) is the limiter.

## Foundational add — verified reward induction (framework primitive)
Added `openworld.CodeObjective` + `induce_reward` to the CORE (zero-dep, sandbox-run, 102 tests pass):
a reward/goal as VERIFIED sandboxed code, induced + exact-match-verified from observed rewards --
symmetric to CodeTransition. Documented in the framework paper (sec:reward-induction).
- **E97 (apply to sp80):** induced a verified level-completion reward, **held-out acc = 1.000**
  (228 examples, 6 positive). The win condition is now a first-class verified artifact -- the agent
  can RECOGNIZE a goal-state, the missing pillar for goal-directed solving.
- **E95 ensemble (honest, partial):** per-cell majority HURTS on ka59 (0.13 vs best-single 0.27);
  naive voting over weak correlated code models is worse than selection -> switch combiner to
  verification-based SELECTION (choose, don't average). Other games pending.

## Verified-loop closure + foundational primitives (a/b/c)
- **(a/c) ConsensusTransition** (core): committee of world models, verification-based SELECT (default)
  or VOTE -- "average or choose from multiple worlds" as a primitive. E95 showed averaging weak code
  models hurts, so SELECT is the right default. Zero-dep, tests pass.
- **(b) E98 goal-recognizing solver:** the agent recognizes the level-1 win via its OWN induced
  verified reward (E97 CodeObjective) -- **loop-closure acc 1.0 (18/18 steps)** vs env truth. The full
  verified loop (synthesized dynamics + synthesized reward) is closed. Level-2 reward did not fire in
  1500 attempts -> L2 remains the wall (recognition != reachability).

## E99 — apply the toolkit to ALL games (interact-biased reward search + replay-verify)
The sp80 win needed an INTERACT action (5); the uniform E93 sweep under-weighted those. E99 runs an
interact-biased (0.45) reward search per game + deterministic replay-verify.
- **2/25 SOLVED (verified): sp80 (L1, 18 actions) + sk48 (L1, 309 actions, NEW).** Interact-bias
  unlocked sk48 where uniform search found nothing.
- 23 games: no reward in 12k interact-biased steps -> need deeper/multi-seed search or goal-directed
  methods. Deep multi-seed sweep (25k x 4 seeds) running.

## Final solve tally (toward all 25)
- **Search (E99, interact-biased real-env search + replay-verify): 6/25 level-1 solves** --
  ar25, cd82, ls20, sk48, sp80, tu93. (NOT world-model-driven: pure env search.)
- **Model-based (E101, predict-lookahead MPC THROUGH the synthesized code world model + typed
  perception + verified-reward recognition): cd82 solved world-model-driven** (proof the code world
  model can drive a solve; slow MPC, sweeping the rest in background).
- **~19 games remain walled:** no reward found by 25k x 4-seed interact-biased search -> the win needs
  goal *understanding*, not reachability. This is the genuine frontier (cf. leaderboard top ~1.5%).
- Honest split: directed *search* establishes reachability (6); the *world model* drives solving where
  navigation is the bottleneck (cd82); goal-inference remains the wall for the rest.

## E102 — core-knowledge goal-discovery (the principled attack): honest NEGATIVE
Generated candidate-goal CodeObjectives (reach/cover/collect/extremize over typed roles) and planned
to each THROUGH the code world model (MPC), per game. **0/9 walled games solved -- even s5i5 with a
fidelity-1.0 model and 20 candidate goals.** Interpretation (this RULES OUT a hypothesis):
- The walled games' win conditions are **NOT atomic core-knowledge configurations** (reaching/covering/
  collecting/extremizing a perceived object). Achieving those candidate configs never triggers reward.
- So the goals are **compositional** (multi-step sub-goal sequences) and/or depend on structure our
  typed perceptor + atomic-relation space doesn't capture (precise patterns, timing, hidden state).
- This sharpens *why* these are walls (consistent with ARC resisting simple priors; SOTA ~1.5%).
Remaining genuinely-different directions: (i) **composite objectives** (sub-goal sequences -- a much
larger discovery search); (ii) **visual perception** (render + Claude vision for semantic/pattern goal
hypotheses the atomic templates miss). Both harder; may not crack it -- this is the frontier.

## E103 — end-to-end Claude-driven hypothesis embedding (the full ultrathink): honest NEGATIVE + a deep finding
Claude GENERATES the goal-hypothesis space (vs E102's hand-coded templates): rich, compositional,
geometric, relational hypotheses, closed-loop refined over 3 rounds, each pursued via goal-conditioned
MPC through the (perfect/near-perfect) code world model. Examples Claude proposed for s5i5 across
rounds: drain_then_clear, steps_eq_boxcount, boundary_to_centroid, mirror_symmetry, boxes_identical,
drainhalf_then_sym. Result on high-fidelity walled games (s5i5 fid1.0, r11l 0.93, vc33 0.83): 0 solved.
**Deep finding:** the failure isn't poor hypotheses -- they're sophisticated. It's REPRESENTATIONAL:
the win is scored as goal_score(FRAME) (a state-configuration), but many ARC-3 wins are a PROCEDURE /
protocol (a specific action sequence/timing), NOT a reachable frame-configuration. A state-score can't
express a procedure, so MPC optimizes the wrong thing -- which is exactly why brute search (E99) finds
some wins (it stumbles into the procedure) while hypothesis-driven planning (E102/E103) cannot. The
frontier is goal-as-procedure, not goal-as-state; hypotheses over TRAJECTORIES (much larger) is the
open direction.

## E104 — Bayesian subworld + semiring procedure-search: honest NEGATIVE (completes the trilogy)
Procedure-aware (ordered sub-goal sequences over subworlds), hierarchical execution, TROPICAL-semiring
planning through the code world model, Bayesian-ordered hypothesis search. Result on s5i5/r11l/vc33
(incl. fid-1.0 s5i5): **0/3**. (Segments often terminate into game-overs -- pursuing an extremize
sub-goal drives the agent into losing states, itself evidence the win isn't a simple extremize-then-X.)

### The rigorous conclusion (three independent principled attacks)
- E102 atomic core-knowledge goals: 0/9
- E103 Claude-generated rich/compositional hypotheses, closed-loop: 0/3
- E104 procedure-aware Bayesian subworld + semiring planning: 0/3
None crack the walled games **even with perfect world models**. The goal-inference wall resists the
entire structured-goal-discovery + verified-world-model + Bayesian/semiring toolkit. The win
conditions are opaque in a way these methods cannot reach -- consistent with frontier agents at ~1.5%.
The one qualitatively-untried lever: **pure visual reasoning** (Claude vision on official-rendered
frames), which is the only representation we haven't fed the model.
