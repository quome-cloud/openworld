# E125 Phase-2 — FunSearch world → codex-as-tool traversal (OpenWorld-native)

**Date:** 2026-06-27
**Branch:** `arc3-runner-fix`
**Status:** design approved; implementation pending
**Builds on:** `2026-06-27-e125-executable-world-model-design.md` (steps 1–3 done: verifier gate, faithful
FunSearch synth + failure memory + per-level seeding, goal-energy + autonomous win hypothesis, energy-descent
plan, online-oracle goal-directed grounding, false-win refutation).

## Problem

E125 reliably synthesizes a **verified** world model (`predict()`) on a clean map (dc22: FunSearch
9→12→15→accept = 18/18 exact-match), but **stalls at the objective**: an offline win hypothesis + plan-in-sim
yields "no sim plan" (the hypothesised `level_up` never fires reachably), and folding a grounded level-up back
into one `predict()` hits a compositionality wall (the win→board-reload transition is unmodelable).

The restartable Claude **loose sweep agent** gets past this because it acts in the **real** env with an
**online `g.levels` oracle** + goal-directed play + per-level regime reset (`scripts/sweep_routed.py`,
`run_arc_agent_sandbox.sh`, `arc_experts.py`), and it builds its solver **as an OpenWorld World**. E125 should
do the same: keep its verified-world rigor, but split MODEL (FunSearch) from OBJECTIVE (codex-as-tool
traversal), and make the verified world a real `openworld.World`.

## Decisions (from brainstorming)

1. **Win handling — commit + reset per level.** Reaching a real `g.levels` bump = that level SOLVED: commit
   the action path, then **hard-reset** to a fresh Phase-1 for the next level. The win→reload transition is
   **never** folded into a `predict()`. (Tempered by decision 8 below: shared sub-dynamics may transfer.)
2. **Traversal — plan-in-imagination PRIMARY, real-env play as a confidence-gated fallback** (revised per the
   world-model review, critique #3). The whole point of a world model is to plan in imagination (Dreamer/
   MuZero). So Phase 2 plans in the World by default and only drops to real-env macro play where the model is
   **not trustworthy** (see decision 7, ensemble confidence). The env (`g.levels`) always decides the win.
3. **Cadence — per-macro.** codex proposes a short MACRO (3–5 actions) per call, previewed via the World; the
   harness executes the macro vs the real env, halting on a sim-vs-real surprise or a level-up, then re-calls
   codex with the outcome. ~1 codex call per 3–5 real actions.
4. **OpenWorld-native backbone.** Phase-1 outputs an actual `openworld.World`; its `to_spec`→`preview.graph`
   is the map, `render_card` the atlas, serve `/view` the UI. (Repo mandate: "Build solvers as OpenWorld".)
5. **Object-centric state (critique #2 — highest leverage).** The `CodePerceptor` abstracts each frame to an
   **object state** (reuse `arc3_graph.objects` / `e119/perceive.object_json`): entities (player, goal,
   toggles, walls…) with colors, sizes, positions, relations. `predict()` is **object dynamics**, not pixel
   dynamics — compact, layout-general, far easier for codex to synthesize correctly, and not pixel-brittle.
6. **Decision-equivalent fidelity gate (critique #1).** The verifier scores `predict()` on the
   **decision-relevant** state (object positions/relations + the win predicate), not every cell — a
   value-equivalent criterion (MuZero), not observation reconstruction. This un-blocks maps the pixel-exact
   gate cannot model (ls20 animation, dc22-with-actions). Pixel exact-match is retained only as a *diagnostic*
   metric, not the pass/fail.
7. **Ensemble uncertainty (critique #4).** Keep the top-k verified programs from the FunSearch population;
   their **disagreement** on a transition is an epistemic-uncertainty signal (cf. PETS ensembles; MSA
   Bayesian-over-programs). Plan-in-imagination where they agree; trigger exploration / real-env fallback
   where they diverge. This makes decision 2's confidence gate concrete (and is nearly free).
8. **Compositional transfer across levels (critique #5).** Hard reset is the v1 default, but carry a
   **library of verified rules** forward (`PhasedTransition`): on a new level, seed FunSearch with the prior
   verified sub-dynamics (e.g. movement) and synthesize **only the new regime**. Avoids re-deriving shared
   dynamics from scratch while still never folding the win→reload transition into a model.
9. **Energy is shaped from grounded outcomes (critique #6).** `goal_score` starts as codex's hypothesis but,
   once a real level-up is grounded, is **refit/shaped** toward that outcome (value learning) and used as a
   heuristic inside a *complete bounded* search (not greedy descent), to dodge spurious local minima.

## Architecture

```
solve_game(game):
  level = 0;  rule_library = []                         # verified sub-dynamics carried across levels
  while not done:
    # ── Phase 1: world synthesis (per level) ──────────────────────────────
    explore THIS level (real env)         → transitions
    perceive each frame → OBJECT STATE    (arc3_graph.objects: entities + positions + relations)
    synth.synthesize  (FunSearch + failure-memory + seed-within-level + seed from rule_library)
       gate = DECISION-EQUIVALENT (object positions/relations + win predicate), NOT pixel-exact
                                          → top-k verified programs (an ENSEMBLE) + goal_score
    world = build_world(predict_ensemble, goal_score, perceptor, current_state)   # an openworld.World
    #  to_spec(world).preview.graph = the MAP;  render_card = the atlas;  serve /view
    #  two checks: (a) decision-equivalent fidelity gate (best-effort); (b) OpenWorld round-trip (lossless)

    # ── Phase 2: codex-as-tool traversal (per level) ──────────────────────
    loop (bounded by a per-level macro budget):
      # PLAN-IN-IMAGINATION is primary:
      plan = plan_in_world(world, goal_score)           # bounded search; energy as heuristic
      if plan and ensemble AGREES along it (low disagreement):
          execute verified plan vs REAL env  → level-up? COMMIT, SOLVED;  surprise? record + re-synth
      else:                                             # low model confidence -> codex-as-tool fallback
          ctx = {predict src, goal_score src, World MAP, recent OBJECT states, g.levels,
                 actions + ensemble-previewed next-states (+disagreement), history, FAILURE MEMORY}
          macro = codex(ctx)              # 3–5 actions toward the hypothesised win
          execute macro vs REAL env: level-up → COMMIT, SOLVED;  surprise → record + Phase-1 re-synth
      on grounded level-up: SHAPE goal_score toward it (value learning)
      STALL: S iters, no level-up, no energy progress → codex RE-REASONS the goal (with failure memory)
      ABANDON: R re-reasons without progress → stop this level, report honestly (NO banked answers)

    if level solved:  rule_library += verified sub-dynamics;  level += 1   # carry transfer forward
                      fresh Phase 1 for the new regime, SEEDED from rule_library
    else:             done = True
```

### Components

1. **`build_world(predict_ensemble, goal_src, perceptor, initial_state) → openworld.World`** (reuse E112
   `build_world`, `openworld.spec`/`card`). Assembles:
   - **`CodePerceptor` (object-centric, critique #2)** — abstracts each raw `(1,64,64)` frame to an **object
     state**: entities (player/goal/toggles/walls…) with color, size, position, and relations, via
     `arc3_graph.objects` / `e119/perceive.object_json` (status/animation noise dropped by abstraction, not a
     pixel mask). Carries runnable `code`; round-trips + runs server-side (no LLM).
   - **`FunctionTransition`** wrapping the synthesized `predict()` over **object state** (the verified
     dynamics); `level_up` drives the win flag. Backed by the top-k ensemble (critique #4) so the World can
     report per-transition **disagreement** (epistemic uncertainty) to the planner.
   - **`CodeObjective`** from `goal_score` (reward = −energy, shaped from grounded outcomes per decision 9) +
     a win objective on `level_up`.
   - `initial_state` = the current object state; `card`/`name`/`description` set for the level.
   The World is the durable per-level artifact: `to_spec`→`preview.graph` map, `render_card` atlas, serve
   `/view`. Two **distinct** Phase-1 checks: (a) the **decision-equivalent fidelity gate** (critique #1) =
   `predict()` reproduces the **decision-relevant** object state + win predicate of held-out transitions (may
   be best-effort `<100%`; value-equivalent, not pixel reconstruction — pixel exact-match kept only as a
   diagnostic metric); (b) the **OpenWorld round-trip** = lossless-serialization self-consistency,
   `from_spec(to_spec(w), allow_code=True)` reproduces `w`'s own rollout (always required — structural
   integrity, independent of real-data fidelity). Both are reported in metrics.

2. **`plan_in_world(world, goal_score)` — imagination planning (PRIMARY, decision 2).** Bounded best-first
   search in the World (reuse `e125/simworld.plan` over the World's transition) using the shaped `goal_score`
   as a heuristic inside a *complete bounded* search (not greedy descent, decision 9). Returns a candidate
   plan **plus the ensemble disagreement** along it (decision 7). High agreement → execute the verified plan
   vs the real env. Low agreement (or no plan) → hand to the codex-as-tool fallback.

3. **`traverse.py` — Phase-2 loop + codex-as-tool fallback** (new). Runs `plan_in_world` first; on low model
   confidence assembles codex context (predict src, `goal_score` src, the World **map**, recent **object**
   states, `g.levels`, actions + ensemble-previewed next-states **with disagreement**, history, FAILURE
   MEMORY) and gets a macro (3–5 actions). Owns goal_score **shaping** on a grounded level-up, and the
   stall→re-reason→abandon policy. New strict macro schema `{macro:[actions], rationale, goal_note}`, prompt
   source-free + M0-audited via `codex_iso`.

4. **`agent.solve_game` — the game-level loop** (extend `agent.py`): per-level Phase-1→Phase-2, commit on
   level-up, then carry the **verified rule library** forward and seed the next level's FunSearch from it
   (decision 8); stop on abandon or `win_levels`.

5. **Macro/plan executor** (extend `execute.py`): run a plan or macro vs the real env step-by-step, halting on
   level-up (solved) or sim-vs-real surprise (record the real **object** transition for re-synth), returning
   the verified prefix.

6. **Entry** (`e125_executable_world.py`): `--mode traverse`, per-level World persistence
   (`<game>_L<n>.spec.json` + optional `.svg` card), `save_results` before asserts.

### Imagination-primary, ensemble-gated

Plan-in-imagination is the default (the world-model value proposition: action efficiency / low RHAE). The
**ensemble disagreement** (decision 7) is the confidence gate: where the top-k programs agree, a sim-plan is
trusted and only its verified prefix touches the real env; where they diverge, the model is untrustworthy and
the codex-as-tool fallback plays in the real env with the World as advisor. A sim-vs-real surprise halts and
feeds a Phase-1 re-synth (extending the World). The World need not be a perfect model to be useful — graceful
degradation under a *decision-equivalent* (not pixel-exact) fidelity bar.

## Invariants, honesty, metrics

- **Source-free + solution-free:** codex sees only frames/object-states + the World derived from its **own**
  exploration — never game source or banked solutions. M0 isolation audit (`codex_iso`) on every call; a
  tainted call is discarded. The env (`g.levels`) decides the win; codex only proposes programs/macros.
- **Honesty:** `save_results` before any assert. Report per-level outcome as-is; if a level is abandoned, say
  so. No tuning to a desired number.
- **Metrics (world-model quality, not just outcomes — critique #7):** levels solved/game · **real-env actions
  used (RHAE proxy)** · codex calls · **decision-equivalent fidelity %** (and pixel-exact % as a diagnostic) ·
  **ensemble disagreement rate** · **model-based-planning success rate** (sim-plans that hold up in the real
  env) · **imagination horizon** (steps the model stays decision-accurate) · macros/level. **Head-to-head vs
  the loose sweep agent** on the same maps.

## Testing (TDD, hermetic)

Mock codex + a synthetic 2-level game (each level a small grid-mover with a distinct win):
- object perceptor: frame → object state (entities + relations); stable under cosmetic/animation noise.
- `build_world` → an `openworld.World` whose `to_spec`/`from_spec` round-trips and whose `preview.graph` is
  non-empty (the map).
- decision-equivalent gate: accepts a `predict()` correct on object state + win predicate even if some
  cosmetic cells differ; rejects one wrong on a decision-relevant object.
- ensemble: ≥2 programs agreeing → low disagreement (plan); disagreeing → high disagreement (fallback).
- plan-in-imagination primary: agreeing ensemble + valid sim-plan → executes the verified plan, level solved.
- codex fallback: low agreement → mock codex macro → solves; halt-on-surprise records the real transition and
  triggers a (mocked) re-synth.
- stall → re-reason → abandon: a mock codex that never finds the win → bounded restarts → honest abandon.
- `solve_game` + transfer: a 2-level game solved where level-1's FunSearch is **seeded from level-0's verified
  rule library** (and still hard-resets the win handling).

Live smoke (one clean map, e.g. dc22): Phase-1 builds + round-trip-verifies an object-state World; Phase-2
plans-in-imagination (or falls back) and runs ≥1 plan/macro vs the real env; report whether a real level-up is
grounded — honestly, no banked answers.

## Reuse

| Piece | Reuses |
|---|---|
| object perceptor | `arc3_graph.objects`, `e119/perceive.object_json` / `contrastive_diff` |
| Phase-1 synth (now object-state, decision-equivalent gate, top-k ensemble) | existing `e125/synth.py` (FunSearch + failure memory + seed-within-level) |
| World build + spec/card/serve | E112 `build_world`; `openworld.World`/`FunctionTransition`/`CodePerceptor`/`CodeObjective`/`spec`/`card`/`serve` |
| imagination plan / energy | `e125/simworld.py` (`plan`, `_energy`) over the World |
| macro/plan execute + halt | `e125/execute.py` (extend for macros + level-up) |
| real-env fallback | `e125/explorer.goal_directed_collect` (energy-descent real-env play inside Phase 2) |
| codex + telemetry + audit | `e124/codex_iso`, `capture_lib.codex_record`, strict `--output-schema` |

New code ≈ object perceptor + decision-equivalent gate + top-k ensemble in `synth`; `build_world` glue;
`plan_in_world`; `traverse.py`; `agent.solve_game` + rule-library transfer; macro schema/prompt; per-level
World persistence. The core harness (verify/synth/simworld/execute) is reused and extended, not rewritten.

## Searcher seam (noted, deferred — YAGNI for now)

Phase-1's `synthesize` is one **searcher** behind the harness-owned evaluator (the gate). AlphaEvolve-style
(no public impl → would be an approximation), DEAP-GP (typed grid-op primitives), and PySR (symbolic
regression — best for the `goal_score` energy) are candidate alternative searchers, scored by the same gate
for a clean head-to-head. Not built in this phase.

## Out of scope (YAGNI)

- Multi-island FunSearch migration/reset (single island suffices at our budget).
- Building AlphaEvolve/DEAP/PySR searchers (deferred; the seam is noted).
- Wiring E125 into the live sweep router until it shows a head-to-head result.
- All 25 games — pilot on the cleanest maps first (dc22, then a couple of headroom maps).

## Risks

- **Object extraction quality (new top risk from critique #2).** The whole design now rests on `arc3_graph.objects`
  abstracting frames into the *right* entities. If it under/over-segments (merges player+wall, splits one
  object), the object state is wrong and so is every downstream model. Mitigation: validate the perceptor on
  the probe frames (stable entity count/identity across steps) before trusting it; fall back to a finer-grained
  (per-cell-region) state if abstraction is unreliable on a map. This is the first thing to verify live.
- **Decision-equivalent gate needs a decision-relevance definition.** "Decision-relevant" must be made
  concrete (which objects/relations + the win predicate). Risk of choosing it wrong (masking out something that
  *does* matter). Mitigation: derive it from objects that *move/change* under actions + the goal entity; keep
  pixel-exact % as a diagnostic to catch silent abstraction errors.
- **Ensemble cost** — keeping/scoring top-k programs is k× evaluation (cheap, in-process) but the k programs
  come from extra FunSearch iterations (codex cost). Bounded by reusing the population FunSearch already
  produces; k small (2–3).
- **Energy local minima** — a codex-guessed `goal_score` may trap greedy descent; mitigated by using it as a
  heuristic in a complete bounded search and shaping it from grounded outcomes (decision 9).
- **codex traversal cost/latency** — bounded by per-macro cadence + per-level macro/iteration budget; telemetry
  records spend.
- **Real-action budget** — the codex-fallback path costs more real actions than imagination planning; the
  head-to-head measures RHAE honestly. If E125 needs *more* real actions than the loose agent, report it.
- **Compositionality across levels** — hard-reset *win handling* + rule-library *transfer* (decision 8); never
  one growing model that must predict the win→reload transition.

## Reproducibility

`~/.arcv/bin/python` (`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`); codex `~/.local/bin/codex` (`gpt-5.5`),
strict `--output-schema`. Every codex call captured to `experiments/results/e125_traces/`; per-level Worlds
persisted as specs/cards. Results via `save_results("e125_executable_world", ...)`.
