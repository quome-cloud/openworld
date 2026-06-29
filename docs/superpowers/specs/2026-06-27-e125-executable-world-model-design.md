# E125 — Structured executable-world-model agent (design)

**Date:** 2026-06-27
**Branch:** `arc3-runner-fix`
**Status:** design approved; implementation pending

## Goal

Close the gap between our ARC-AGI-3 sweep (8/25 full games, loose agentic recipe) and the documented SOTA —
**baseline1 (Rodionov), 63.7%, GPT-5.5, "Executable World Models", arXiv 2605.05138** (≈ the repo's own
`papers/arc-3/RECIPE.md`). The gap is **not the recipe** (the sweep already tells the agent to write
`predict()`, verify, plan) — it is **structural enforcement and action-efficiency**. E125 makes the
verify→plan→execute loop a set of **harness-enforced deterministic gates**, with codex (GPT-5.5) only ever
*proposing* `predict()` code; the Python harness owns verification, planning, execution, and repair.

**Success criterion (falsifiable):** beat E124 by **planning in simulation** — solve a level E124's real-env
search could not (e.g. `g50t` L0, a 17-action procedure: blind BFS = 5¹⁷, hopeless) by searching a *verified
code world model* and executing only verified plans. Headline result: **head-to-head vs the loose sweep
agent** on the same walls — does harness-enforced structure + plan-in-sim beat model-discretion, with fewer
real-env actions (RHAE)?

## Decisions (from brainstorming)

1. **Success:** beat E124 via plan-in-simulation (and head-to-head vs the loose sweep agent).
2. **Direction:** a **standalone structured agent**, tested against the loose sweep agent — the clean
   experiment is *structure vs model-discretion*.
3. **Approach: OpenWorld-native, maximal reuse.** codex writes the `predict()` body; the harness wraps it as
   a `World` `FunctionTransition`, verifies via the E123 fidelity gate, plans via `e119/planner` against the
   World, executes+halts via the E122 surprise monitor, re-synthesizes via E123, composes via
   `PhasedTransition`. Free-form 3-file codebases (baseline1 literal) are out of scope — they trade our clean
   verification gate for power we don't yet need.
4. **Brain:** OpenAI **Codex** (`codex exec`, default `gpt-5.5`), reusing E124's `codex_iso` (source-free
   run + M0 audit), `sandbox_exec`, and telemetry.

## Thesis (what makes E125 ≠ the sweep)

The sweep *asks* the agent to verify+plan in its own context (model discretion); E125 makes those
**deterministic harness gates**. Reliability moves from the model's good behavior to the scaffold:

| Concern | Loose sweep agent | E125 (structured) |
|---|---|---|
| Verification | "verify on held-out" (agent's choice) | **harness gate**: `predict()` rejected unless it exact-matches recorded transitions |
| Planning | agent reasons + acts in the **real env** | **plan in simulation** (search the verified World), execute only verified plans |
| Mismatch | agent notices (or not) | **harness halts** on any sim-vs-real divergence (E122), adds the transition, re-synthesizes |
| Action budget (RHAE) | wasted on real-env exploration | minimized — search is free in sim |
| Across levels | ephemeral context | durable World composed via `PhasedTransition`; MDL refactor on stall |

**Invariant:** codex is a proposal engine inside a verification loop, never an authority. The env decides
correctness (a replay-verified `levels_completed` bump); a wrong `predict()` is caught by the gate or the
executor halt, never by trusting the model.

## Architecture

New package `experiments/e125/`. The loop is **iterative model-extension** (not one-shot synthesize-then-plan
— for a deep procedure you plan as far as the verified model allows, execute, hit new territory, explore
there, re-synthesize, replan, pushing the model's frontier toward the goal).

```
explore -> synthesize predict() -> [VERIFIER GATE] -> build World
        -> plan-in-sim toward predicted level-up -> [PLAN VERIFIER]
        -> execute vs real env, step-by-step
              -> sim==real: keep going
              -> sim!=real (SURPRISE): HALT, add transition, re-synthesize, replan
        -> on level-up: compose World forward (PhasedTransition), continue
        -> on stall: MDL refactor (ask codex to simplify), re-explore
```

### Components (each an enforced gate)

1. **`explorer.py`** — budgeted change-seeking policy (reuse `e112` action model: directional actions +
   pixel-inferred click targets) collects exact `(s, a, s′, level_up)` transitions. Env = ground truth.
2. **`synth.py` — synthesizer + verifier gate.** codex writes
   `predict(frame, action) -> (next_frame, level_up: bool)` (Python, numpy). The harness **accepts only if it
   exact-matches every held-out transition** (masked-frame equality + level_up equality — the E123 fidelity
   check). On any miss → feed the counterexample(s) back → re-propose (bounded retries). Telemetry per call
   (`capture_lib.codex_record`); source-free audit (`codex_iso`); code run sandboxed (`sandbox_exec`).
3. **`simworld.py` — the World + plan-in-sim.** Wrap the accepted `predict()` as a `World`'s
   `FunctionTransition`; expose a `SimGame` (reset/step/frame/levels/done driven by `predict()`, NOT the env)
   so `e119/planner.search_level` searches **in imagination** for a trajectory whose predicted `level_up` is
   true. Free and deep — this is where the depth-17 wall dissolves.
4. **`execute.py` — executor + halt + resync.** Dispatch the planned actions against the **real** env
   step-by-step; at each step compare the real masked next-frame to `predict()`'s. **On any divergence
   (E122 surprise): halt at that step, add the real transition to the dataset, re-synthesize, replan from the
   reached state (E123 replay-to-boundary).** A real env-verified `levels_completed` bump = solved + banked.
5. **`agent.py` — the loop** (orchestration above) + **compose** (carry the level's World forward via
   `PhasedTransition`) + **MDL refactor** (on stall, ask codex to simplify the code; reject if it breaks the
   regression set).
6. **`e125_executable_world.py`** — entry: `--games`, `--mode {structured,loose}`, `--budget`, `--model`,
   `save_results` before asserts.

## Head-to-head experiment (the result)

Run E125 (`structured`) vs the sweep's **loose** agent on the **same** pilot games/walls, same brain where
possible. Report a table: **levels solved · real-env actions used (RHAE proxy) · world-model verification
rate · codex calls/cost · plan depth reached in sim.** Hypothesis: structure + plan-in-sim solves walls the
loose agent can't, with fewer real actions. Control: a level E124's real-env search could not crack, E125
cracks by planning in sim.

## Reuse map

| Piece | Reuses |
|---|---|
| explorer | `e112` action model; `e119/perceive` (`status_mask`, `state_key`) |
| synth + telemetry + sandbox | E124 `codex_iso`, `sandbox_exec`, `capture_lib.codex_record` |
| verifier gate | E123 round-trip fidelity (masked-frame + level_up exact match) |
| World + plan-in-sim | `openworld.World`/`FunctionTransition` (E112 `build_world`); `e119/planner` |
| executor halt | E122 `OnlineRegimeMonitor` (sim-vs-real mismatch) |
| resync + compose | E123 replay-to-boundary; `openworld.PhasedTransition` (E121/E123) |

New code ≈ orchestration only (`explorer/synth/simworld/execute/agent`). ~80% reuse.

## Source-free, honesty, metrics

- **Source-free:** codex sees only collected transitions (frames + actions + level_up), never game source;
  M0 audit (`--json` event-stream) on every call; `predict()` is verified against **env-recorded** history,
  so codex cannot fake a solve.
- **Honesty:** `save_results` BEFORE any assert; report the loose-agent baseline and the E124 control as-is;
  if structure does not beat loose, say so plainly.
- **Metrics:** levels solved/game; real-env actions used (RHAE proxy); world-model verification rate (held-out
  exact-match); codex calls + token cost; plan depth reached in sim.

## Testing

Hermetic (mock codex, synthetic game):
- verifier gate **rejects** a `predict()` that mispredicts a held-out transition, **accepts** an exact one.
- `SimGame` + planner **find a winning trajectory** in a synthetic World whose real env is too deep for BFS.
- executor **halts on an injected sim-vs-real mismatch** and triggers resync.
- full `agent` loop solves a synthetic game end-to-end with a mock codex that writes the right `predict()`.

Live smoke (one game): codex synthesizes a `predict()`; the gate verifies it against **real** recorded
transitions; plan-in-sim finds a trajectory; execute one verified step against the real env.

## Milestones

0. **Isolation** — reuse E124's M0 gate (already green: codex returns JSON, 0 source reads, audit clean).
1. **Synthesizer + verifier gate** — codex writes `predict()` for one pilot game; the gate checks exact-match
   on held-out **real** transitions. **Gate: can codex model the dynamics at all?** (If `predict()` can't
   pass the fidelity check on real data, that is the new bottleneck — stop and report.)
2. **Plan-in-sim + execute-verify** — solve a level by planning in the verified World and executing the
   verified plan (the **E124-beating proof**: a level real-env search couldn't crack, cracked in sim).
3. **Full loop + head-to-head** — explore→synth→plan→execute→halt→resync→compose vs the loose sweep agent on
   the pilot (`g50t` + a couple of headroom games like `sp80`/`dc22`); paper §/figure if a real result.

## Out of scope (YAGNI)

- Free-form multi-file codebases codex edits directly (baseline1 literal) — our `predict()`-as-World gives
  the same verified-world-model property with clean reuse.
- GPT-5.5-vs-Claude as a separate study (the head-to-head holds the brain fixed where possible).
- Wiring E125 into the live sweep router until it shows a result.
- All 25 games — pilot first.

## Risks

- **Chicken-and-egg / coverage:** to plan a level-up in sim, the explored transitions must reach (or the
  model must generalize to) the win-relevant dynamics. Mitigation = the iterative extend-the-frontier loop
  (plan as far as verified → execute → hit new territory → explore → resync). If the model never reaches the
  level-up transition, E125 fails honestly — but with far fewer real actions than blind search.
- **codex can't model the dynamics** (Milestone-1 gate fails) — a real, reportable outcome and a sharper
  finding than E124's.
- **Generated-code execution** — subprocess + timeout (E124 `sandbox_exec`); robustness, not security.
- **Cost/latency** — bounded by synthesis calls per game (not per search node — planning is in-process in the
  code model); telemetry records spend.

## Reproducibility

Runs with `~/.arcv/bin/python`; codex at `~/.local/bin/codex` (`gpt-5.5`). Every codex call captured to
`experiments/results/e125_traces/` and the synthesized `predict()` code retained. Results via
`save_results("e125_executable_world", ...)`.
