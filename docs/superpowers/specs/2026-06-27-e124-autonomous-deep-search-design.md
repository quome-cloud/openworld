# E124 — Autonomous deep search, codex-steered (design)

**Date:** 2026-06-27
**Branch:** `arc3-runner-fix`
**Status:** design approved (revised after expert review); implementation pending

## Goal

Attack the **goal-inference wall** (E88–E90, E119): on ARC-AGI-3, blind search cannot crack a level whose
win condition is an ordered *procedure* rather than a static *state* score — E119's search-only control banks
**0 levels** autonomously. This experiment builds an **autonomous deep search** loop in which **OpenAI
Codex compiles the goal structure** that steers a programmatic search, while the **environment decides
correctness**. Success is defined narrowly and falsifiably: **beat the blind-BFS control** — solve ≥1 level
with NO trace guidance, same budget, on games where blind BFS banks 0.

Autonomous (no trace, unlike E123 which was trace-guided) and deep (chains the goal across the compositional
levels using the E121–123 surprise/resynthesis machinery).

## Decisions (from brainstorming + expert review)

1. **Success criterion:** beat the blind-BFS control (a measurable lift), reported honestly even if small/zero.
2. **Brain:** OpenAI **Codex** via the local `codex exec` CLI (v0.142.3, authenticated at `~/.codex`). No
   `OPENAI_API_KEY` / `openai` package present — the CLI is the integration surface. Use `--output-schema`
   (force JSON shape) + `-o` (capture final message) + `--json` (event stream, for the source-free audit).
   **Default model: GPT-5.5** (OpenAI's flagship reasoning/coding model — best fit for goal-synthesis),
   passed via `codex exec -m <model>` and configurable per run; the resolved model+version is captured in
   telemetry for every call.
3. **Integration shape:** codex **steers a search loop** — it compiles a goal *structure* that *orders*
   search; the env verifies. NOT an agentic end-to-end codex solver.
4. **Guidance primitive (revised — expert review #1):** the primary output is an **ordered list of subgoals**
   (each an executable predicate) **+ macro-actions** (multi-step options), NOT just a scalar `score_fn`. A
   scalar state-potential alone is the very thing that fails on procedure goals; subgoal decomposition +
   macros-as-options make the search procedure-aware and collapse depth. `score_fn` is one optional channel.
5. **Scope:** full integrated loop (A), built **isolation-spike-first, then MVP, then depth**.
6. **Telemetry:** first-class reproducibility — **reuse `capture_lib.py`** + the existing audit (extend, do
   not duplicate); add a **record/replay-from-cache** execution mode for deterministic re-runs.

## Invariant (non-negotiable)

Codex only **compiles a goal structure that orders search**. Correctness comes entirely from deterministic
env replay: a level is solved iff `levels_completed` rises, replay-verified, then banked. A wrong or
abstained goal costs search budget, never a false solve (same trust model as E119).

## Architecture

New package `experiments/e124/`, reusing E119 (`planner`, `abstain`, `perceive`), E121–123
(surprise/resynth/world), and `scripts/capture_lib.py`. Units (each independently testable):

| Unit | Responsibility | Depends on |
|---|---|---|
| `codex_goalc.py` | Call `codex exec` (or replay from cache); parse a goal = **ordered subgoals + macros (+ optional score_fn)**; subprocess-sandbox-validate the predicate code; best-of-N + abstain; emit telemetry | codex CLI, `capture_lib`, E119 `abstain` |
| `codex_iso.py` | Source-free isolation: build the clean-dir invocation, parse the `--json` events, **audit** for any source read, decide taint | stdlib |
| `search.py` | **Subgoal-by-subgoal hill-climbing** + macros as multi-step options + optional `score_fn`; env-verifies a level-up | E119 `planner` |
| `deep.py` | Per-game orchestration: probe → compile goal → search → on level-up surprise/resynth → re-compile for new regime → chain; bank replay-verified; cost circuit-breaker | E121–123, the above |
| `e124_autonomous_search.py` | Entry: `--mode`, `--games`, `--replay`, `save_results` before asserts | all |

## §0 codex isolation spike (MILESTONE 0 — blocking, expert review #2)

The `read-only` sandbox still permits reading the whole disk (there is a `disk-full-read-access` permission),
so codex could `cat` a game's `<game>.py` answer key by absolute path — a cardinal-rule violation. **Before
any real run we must prove isolation:**

- Invoke `codex exec` with `--output-schema` + a **pure-data prompt** (frames + API inline) so it has no
  reason to use shell tools, `--cd <clean dir>` (no source), `-s read-only`, `--json`, `-o <final>`.
- Parse the `--json` event stream; **audit** every shell/file event for reads of source paths
  (`experiments/*.py` game files, `arc_agi`, `environment_files`, the game id). Any such read ⇒ the call is
  **tainted and discarded**.
- Smoke-test whether `read-only` blocks out-of-workspace reads on this machine. If it does **not**, harden:
  tighten `-c sandbox_permissions` to deny disk reads outside the workspace, or run codex behind an OS
  sandbox. **Do not run the experiment until a green isolation check exists.**

This spike's output is a reusable `codex_iso.run(prompt, schema) -> (final_json, events, tainted)`.

## §1 codex_goalc — the goal compiler

- **Call:** via `codex_iso.run` (above). The **source-free** prompt contains the observed frames as inline
  data (rendered grids), the action API, and the discovered `(s,a)->s'` dynamics summary.
- **Return (forced by `--output-schema`):** JSON
  `{subgoals: [{name, predicate_src}], macros: [[...]], score_fn_src?: "...", rationale: "..."}`.
  - **subgoals** — an *ordered* list; each `predicate(frame)->bool` marks a sub-state achieved. The search
    pursues them in order (procedure-aware).
  - **macros** — named multi-step action sequences added to the search as **options** (one search step can
    apply a whole macro), collapsing effective depth.
  - **score_fn** — optional scalar potential, used only to order candidates *within* the current subgoal.
- **predicate/score_fn contract:** operate on the **masked** 64×64 frame. Validated and executed in a
  **separate subprocess** with a hard timeout + resource cap (expert review #6: namespace restriction is NOT
  a security boundary; codex is not adversarial, so this is **robustness**, not a security claim). Failure /
  timeout ⇒ that hypothesis is dropped.
- **Bayesian layer:** best-of-N proposals → cluster by **behavioral effect** on probed frames (E119
  `abstain.best_of_n`) → weight by mass → **abstain** below τ (fall back to blind for that regime).

## §2 telemetry / reproducibility (reuse `capture_lib`)

Extend `scripts/capture_lib.py` (do not duplicate). One record per codex call → `experiments/results/e124_traces/`:

- `model` + resolved **version** (`codex --version` + any version in events); exact **prompt** (sidecar);
  raw **response** + `--json` event stream (sidecar); parsed subgoals/macros/score_fn; subprocess-validation
  result; best-of-N cluster + **commit-or-abstain** decision; token usage + latency; ISO `started`/`finished`;
  `game`/`level`/`regime`; `run_id`; content `hash`; **isolation audit verdict** (clean/tainted).
- **Record/replay (expert review #4):** `--replay <dir>` reads captured responses by `run_id`/hash instead
  of calling codex ⇒ deterministic re-runs + zero-cost dev iteration. The downstream search is deterministic
  given a fixed response, so a replay reproduces the experiment exactly.

## §3 search loop + ablation control

`search.py` wraps `e119/planner` to add subgoal/macro awareness:

- **Subgoal hill-climbing:** pursue subgoal *k* (best-first toward `predicate_k`, macros as options); on
  achieving it, advance to *k+1*; a level-up (env-verified) at any point = solved.
- **Macros as options:** candidate set = pixel-inferred targets (E112) ∪ codex macros (applied atomically).
- **Ablation rungs (expert review #3 — clean attribution), same budget each:**
  1. `blind` — BFS, pixel candidates, no `score_fn`/subgoals/macros (the control floor).
  2. `blind+macros` — BFS, pixel ∪ codex macros (isolates the value of macros).
  3. `subgoals` — subgoal hill-climbing, pixel candidates only (isolates the value of decomposition).
  4. `full` — subgoals + macros (+ score_fn).
- A level is solved only on an env-verified `levels_completed` bump, then banked.

## §4 deep chaining

```
regime = 0; actions = []
while not done and within cost budget:
    goal = codex_goalc.compile_goal(frames, api, dynamics, regime)   # subgoals+macros, best-of-N, abstain
    seq  = search.run(PrefixGame(game, actions), goal, budget, rung)  # subgoal hill-climb
    if seq raises levels (env-verified):
        actions += seq; bank(); committed_by_codex[regime] = not goal.abstained
        # E122 surprise fires on the level-up -> E123 replay-to-boundary -> NEW regime
        regime += 1
    else:
        record abstain/no-progress; break if stalled
```

Codex re-decomposes subgoals **per regime**, chaining goal-inference across compositional levels via the
surprise/resynthesis machinery.

## §5 measurement, honesty, source-free

- **Headline:** autonomous levels banked across the **ablation ladder** (rungs 1–4), same budget — the lift
  of decomposition vs macros vs both over the blind floor. Count **codex-committed** levels separately from
  abstain→blind-fallback (expert review #5), so the lift is not contaminated.
- **Pilot set (expert review #7 — verify headroom first):** before committing, **measure the blind-BFS
  baseline** (E119 `--mode search`) on candidate games and keep only **short** games where blind banks ~0 at
  a reachable level. The current banked depths (`tn36`=2, `ar25`=8, `vc33`=3, `ls20`=2) are from the claude
  agents, *not* blind BFS — they do not establish headroom. `ar25` (253-action solution) is too deep for
  search; **drop it or target L0 only**. Favour short-solution games (`tn36`~17, `ls20`~62, `vc33`~33 steps).
- **Cost circuit-breaker (expert review #8):** hard caps on total codex calls, per-game wall-clock, and a
  kill switch; telemetry records token spend so cost is auditable.
- **Honesty:** `save_results` BEFORE any assert; report the blind floor as-is (expected ~0); if no rung beats
  blind, say so plainly — a falsifiable test, not a guaranteed win.
- **Source-free:** §0 isolation + audit gate; tainted calls discarded; the same standard as the claude runner.

## §6 testing

Hermetic (no codex, no env):

- `codex_iso` audit: synthetic `--json` event streams with/without a source read ⇒ correct taint verdict.
- `codex_goalc` with a **mock** codex (canned JSON): parses subgoals/macros; subprocess-sandboxes a
  predicate; **rejects** broken/timing-out code and abstains; honours `--replay`.
- `abstain.best_of_n` clusters + abstains on synthetic behaviors.
- `search`: subgoal hill-climbing + a known macro **solves** a crafted synthetic task that single-step BFS
  cannot within budget (demonstrates the depth-collapse value); rung 1 matches the blind baseline.
- `capture_lib` record round-trips (write → read → fields intact); replay returns the recorded response.

Plus a **live smoke** on one short pilot game: real `codex exec`, parseable goal lands, telemetry written,
isolation audit clean.

## §7 milestones

0. **Isolation spike (blocking):** `codex_iso` proves codex cannot read game source (or hardens until it
   can't). No real run before this is green.
1. **MVP:** `codex_goalc` + telemetry + the ablation ladder on a single level of the verified-headroom pilot.
   **Gate:** does any rung beat blind on ≥1 level? If no, stop and report honestly before building depth.
2. **Deep:** add `deep.py` surprise/resynth chaining across levels.
3. **Paper:** §"E124: codex-steered autonomous deep search" + ablation-ladder figure — only if real result.

## Out of scope (YAGNI)

- The agentic end-to-end codex solver (integration shape C).
- Click-heavy games in the first pilot.
- OpenAI API / `openai` package — `codex exec` CLI is the only surface.
- Wiring E124 into the live sweep router until it shows a result.

## Risks

- **No rung beats blind** on procedure goals — acceptable, reportable outcome.
- **Source-free isolation** is the top risk; §0 spike is mandatory and blocking.
- **Depth explosion** within a level — mitigated by macros-as-options + subgoal decomposition; if a goal is
  deep with no decomposition, the search will still fail (reported honestly).
- **Cost** — bounded by best-of-N per regime + small pilot + circuit-breaker; token spend in telemetry.
- **Generated-code execution** — subprocess + timeout + resource cap; robustness not security (codex is not
  adversarial); a failure degrades to abstain, never a crash or false solve.

## Reproducibility

Runs with the arc venv (`~/.arcv/bin/python`); codex via `~/.local/bin/codex`. Every codex call captured to
`experiments/results/e124_traces/` and replayable with `--replay`. Results via
`save_results("e124_autonomous_search", ...)`.
