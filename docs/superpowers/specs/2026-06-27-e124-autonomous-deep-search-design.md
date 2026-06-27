# E124 — Autonomous deep search, codex-steered (design)

**Date:** 2026-06-27
**Branch:** `arc3-runner-fix`
**Status:** design approved; implementation pending

## Goal

Attack the **goal-inference wall** (E88–E90, E119): on ARC-AGI-3, blind search cannot crack a level whose
win condition is an ordered *procedure* rather than a static *state* score — E119's search-only control banks
**0 levels** autonomously. This experiment builds an **autonomous deep search** loop in which **OpenAI
Codex compiles the goal** that steers a programmatic best-first search, while the **environment decides
correctness**. Success is defined narrowly and falsifiably: **beat the blind-BFS control** — solve ≥1 level
with NO trace guidance, same budget, on games where blind BFS banks 0.

This is autonomous (no trace, unlike E123 which was trace-guided) and deep (chains the goal across the
compositional levels using the E121–123 surprise/resynthesis machinery).

## Decisions (from brainstorming)

1. **Success criterion:** beat the blind-BFS control (a measurable lift), reported honestly even if small.
2. **Brain:** OpenAI **Codex** via the local `codex exec` CLI (v0.142.3, authenticated at `~/.codex`). No
   `OPENAI_API_KEY` / `openai` python package is present — the CLI is the integration surface.
3. **Integration shape:** codex **steers a search loop** (E119-style) — it compiles a goal (`score_fn` +
   action macros) that *orders* search; the env verifies. NOT an agentic end-to-end codex solver.
4. **Scope:** full integrated loop (A), built **MVP-first** (single-level lift, then deep chaining).
5. **Telemetry:** first-class reproducibility capture of every codex call (model+version, prompt, response,
   parsed goal, decision, timings) — mirrors the existing `arc3_traces` capture for the claude runner.

## Invariant (non-negotiable)

Codex only **compiles a goal that orders search**. Correctness comes entirely from deterministic env
replay: a level is solved iff `levels_completed` rises, replay-verified, then banked. A wrong or abstained
goal costs search budget, never a false solve. This is the same trust model as E119 (`score_fn` priority)
and is what makes the result honest.

## Architecture

New package `experiments/e124/`, reusing E119 (`planner`, `abstain`, `perceive`) and E121–123
(surprise/resynth/world). Five units, each independently testable:

| Unit | Responsibility | Depends on |
|---|---|---|
| `codex_goalc.py` | Call `codex exec`; parse a goal = `score_fn(frame)->float` + action macros; sandbox-validate the code; best-of-N + abstain; emit telemetry | codex CLI, `codex_capture`, E119 `abstain` |
| `codex_capture.py` | Reproducibility dataset: one JSONL record per codex call | stdlib only |
| `search.py` | Best-first search guided by `score_fn` over pixel + macro candidates; env-verifies a level-up | E119 `planner` |
| `deep.py` | Per-game orchestration: probe → compile goal → search → on level-up surprise/resynth → re-compile for new regime → chain; bank replay-verified | E121–123, the above |
| `e124_autonomous_search.py` | Entry: `--mode blind\|codex`, `--games`, `save_results` before asserts | all |

## §1 codex_goalc — the goal compiler

- **Call:** `codex exec` headless, `--cd <clean scratch dir>` (no game source) + read-only sandbox. The
  prompt is **source-free** and contains the observed frames as inline data (rendered grids), the action
  API (`g.step(a)` / `g.step(6,x,y)`, available actions), and the discovered `(s,a)->s'` dynamics summary.
- **Return:** strict **JSON** `{score_fn: "<python source>", macros: [[...]], rationale: "..."}`. The model
  reasons over provided data; it does not touch the env or the source.
- **score_fn contract:** `def score_fn(frame: np.ndarray[64,64]) -> float` where `frame` is the **masked**
  64×64 frame (status bar zeroed, consistent with the rest of the pipeline); higher = closer to raising
  `levels`. Executed in a **restricted namespace** (numpy only; no imports, no IO, no builtins beyond a safe
  subset), per-call **timeout**, wrapped in `try/except`. Any failure → that hypothesis scores 0 and is
  dropped. (Mirrors E119 `slm.compile_predicate` / `satisfiable`.)
- **macros:** named action sequences (e.g. `[[1],[1],[6,12,30]]`) added to the search candidate set.
- **Bayesian layer:** call codex N times (or one call returning N hypotheses); cluster hypotheses by their
  **behavioral effect** on probed frames (E119 `abstain.best_of_n`); weight by cluster mass; **abstain**
  below τ — on abstain, the level falls back to blind BFS rather than chasing a bad goal.
- **Source-free gate:** an audit scans the codex telemetry for any sign of source access; a tainted call is
  discarded (same gate as the claude runner).

## §2 codex_capture — telemetry / reproducibility

One JSONL record per codex call → `experiments/results/e124_traces/`. Fields:

- `model` and resolved **model version** (from `codex --version` and any version string in the response)
- exact **prompt** (also written to a `prompts/` sidecar), raw **response** (a `transcripts/` sidecar)
- parsed `score_fn` source + macros; sandbox-validation result (ok / error / timeout)
- the N-sample best-of-N cluster summary + abstain decision (committed hypothesis or `abstain`)
- token usage + latency if available; ISO `started`/`finished`
- `game`, `level`, `regime`, a deterministic `run_id`, and a content `hash`

Fully replayable and audit-scannable, structurally parallel to `arc3_traces`.

## §3 search loop + control

Reuse `e119/planner.search_level(game, candidates_fn, key_fn, budget, score_fn)` unchanged. One code path,
two modes:

- **codex mode:** `score_fn` = compiled codex predicate; `candidates_fn` = pixel-inferred targets (E112:
  small connected components + rare colors; directional actions) **∪** codex macros.
- **blind mode (control):** `score_fn=None`, **same** `candidates_fn`, **same budget**.
- A level is solved only on an env-verified `levels_completed` bump, then banked.

## §4 deep chaining

```
regime = 0; actions = []
while not done and budget remains:
    goal = codex_goalc.compile_goal(frames, api, dynamics, regime)   # best-of-N + abstain
    seq  = search.search_level(PrefixGame(game, actions), cands, budget, goal.score_fn)
    if seq raises levels (env-verified):
        actions += seq; bank()
        # E122 surprise fires on the level-up -> E123 replay-to-boundary -> NEW regime
        # -> re-compile the goal for the new rules -> continue DEEP
        regime += 1
    else:
        record abstain/blind-fallback; break if no progress this regime
```

Codex re-compiles the goal **per regime**, chaining goal-inference across compositional levels via the
surprise/resynthesis machinery — autonomous depth, no trace.

## §5 measurement, honesty, source-free

- **Headline:** autonomous levels banked, **codex-guided minus blind**, same budget, over the pilot set —
  the lift attributable to codex's goal inference. Plus abstention rate, codex calls/level, token cost.
- **Honesty:** `save_results` BEFORE any assert; report the blind baseline as-is (expected 0); if codex does
  not beat blind, say so plainly — it is a falsifiable test, not a guaranteed win.
- **Source-free:** clean dir, source-free prompt, audit gate on telemetry; tainted calls discarded.
- **Pilot set:** short directional games where a good goal could plausibly help search —
  `tn36, ar25, vc33, ls20` — for signal without burning tokens on 25 games. (Click games are a later
  extension once the directional result holds.)

## §6 testing

Hermetic (no codex, no env):

- `codex_goalc` with a **mock** codex (canned JSON): parses + sandboxes `score_fn`; **rejects** broken /
  malicious / timing-out code and abstains correctly.
- `abstain.best_of_n` clusters + abstains on synthetic behaviors (reuse/extend E119 tests).
- `search` with a synthetic env + a known-good `score_fn` **beats** blind on a crafted task; with no
  `score_fn` it matches the blind baseline.
- `codex_capture` round-trips a record (write → read → fields intact).

Plus a **live smoke** on one pilot game: real `codex exec`, confirm a parseable goal + a telemetry record
land, source-free audit clean.

## §7 milestones

1. **MVP (B):** `codex_goalc` + `codex_capture` + single-level `codex` vs `blind` on the pilot. Prove the
   lift exists and telemetry lands. **Gate:** does codex-guided beat blind on ≥1 level? If no, stop and
   report honestly before building depth.
2. **Deep (A):** add `deep.py` surprise/resynth chaining across levels.
3. **Paper:** §"E124: codex-steered autonomous deep search" + a lift-vs-blind figure — only if there is a
   real result.

## Out of scope (YAGNI)

- The agentic end-to-end codex solver (integration shape C) — deliberately set aside for a clean search
  comparison.
- Click-heavy games in the first pilot.
- OpenAI API / `openai` python integration — the `codex exec` CLI is the only integration surface here.
- Wiring E124 into the live sweep router — it is a standalone experiment until it shows a result.

## Risks

- **Codex may not beat blind** on these procedure goals — that is an acceptable, reportable outcome.
- **Cost:** `codex exec` spends OpenAI tokens; bounded by best-of-N calls per level (not per search node) +
  the small pilot set. Telemetry records token usage so cost is auditable.
- **Sandbox escape of a generated `score_fn`** — mitigated by a restricted namespace, no imports/IO,
  timeout; a failure degrades to abstain, never to a crash or a false solve.
- **codex exec agentic drift** (it could try to run tools) — mitigated by clean dir + read-only sandbox +
  a prompt that asks only for JSON; the audit catches source access.

## Reproducibility

Runs with the arc venv (`~/.arcv/bin/python`); codex via `~/.local/bin/codex`. Every codex call is captured
to `experiments/results/e124_traces/`. Results JSON via `save_results("e124_autonomous_search", ...)`.
