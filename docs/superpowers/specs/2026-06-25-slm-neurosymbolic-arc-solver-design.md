# E119 — Neuro-symbolic SLM solver for ARC-AGI-3

**Date:** 2026-06-25
**Status:** design (pending review)

## 1. Research question

Can a *set* of small language models (SLMs, ≤~9B), inside a verification-grounded
neuro-symbolic harness, solve interactive ARC-AGI-3 games — and how much of the
result comes from the **harness** vs the **model**?

This is the [combined-paper](../../../../judge-experiments-papers/combined-paper)
"engineer the harness, not the model" law, ported from *static* verifiable tasks
(GSM8K / multi-hop QA / HumanEval) to an *interactive, sequential, world-model*
task. The combined-paper findings were measured on the same SLM band we use here
(qwen2.5 1.5B–7B, llama3.1, gemma3), so its principles are our priors.

**Honest framing.** Claude solves 21/25 full games (`agent_full_game.json`). E118
(pure-LLM ReAct, `qwen3-coder:30b`) scored **0/8 on ar25** even after a memory +
24-round + 32k-ctx upgrade. The bottleneck is the harness/representation, not the
model's memory. E119 tests whether the right harness closes the gap with SLMs.

## 2. The law we design against

> A harness knob helps in proportion to how much it routes the consequential
> decision to an **executor** rather than to **model self-judgment** — *provided
> the model can produce what the executor checks.*

Every component below routes the *decision* to an executor (env replay, a
deterministic grader) and lets the SLM only *propose*. Corollaries we obey:

| combined-paper finding | E119 consequence |
|---|---|
| Sample + vote (+14) | best-of-N per SLM slot, aggregate by agreement |
| Model diversity (+21) | best-of-N draws across **uncorrelated SLMs**, not repeats |
| **Abstain on low agreement** (prec 0.75→0.97) | an SLM hint is **used only when N samples agree ≥ τ**; else abstain → deterministic fallback |
| Domain routing (0.95) | per-slot model routing (coder vs reasoning vs tiny) |
| Adaptive sampling (½ calls) | stop sampling once samples agree |
| **No self-review repair** (0.65→0.61) | NO iterative self-correction loops anywhere |
| **No few-shot on small models** | retrieval/few-shot is an **ablation**, not a default |
| **No model-judge** (worse than none) | every gate is executable; no SLM grades another SLM |
| PAL hurts unless artifact is correct | search-only rung + replay-verify guarantee correctness; SLM codegen lift must be *measured* |
| Decoding cliff | generous `max_tokens`; stock penalties; thinking models 4k+ |
| Verifier must be good (≤10% noise) | replay-verify is exact (0% noise) |
| Lean beats teams | single agent + diversity-voting; no gratuitous multi-agent |

## 3. Representation (the critical fix)

The SLM **never sees a 64×64 grid** (4k digit-tokens, lossy number tokenization,
no reliable 2D indexing). `perceive.py` converts each frame to **compact JSON**:

- object list via `arc3_graph.objects`: `{id, color, bbox, centroid, size}`
- frame→frame **deltas** (which objects appeared/moved/changed)
- status-bar **masking**: cells changing on >0.95 of probe steps are zeroed
  before hashing (the e107/e111 trick; keep τ high to avoid collapsing signal)

## 4. Components (isolated, independently testable units)

### 4.1 `perceive.py` (deterministic — no SLM)
- probe from `reset()`: try each `available_actions` directional move; for click
  games, click the **pixel-inferred candidate set** (small connected components +
  rare-color cells). Record `(frame, action, next_frame, levels)` transitions.
- status-bar masking; object/delta extraction; object-JSON renderer.
- a compressed **facts ledger** (verified bullet-facts the *harness* maintains —
  NOT a raw transcript; combined-paper: long raw context hurts small models).
- Testable on recorded frames with no env/LLM.

### 4.2 `slm.py` — proposer with abstention (the model layer)
Each slot: route to model(s) → best-of-N sample → **execution-grade** each
candidate → **abstain unless ≥ τ agree** → return winner or `ABSTAIN`.

| slot | proposes | executable grader | on abstain/fail |
|---|---|---|---|
| `interactive_cells` | clickable/agent object ids | do those clicks change the board (vs no-op)? | use full pixel-candidate set |
| `rule(action)` | **object-level** transition rule (NOT frame→frame) | reproduces observed object deltas on held-out transitions | search uses env directly |
| `subgoal` | **tool-call schema**, enum predicate `{reach(color), align(a,b), count(color)==k,…}` | predicate satisfiable on observed frames | BFS runs unguided |
| `macro` | short action template when search stalls | replay raises progress toward subgoal | continue plain search |

Rules: best-of-N is across **diverse models**; **adaptive stop** once agree;
**no self-repair**; **no model-judge**; generous `max_tokens`; thinking models 4k+.

### 4.3 `planner.py` (deterministic — the solver)
- candidate actions = directional ∪ pixel click-targets, **ordered** by
  `interactive_cells` + `subgoal` heuristic (hints, never ground truth).
- search: BFS / greedy / iterative-deepening over the **masked state-graph**;
  **env is ground truth** (replay from reset: 0.6 ms reset, 0.04 ms step;
  `levels_completed` rise = level solved). Non-target clicks dedup away.
- optional **in-model** search when a `rule`-set grades high-accuracy: plan in the
  model, then **replay-verify** against the env (cheap correctness gate). This is
  where the SLM accelerates depths blind BFS can't reach.
- per-level budgets (depth, node count, wallclock) — masking + dedup prevent the
  e107 73k-state blowup.
- keep the planner **simple** (combined-paper: graph planners lose to plain ones
  on small models).

### 4.4 `solve.py` — orchestration
- per-level chaining: solve level → **re-probe** (mechanics change per level) →
  continue to `g.levels == g.win`.
- bank **replay-verified** `solved.json` (`{game, actions, levels, win}`), same
  format/pipeline as the Claude thread; build the solver **as an OpenWorld World**
  (masked-frame perceptor → state, `FunctionTransition` over the learned table →
  dynamics, induced `CodeObjective` → reward; `to_spec` → `preview.graph`).
- **JSONL-log every SLM call**: model, slot, samples, agreement, abstain/used,
  grade, tokens, seconds (reproducibility, like E118).

### 4.5 `e119_slm_solver.py` — experiment entry
Runs games, the rungs, the model sweep; `save_results` BEFORE asserts; asserts the
sign/shape of every claim; deterministic seeds.

## 5. Model pool (SLMs are the subject; ≥27B is a labeled ceiling)

**SLM band (the actual subjects, ≤~9B):**
- reasoning/general: `gemma2:9b`, `gemma4`, `llama3.1:8b`, `llama3.2:3b`,
  `phi3.5:3.8b`, `qwen2.5:7b`, `qwen2.5:3b`
- thinking: `qwen3:8b`, `qwen3:4b` (need 4k+ output budget)
- tiny (routing/extraction): `qwen2.5:1.5b`, `qwen2.5:0.5b`
- **coder gap — pull:** `qwen2.5-coder:7b` (and `:3b`); optionally `codegemma:7b`
  (a Gemma-family coder). No small coder is installed today; this is the most
  important pull for an SLM-codegen study.

**Per-slot routing (domain routing):** codegen slots → small coder; reasoning
slots → `gemma2:9b` / `qwen3:8b`(thinking); routing/extraction → `qwen2.5:1.5b`.
best-of-N draws across the diverse set.

**Ceiling reference (labeled NOT-SLM):** `qwen2.5-coder:32b` (dense),
`qwen3-coder:30b` (A3B MoE), `gemma3:27b`, `qwen3.6`. Used only to chart how much
raw size buys on the *same* harness — never reported as an "SLM" result.

## 6. Honest reporting

**Rungs (per game, full-game = `levels==win`):**
1. **search-only, NO SLM** — deterministic planner, all hints disabled. *The
   control that proves whether the SLM adds anything.*
2. **single-SLM-in-loop** — best single SLM.
3. **SLM-set-in-loop** — diverse best-of-N + abstention (the headline system).
4. **ceiling** — 27–30B on the same harness; and Claude 21/25 cited as reference.

**Beyond a single number:**
- **Pareto vs compute:** solve-rate vs **env-step budget** AND vs **LLM-token
  budget** — else "SLM helped" might just be "more search."
- **Abstention curve:** coverage vs precision of each SLM slot (the 0.75→0.97
  mechanism, measured here).
- **Held-out game:** ≥1 game the harness is never tuned against (guards against
  overfitting to the known 25 solutions).
- replay-verify every banked solution; `save_results` before asserts.

## 7. Error handling / budgets / decoding

- every LLM call `try/except` (timeout = abstain that hint; search continues).
- `num_ctx` capped (effective reasoning ctx is small — keep prompts compact);
  **output** `max_tokens` generous; thinking models 4k+; stock repetition penalty.
- per-level depth/node/wallclock caps; masking + dedup bound the state space.
- `arc.make(game)` once per game, then `reset()` + replay (never in a loop).

## 8. Phasing & scope

- **Phase 1 (validate the pipeline):** ~5 games spanning modalities — e.g.
  `tn36`, `ar25` (directional), `vc33`, `lp85` (click), + one deeper. Get rungs 1–3
  working end-to-end on a couple of SLMs. Prove abstention + diversity-voting +
  replay-verify chain works.
- **Phase 2:** expand to all 25 + the model sweep + the ceiling row.
- **Expectation (honest):** wins on search-tractable + retrieval-matched games;
  honest losses on deep, novel-mechanic games (e.g. wa30's 657 actions). Report the
  boundary, don't tune to a number.

## 9. Out of scope (YAGNI)

- multi-agent debate / dialogue (combined-paper: lean beats teams).
- self-review repair loops; model-judge selection; graph planners for the SLM.
- privileged engine access (`_get_valid_clickable_actions`, internal level index) —
  stay pixel-honest.
- new core dependencies (OpenWorld core stays zero-dep; this lives in
  `experiments/`, which may use numpy etc.). QuHarness reuse is optional, not required.

## 10. Open items for review

- Phase-1 game set (the 5) — right spread?
- Which pulls to approve: `qwen2.5-coder:7b` (strongly recommended), `:3b`,
  `codegemma:7b`?
- Token/time budget ceiling for the full sweep (it's many models × 25 games).
