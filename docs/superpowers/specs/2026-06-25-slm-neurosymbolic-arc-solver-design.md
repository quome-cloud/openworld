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
- **relational, not absolute** (Qwen review): positions expressed relative to the
  agent ("red 2-left of agent"); the arbitrary 0–15 color indices mapped to stable
  symbolic tokens (they're meaningless integers to the model).
- transitions fed as explicit **before→after contrastive diffs**, never two raw
  frames — the model learns dynamics from the diff, not by re-deriving it.
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

**Where the SLM earns its keep (Qwen review — concentrate, don't spread).** The
deterministic probe already *observes* which clicks change the board and which
object moves under each action, so `interactive_cells` and `rule` are **env-oracled
— the SLM adds near-zero value there and is demoted to an opt-in ablation only.**
The SLM's real leverage is the one thing the env does *not* hand you: the **goal /
subgoal prior** on goal-as-*procedure* games (E102/103/104 died precisely because
the win is a procedure, not a visible state-score) where search is otherwise
intractable. Pour the budget into `subgoal`/`macro`.

Each slot: route → best-of-N → **execution-grade** → **abstain unless candidates
agree ≥ τ** → return winner or `ABSTAIN`.

| slot | owner | proposes | executable grader | on abstain/fail |
|---|---|---|---|---|
| `interactive_cells` | **deterministic** (SLM = ablation) | clickable/agent ids | clicks change board vs no-op? | full pixel-candidate set |
| `rule(action)` | **deterministic** (SLM = ablation) | object-level transition rule | reproduces observed deltas on held-out? | search uses env directly |
| **`subgoal`** | **SLM (primary)** | predicate `{reach(color), align(a,b), count(color)==k,…}` | satisfiable on observed frames | BFS runs unguided |
| **`macro`** | **SLM (primary)** | short action template when search stalls | replay raises progress toward subgoal | continue plain search |

**Abstention is behavioral, not textual (Qwen review #3).** Two predicates/rules
can differ in text yet be identical in effect (and vice-versa). Cluster the N
candidates by **observed behavior** — same cells selected / same frames accepted on
the probe set — and vote by **cluster mass**; abstain if no cluster clears τ.
**Calibrate τ per slot** against held-out probe transitions (the τ that buys
precision ≈0.97 differs by slot and model).

**Native scaffolds (Qwen review #4):** structured outputs use **grammar-constrained
decoding** (GBNF) on the predicate schema — small Qwen tool-calling is too fragile
to trust raw JSON, and a grammar lets us *lower* N. Any retained code slot uses
**FIM / skeleton-completion** (a function with a hole), Qwen-Coder's pretrained
mode, not free generation.

Rules: best-of-N across **diverse models**; **adaptive stop** once a cluster clears
τ; **no self-repair**; **no model-judge**.

**Per-slot thinking mode (Qwen review #2):** thinking is a *slot* property, not a
model property. **On** for `subgoal`/`macro` (Qwen3-4B-thinking can beat a 9B
non-thinking on reasoning), **off** for any structured/grammar output (thinking
derails tool formatting and triples latency). Toggle via `enable_thinking`/`/no_think`.

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

**Pinned decoding (Qwen review #2 — Ollama stock ≠ Qwen's published defaults; the
gap is a silent accuracy cliff):**
- Qwen2.5 / Qwen2.5-Coder: `temp 0.7, top_p 0.8, top_k 20, repeat_penalty 1.05`.
- Qwen3 **thinking**: `temp 0.6, top_p 0.95, top_k 20` — **never greedy** (temp 0
  loops in thinking mode).
- Qwen3 **non-thinking**: `temp 0.7, top_p 0.8, top_k 20`.
- All: generous **output** `max_tokens` (thinking 4k+ or it returns nothing
  mid-thought); **never** `repeat_penalty ≥ 1.3` (breaks generation — matches the
  combined-paper decoding cliff). Pin per family in a config table, not by Ollama default.

## 6. Honest reporting

**Rungs (per game, full-game = `levels==win`):**
1. **search-only, NO SLM** — deterministic planner, all hints disabled. *The
   control that proves whether the SLM adds anything.*
2. **single-SLM-in-loop** — best single SLM.
3. **SLM-set-in-loop** — diverse best-of-N + abstention (the headline system).
4. **ceiling** — 27–30B on the same harness; and Claude 21/25 cited as reference.

**Headline = a capability-substitution curve (Qwen review #6).** Hold the harness
fixed and walk **one family's parameter ladder** — `qwen2.5: 0.5b → 1.5b → 3b → 7b`
(+ `coder-7b`) → ceiling — plotting solve-rate. This isolates *how much harness
structure substitutes for parameters*, the actually-interesting result. Cross-family
models belong in the **diversity-voting pool, not on the x-axis** (confounded by
tokenizer/training). Add one sentence ruling out ARC-AGI-3 pretraining contamination.

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

**One-box throughput (Qwen review #5 — or the sweep takes days for the wrong
reasons):** Ollama **reloads weights on every model switch** (tens of seconds), so
run **model-outer, games-inner** (all games for model A, then B) — never interleave
models. Pin **quantization** (Q4_K_M default; Q8 only for the ceiling row) and a
**concurrency cap** sized to VRAM so best-of-N doesn't thrash. Grammar-constrained
decoding lets us lower N, cutting total calls.

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
