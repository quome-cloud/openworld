# E58 — Brain simulator: world-of-worlds + tree-of-thoughts ReAct with real memory

**Goal:** A composable *brain* world model — perceive from outside, act on the
outside — split into a **conscious** world (working memory / current thoughts) and
an **unconscious** world (long-term, content-addressable memory). An agent runs a
**ReAct loop with tree-of-thoughts** over a learned model and **real persistent
memory**, and we show, deterministically, that memory + simulated lookahead beat
memoryless and random baselines.

**Decisions (approved):** deterministic core + optional LLM demo; prove **both**
claims — (A) memory wins, (B) how-much-to-think curve.

## Architecture

### The brain (`CompositeWorld`, serializable artifact + demo)
- **`conscious`** (leaf `World`): `{node, plan, step}` — current focus + intended plan.
- **`unconscious`** (leaf `World`): long-term memory — a learned transition `model`
  `(node,tool)->next` and content-addressable `cache` `node->plan-to-goal`.
- **`environment`** (leaf `World`): `{node, goal, solved}`; its `CodeTransition`
  *is* the world the brain acts on — `env_step(node, tool) -> next node` over a
  deterministic graph.
- **Bridges:** `retrieve` (unconscious→conscious: surface a cached plan/known
  edges for the current node), `consolidate` (conscious→unconscious: encode the
  observed transition + a working plan), `act` (conscious→environment: apply the
  chosen tool). **Perception:** external situation → conscious (`CodePerceptor`
  deterministically; optional `TextPerceptor` for the LLM demo). **Emit:** the
  environment outcome is the brain's output.
- Serialized via `to_spec`, rendered to a card, **asserted to round-trip**; the
  optional `openworld serve` → interactive `/view` is the live demo.

### The agent (tree-of-thoughts ReAct)
Navigate a deterministic environment graph from start to goal. Each step:
**perceive** the current node → **retrieve** from the unconscious (a cached plan,
or the known edges) → **think**: expand a depth-`D` tree over the *learned* model
to find a path to the goal → **act**: take the first tool of the best plan on the
real environment → **observe** the next node → **consolidate** the transition (and
a found plan) into the unconscious. Unknown edges force real exploration; once
learned, they are recalled — *real* memory, persisting across episodes.

## What it proves (deterministic, seeded, self-checking)

Deterministic graph env (K≈12 nodes, T≈4 tools/node, one goal, shortest path L).

- **Panel A — memory wins** (E episodes on the *same* env; steps-to-goal):
  - `brain` (persistent unconscious + tree planning) — learning curve falls to the
    optimal `L`.
  - `no-memory` (unconscious wiped each episode) — flat, re-learns every time.
  - `random` (random tool each step) — high.
  - **Asserts:** `brain` final-episode steps `== L`; mean steps `brain < no-memory
    < random`. (The unconscious / long-term memory carries the cross-episode gain.)
- **Panel B — how much to think** (model pre-known; steps-to-goal vs planning depth
  `D=0…4`): `D=0` (no lookahead) is long; rising `D` reaches optimal once `D≥L`,
  then **plateaus** (no benefit to over-thinking past the needed horizon).
  - **Asserts:** `steps(D≥L) == L`; `steps(D=0) > steps(D=L)`; `steps(D=Dmax) ==
    steps(D=L)` (plateau).

Honest framing: a controlled synthetic cognitive task — *not* a claim about real
brains. It demonstrates the framework can **compose a brain-like architecture**
(conscious/unconscious worlds, perceive/act boundary, agent tree-search with
memory) and that memory + bounded simulated lookahead **measurably** help.

## Files & integration
- `experiments/e58_brain.py`: `make_env` (seeded graph), `brain_world()`
  (`CompositeWorld` builder + `CodePerceptor`), the ReAct/tree agent + baselines,
  Panels A/B, brain spec round-trip + card, `save_results` **before** asserts.
- `experiments/results/e58_brain.json`.
- `tests/test_e58_brain.py`: memory-retrieval + agent-convergence determinism (a
  small focused test), and brain-spec round-trip.
- Paper: `fig_brain` (schematic + 2 panels) + `table_brain`, macros
  (`\BrainNodes`, `\BrainOptimal`, `\BrainBrainSteps`, `\BrainNoMemSteps`,
  `\BrainPlateauDepth`), `EXPERIMENTS` += `e58_brain`, `main()` calls,
  `\NumExperiments` 56→57; `paper/main.tex` `\subsubsection{...E58}`
  (`\label{sec:brain}`) with the world-of-worlds schematic + both result panels.
- One branch off `main` (`e58-brain`), one PR.

## Out of scope (YAGNI)
- Real neuroscience fidelity; learned/neural memory (memory is content-addressable
  symbolic). The LLM demo is optional and not in the deterministic asserts.
