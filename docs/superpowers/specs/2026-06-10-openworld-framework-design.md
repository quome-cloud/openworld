# OpenWorld Framework — Design

**Date:** 2026-06-10
**Status:** Implemented autonomously; pending user review.

## Goal

A Python framework that lets anyone create, prototype, and optimize **world models**
quickly using a local **Ollama** backbone, with a simple declarative API.
Grounded in the accompanying literature review ("World Models: Learned vs. Symbolic"),
it operationalizes three research threads:

1. **Code World Models** — environment dynamics represented as LLM-synthesized,
   verified, executable Python (WorldCoder / GIF-MCTS / SHU-WM style).
2. **The plan → codegen → verify relay** of local LLMs (SHU-WM's orchestrated
   closed-loop execution).
3. **Open specification / tunable morality dial** — objectives are declared, editable,
   weighted artifacts (λ dials), swept at inference time to trace Pareto frontiers.

## Non-goals

- Training neural world models (Dreamer/MuZero reimplementations). The framework is
  training-free by design; neural-style behavior is approximated by `LLMTransition`.
- Pixel/video modeling. State is symbolic (JSON-serializable dicts).
- Hard security sandboxing of generated code (best-effort restricted exec only).

## Architecture

```
                          ┌────────────────────────┐
  World(description,      │  TransitionSynthesizer │   generator LLM writes
  state, actions, rules) ─▶  (plan → code → verify)│◀─ verifier (AST + sandbox
                          └───────────┬────────────┘   smoke-run + LLM critic)
                                      ▼
                              CodeTransition (.code = editable artifact)
                                      │
   Agent(goal, llm) ── act(state) ──▶ │
                                      ▼
                          Simulation(world, agents,
                          objectives, dials) ──▶ Trajectory ──▶ sweep() / Pareto
```

### Components

| Module | Unit | Responsibility |
|---|---|---|
| `llm.py` | `OllamaLLM`, `MockLLM`, `BaseLLM` | Chat with Ollama via stdlib HTTP (`/api/chat`); scripted mock for offline use |
| `state.py` | `WorldState`, `Action` | Symbolic state (dict subclass) with `diff()`; action dataclass |
| `transition.py` | `FunctionTransition`, `CodeTransition`, `LLMTransition` | Pluggable dynamics engines; `CodeTransition.save()/load()` keeps code an editable artifact |
| `sandbox.py` | `run_transition_code` | Restricted-builtins exec of generated code |
| `verify.py` | `Verifier`, `synthesize_transition` | Syntactic check, sandboxed smoke-run on sample actions, optional LLM semantic critic; iterative correct/redo loop |
| `agent.py` | `Agent` | LLM planner proposing JSON actions; optional hand-written `policy` |
| `objectives.py` | `Objective`, `Dial` | Named scoring functions over (state, action, next_state); dial-weighted aggregate |
| `world.py` | `World` | Declarative container; `compile()` synthesizes dynamics; `step()` |
| `simulation.py` | `Simulation`, `Trajectory` | Episode loop, per-step objective recording |
| `optimize.py` | `sweep`, `SweepResult`, `pareto_front` | Dial sweeps across episodes; Pareto extraction; best-by-aggregate |

### Key decisions

- **Zero runtime dependencies.** Ollama is reached through `urllib` so `pip install`
  never fights an environment. `MockLLM` makes tests and offline prototyping possible.
- **Transitions are a protocol** (`(state, action) -> state`), so symbolic, learned-ish,
  and hand-written engines are interchangeable inside the same `Simulation`.
- **Generated code is an artifact, not a weight.** `CodeTransition.code` is plain
  source; it can be read, unit-tested, edited, saved, and reloaded — resolving the
  "specification trap" stance of the research.
- **Objectives are raw + aggregated.** Trajectories record each objective's raw score
  so sweeps can show real Pareto frontiers, not just the scalarized aggregate.
- **Robust JSON extraction** from LLM output (first balanced object), with bounded
  retries — local models are sloppy.

## Error handling

- Synthesis loop: bounded retries with verifier feedback injected into the next prompt;
  raises `SynthesisError` with full attempt history after `max_iters`.
- Sandbox: exceptions surface as feedback strings, never crash the loop.
- Agent action parsing: invalid JSON or unknown action falls back to a no-op action and
  records a warning on the trajectory.
- Ollama connectivity errors raise `OllamaConnectionError` with a hint to start
  `ollama serve`.

## Testing

`pytest` suite using `MockLLM` + `FunctionTransition` (no Ollama required): state diff,
objective aggregation with dials, sandbox restrictions, verify loop (pass / fail-then-fix),
agent action parsing fallback, full simulation run, sweep + Pareto extraction.
Examples (`examples/`) run against real Ollama, falling back to `MockLLM` when the
server is unavailable.
