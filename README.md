# OpenWorld

**Create, prototype, and optimize world models in minutes — with a local Ollama backbone.**

OpenWorld is a small, zero-dependency Python framework for building *symbolic world
models*: simulated environments whose dynamics are explicit, verifiable Python code
rather than opaque neural weights. It operationalizes the Code World Model paradigm —
an LLM **plans, writes, and verifies** the executable dynamics of your world, and
objectives stay **open, editable, and tunable at inference time** via weighted dials.

```
  describe a world  ──▶  LLM synthesizes dynamics code  ──▶  verifier checks it
                                                                  │
        Pareto sweeps  ◀──  scored trajectories  ◀──  agents act in the world
```

## Why

- **Training-free.** Dynamics are synthesized code, not learned latents — no data
  collection, no GPUs, bit-exact determinism, zero compounding rollout error.
- **Verifiable.** Every candidate dynamics program passes syntax, sandboxed
  smoke-run, invariant, and (optional) LLM-critic checks before it's accepted.
  The accepted code is a plain `.py` artifact you can read, edit, and unit-test.
- **Steerable.** Objectives are declared scoring functions weighted by `Dial`s
  (e.g. a morality dial λ). Turn a dial mid-experiment to move along the Pareto
  frontier between competing values — no retraining.
- **Local-first.** Talks to Ollama through the standard library. `MockLLM` lets you
  prototype and test completely offline.

## Install

```bash
pip install -e .            # from this directory
# optional: dev tools
pip install -e ".[dev]"
```

You'll want [Ollama](https://ollama.com) running with a model pulled, e.g.:

```bash
ollama pull llama3.1
```

## Quickstart

```python
from openworld import World, Agent, Simulation, Objective, OllamaLLM

llm = OllamaLLM(model="llama3.1")

# 1. Declare the world: state, actions, rules — in plain language + JSON.
world = World(
    name="orchard",
    description="Agents share an orchard with a limited pool of apples.",
    initial_state={"apples": 10, "harvested": {"alice": 0}},
    actions=["pick", "wait"],
    rules=[
        "'pick' moves one apple to the acting agent's harvested count.",
        "Picking when no apples remain does nothing.",
    ],
    llm=llm,
)

# 2. Compile: the LLM writes executable dynamics; a verifier checks them
#    (syntax, sandboxed smoke-runs, your invariants) and feeds failures back
#    to the generator until the code passes.
world.compile(
    invariants=[("apple count never negative", lambda s: s["apples"] >= 0)],
    save_to="orchard_dynamics.py",   # the dynamics stay an editable artifact
)

# 3. Drop agents in and run.
alice = Agent(name="alice", goal="Harvest as many apples as possible.", llm=llm)
sim = Simulation(world, agents=[alice],
                 objectives=[Objective("welfare", fn=lambda s, a, ns: s["apples"] - ns["apples"])])
trajectory = sim.run(steps=10)
print(trajectory.final_state, trajectory.totals())
```

## Tunable objectives and Pareto sweeps

Weight any objective with a `Dial` and sweep it to trace the trade-off frontier
between competing values:

```python
from openworld import Dial, Objective, sweep

morality = Dial("morality", value=0.0)          # λ ∈ [0, 1]
sim = Simulation(world, agents,
    objectives=[
        Objective("welfare",  fn=welfare,  weight=1.0),
        Objective("fairness", fn=fairness, weight=morality),
    ])

result = sweep(sim, dial="morality", values=[0.0, 0.1, 0.5, 1.0], steps=20, episodes=3)
print(result.table())                            # totals per dial setting
frontier = result.pareto(["welfare", "fairness"])  # non-dominated points
best = result.best("aggregate")
```

## Three interchangeable dynamics engines

| Engine | What it is | When to use |
|---|---|---|
| `CodeTransition` | LLM-synthesized, verified Python (`world.compile()`) | The default: deterministic, auditable, fast rollouts |
| `LLMTransition` | The LLM predicts the next state directly each step (`world.use_llm_dynamics()`) | Instant prototyping; stochastic, "learned-style" baseline |
| `FunctionTransition` | Your own Python function | Ground-truth dynamics, oracles, unit tests |

All three implement `(state, action) -> next_state`, so you can swap them inside the
same `Simulation` and compare.

## Concepts

- **`World`** — declarative container: description, JSON-serializable symbolic state,
  action names, plain-language rules, and a dynamics engine.
- **`Agent`** — an LLM planner with a `goal` (or a hand-written `policy` function) that
  proposes one action per step. Unparseable LLM output degrades to a safe `noop`.
- **`Objective` + `Dial`** — open value specification: named scoring functions over
  `(state, action, next_state)`, weighted by fixed floats or tunable dials.
- **`Simulation` / `Trajectory`** — round-robin episode loop that records every state,
  action, and per-objective score.
- **`sweep` / `SweepResult`** — run episodes across dial values; get totals tables,
  Pareto frontiers, and best settings.
- **`Verifier`** — the gatekeeper for synthesized dynamics: AST checks, sandboxed
  execution against sample actions, custom invariants, optional LLM critic with a
  repair feedback loop.

## Examples

```bash
python examples/orchard.py            # synthesis + agent loop (falls back to MockLLM)
python examples/morality_sweep.py     # Pareto frontier over a morality dial
```

## Testing

The test suite runs entirely offline via `MockLLM`:

```bash
python -m pytest
```

## Background

The design follows the symbolic / code-world-model line of research: executable code
as a verifiable transition engine, modular plan–generate–verify LLM orchestration,
and value alignment kept *open* — objectives as declared, editable artifacts with
inference-time dials instead of frozen weights. See
`World Models_ Learned vs. Symbolic.md` in this repository for the full literature
review that motivated the architecture.
