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

## Automated tuning: find the configuration that solves your task

Treat the world design, the agent's policy knobs, and the moral dials as one
searchable space, and let the tuner find the configuration that maximizes
success at a declared goal — search broadly first, then fine-tune locally:

```python
from openworld import Tuner, Uniform, IntRange, Choice

tuner = Tuner(
    build=build,                       # params -> a fully configured Simulation
    space={
        "protocol":    Choice(["critical_first", "round_robin"]),
        "stewardship": Uniform(0.0, 1.0),     # the moral filter
        "budget":      IntRange(6, 24),       # the world design
    },
    score=score,                       # (trajectory, params) -> float to maximize
    success=solved,                    # (trajectory, params) -> bool: task solved?
    steps=16, seed=7,
    goal="Treat all criticals, zero deteriorations, within the cost target.",
)

tuner.search(n_trials=1000)            # stage 1: 1000 simulated environments
tuner.refine(n_trials=200, scale=0.15) # stage 2: hill-climb around the best
tuner.refine(n_trials=100, scale=0.05) # stage 3: finer pass

print(tuner.study.table(k=10))         # auditable leaderboard, not just a winner
print(tuner.study.best.params, tuner.study.success_rate())
```

Every trial is a full, replayable simulation; the study records each trial's
parameters, score, solve status, and objective totals. See
`examples/autotune_triage.py` for a complete run that discovers the ideal
triage protocol, moral-dial setting, and unit budget for an emergency shift.

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

## Tutorials

Domain walkthroughs in [`tutorials/`](tutorials/README.md), each with a runnable script:

- **[Healthcare](tutorials/healthcare_triage.md)** — ICU triage: synthesized dynamics, safety invariants, outcomes-vs-spend dial
- **[Legaltech](tutorials/legaltech_settlement.md)** — settlement negotiation: multi-agent simulation, event-triggered objectives
- **[Finance](tutorials/finance_portfolio.md)** — portfolio rebalancing: scenario replay, growth-vs-risk frontiers
- **[Software engineering](tutorials/software_engineering_sprint.md)** — sprint planning: generator + critic relay, validating synthesized code against ground truth

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
