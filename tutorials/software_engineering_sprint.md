# Software Engineering: Sprint Planning with a Quality-Bar Dial

> Script: [`software_engineering_sprint.py`](software_engineering_sprint.py) —
> runs offline; uses two live models when Ollama is up.

An engineering team works a 12-item backlog over a 14-step sprint. Shipping
accrues tech debt; shipping *on top of* debt breeds bugs (`bugs += debt // 4`).
The **quality-bar dial** decides how aggressively the team refactors and fixes
instead of shipping. This tutorial's real subject, though, is trust: how do you
know synthesized dynamics are *right*?

## 1. The generator + critic relay

`compile()` accepts a second model as a semantic critic. The generator writes
the dynamics; the critic reviews the code against your written rules and
returns `PASS` or concrete feedback, which loops back into the next generation
attempt:

```python
generator = OllamaLLM(model="qwen2.5:7b")
critic    = OllamaLLM(model="qwen2.5:3b")

transition = world.compile(
    critic=critic,
    invariants=[
        ("counters never negative",
         lambda s: all(s[k] >= 0 for k in ("backlog", "shipped", "bugs", "debt"))),
    ],
)
```

Three independent gates now stand between the LLM and your simulation:
sandboxed smoke-runs, your invariants, and a second model reading the code.
Dividing the work between specialized local models is cheap — both of these
run comfortably on one laptop.

## 2. Validate against ground truth

Because dynamics are just `(state, action) -> state`, you can probe synthesized
code exactly like any pure function. The script keeps a hand-written
`ground_truth` implementation and diffs the two on a tricky case — shipping
with high debt, where the `debt // 4` ordering matters:

```python
probe = WorldState({"backlog": 5, "shipped": 0, "bugs": 0, "debt": 4})
synthesized = transition.step(probe, Action("ship"))
expected    = WorldState(ground_truth(dict(probe), Action("ship").to_dict()))
assert synthesized == expected, synthesized.diff(expected)
```

In our live run, qwen2.5:7b's code **matched ground truth exactly**. When it
doesn't, `WorldState.diff()` tells you precisely which fields diverged — turn
that into a sharper rule or a new invariant and recompile. This loop (rule →
synthesize → probe → tighten) is the practical workflow for trusting
LLM-written dynamics.

## 3. The dial: ship or pay down?

```python
quality_bar = Dial("quality_bar", value=0.0)

def tech_lead(state, actions):
    debt_limit = round((1.0 - quality_bar.value) * 6)
    bug_limit  = round((1.0 - quality_bar.value) * 4)
    if state["debt"] > debt_limit:  return Action("refactor")
    if state["bugs"] > bug_limit:   return Action("fix")
    if state["backlog"] > 0:        return Action("ship")
    return Action("fix" if state["bugs"] > 0 else "refactor")
```

## 4. A frontier with a free lunch

```
quality_bar |      aggregate |       delivery |        quality
--------------------------------------------------------------
     0.000 |        10.0000 |        10.0000 |        -6.0000
     0.250 |         8.1250 |         9.0000 |        -3.5000
     0.500 |         8.0000 |        10.0000 |        -4.0000
     0.750 |         9.2500 |        10.0000 |        -1.0000
     1.000 |         7.0000 |         7.0000 |         0.0000

Pareto frontier (delivery vs quality):
  lambda=0.75  delivery=10.0  quality=-1.0
  lambda=1.0   delivery=7.0   quality=0.0
```

Only two points survive the Pareto filter. λ=0 (ship at all costs) is
**dominated**: λ=0.75 ships the same 10 features with far less damage, because
periodic refactoring keeps `debt // 4` from ever biting. The model just
reproduced a familiar truth — a moderate quality bar is free; only a maximal
one costs throughput. Sweeps don't just trace trade-offs; they expose policies
that aren't on the frontier at all.

## Try next

- Make bugs expensive: add a rule that each open bug at sprint end consumes a
  backlog slot next sprint, then chain `sim.run(reset=False)` across sprints.
- Swap the `tech_lead` policy for an LLM agent and see where its judgment lands
  relative to the frontier the deterministic sweep mapped.
- Compare engines: run the same sweep with `world.use_llm_dynamics()` (the LLM
  predicts each next state directly) and measure how its stochasticity distorts
  the frontier versus compiled code.
