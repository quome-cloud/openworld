# Healthcare: ICU Triage with a Stewardship Dial

*New to worlds and oracles? Read [OpenWorld for scikit-learn users](from_scikit_learn.md) first.*

> Script: [`healthcare_triage.py`](healthcare_triage.py) — runs offline; does live
> dynamics synthesis when Ollama is up.

A charge nurse works a triage queue: 4 critical patients, 8 moderate, and a
clock that punishes waiting — every second tick, one untreated critical patient
deteriorates. Treating patients improves outcomes but costs money. How hard
should the unit chase outcomes before spend matters? That's not a fact about
the world; it's a **value choice** — so we model it as a dial.

## 1. Declare the world

State is plain JSON-serializable data; rules are plain language. Be precise in
the rules — they are the spec the LLM implements:

```python
world = World(
    name="icu-triage",
    description="An ICU triage queue...",
    initial_state={
        "tick": 0,
        "critical_waiting": 4, "moderate_waiting": 8,
        "treated": 0, "deteriorated": 0,
        "outcomes": 0, "spend": 0,
    },
    actions=["treat_critical", "treat_moderate", "wait"],
    rules=[
        "'treat_critical' treats one waiting critical patient: treated +1, outcomes +3, spend +3.",
        "'treat_moderate' treats one waiting moderate patient: treated +1, outcomes +1, spend +1.",
        "After EVERY action (including 'wait' and 'noop'), tick increases by 1.",
        "Whenever the new tick is even and critical patients still wait, one of them "
        "deteriorates: critical_waiting -1, deteriorated +1, outcomes -2.",
    ],
    llm=OllamaLLM(model="qwen2.5:7b"),
)
```

## 2. Synthesize the dynamics — with safety invariants

`compile()` asks the LLM to write `transition(state, action)`, then verifies
every candidate: syntax, sandboxed smoke-runs on each action, and **your
invariants**. In a clinical setting the invariants are where you encode the
non-negotiables:

```python
world.compile(
    invariants=[
        ("queues never negative", lambda s: s["critical_waiting"] >= 0 and s["moderate_waiting"] >= 0),
        ("spend never negative", lambda s: s["spend"] >= 0),
    ],
)
```

Failed checks are fed back to the generator and it retries. A 7B model usually
passes within an attempt or two.

> **Inspect what you accepted.** Verification proves the code runs and respects
> your invariants — it does not prove the model read your rules perfectly. Pass
> `save_to="triage_dynamics.py"`, read the artifact, and tighten rules or add
> invariants if behavior surprises you. The dynamics being *readable code you
> can diff against your intent* is the whole point of this paradigm.

## 3. The agent and the dial

The nurse always treats critical patients first. The `stewardship` dial throttles
discretionary spend on moderate cases and simultaneously weights a thrift
objective:

```python
stewardship = Dial("stewardship", value=0.0)

def charge_nurse(state, actions):
    if state["critical_waiting"] > 0:
        return Action("treat_critical")
    budget_cap = round((1.0 - stewardship.value) * 14)
    if state["moderate_waiting"] > 0 and state["spend"] + 1 <= budget_cap:
        return Action("treat_moderate")
    return Action("wait")

objectives=[
    Objective("outcomes", fn=lambda s, a, ns: ns["outcomes"] - s["outcomes"], weight=1.0),
    Objective("thrift",   fn=lambda s, a, ns: -(ns["spend"] - s["spend"]),    weight=stewardship),
]
```

## 4. Sweep it

```python
result = sweep(sim, dial="stewardship", values=[0.0, 0.25, 0.5, 0.75, 1.0], steps=12)
print(result.table())
```

A representative run (live qwen2.5:7b dynamics):

```
stewardship |      aggregate |       outcomes |         thrift
--------------------------------------------------------------
     0.000 |        12.0000 |        12.0000 |       -14.0000
     0.250 |         5.5000 |         8.0000 |       -10.0000
     0.500 |         2.5000 |         7.0000 |        -9.0000
     ...
```

λ=0 buys the best outcomes at the highest spend; raising λ sheds discretionary
moderate treatments first — exactly the behavior you'd want to show a budget
committee, produced by turning a dial rather than retraining anything.

## Try next

- Add an arrivals rule ("every 3rd tick a new moderate patient arrives") and
  watch the queue dynamics change.
- Replace the policy nurse with an LLM agent
  (`Agent(name="nurse", goal="triage by acuity", llm=...)`) and compare its
  decisions against the deterministic policy.
- Add a third objective (e.g. `equity`: penalize moderate-queue starvation) and
  sweep a second dial.
