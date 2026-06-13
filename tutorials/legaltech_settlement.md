# Legaltech: Settlement Negotiation with a Cooperativeness Dial

*New to worlds and oracles? Read [OpenWorld for scikit-learn users](from_scikit_learn.md) first.*

> Script: [`legaltech_settlement.py`](legaltech_settlement.py) — fully offline.

A plaintiff demands $90k; the defense opens at $10k. Every turn of posturing
bills $2k in fees. The defense follows a fixed playbook (raise the offer $4k a
turn, take any deal within $5k). The plaintiff's **cooperativeness dial** is the
strategic question every client actually asks: *hold out for more, or settle
early and cheap?*

## 1. World dynamics as a hand-written function

Negotiation rules are crisp enough to write by hand, so this tutorial uses
`FunctionTransition` — the same interface synthesized code implements:

```python
def negotiation_dynamics(state, action):
    s = dict(state)
    if s["settled"]:
        return s                          # a settled matter is inert
    name, agent = action["name"], action.get("agent")
    if name == "concede":
        amount = int(action["params"].get("amount", 5))
        if agent == "plaintiff_counsel":
            s["demand"] = max(s["offer"], s["demand"] - amount)
        elif agent == "defense_counsel":
            s["offer"] = min(s["demand"], s["offer"] + amount)
    elif name == "accept":
        s["settled"] = True
        s["amount"] = s["offer"] if agent == "plaintiff_counsel" else s["demand"]
    if not s["settled"]:
        s["fees"] += 2
        s["round"] += 1
    return s
```

Note the two multi-agent idioms: the transition reads `action["agent"]` to
apply side-specific effects, and accepting takes *the other side's* number.

## 2. Asymmetric agents sharing a world

Agents act round-robin each simulation step. The defense is fixed; the
plaintiff's policy reads the dial:

```python
cooperativeness = Dial("cooperativeness", value=0.0)

def plaintiff(state, actions):
    gap = state["demand"] - state["offer"]
    if gap <= round(cooperativeness.value * 30):   # acceptable gap in $k
        return Action("accept")
    return Action("concede", params={"amount": 1 + round(cooperativeness.value * 9)})

agents=[
    Agent(name="plaintiff_counsel", policy=plaintiff),
    Agent(name="defense_counsel",   policy=defense),
]
```

## 3. Event-triggered objectives

Most objectives score every step. `recovery` should fire **once** — on the step
where the matter settles. The `(state, action, next_state)` signature makes
edge-detection trivial:

```python
Objective(
    "recovery",
    fn=lambda s, a, ns: float(ns["amount"]) if ns["settled"] and not s["settled"] else 0.0,
),
Objective(
    "cost_control",
    fn=lambda s, a, ns: -float(ns["fees"] - s["fees"]),
    weight=cooperativeness,
),
```

## 4. The frontier

```
  lambda=0.0   settled at $74k in round 31        total fees $62k
  lambda=0.25  settled at $54k in round 22        total fees $44k
  lambda=0.5   settled at $42k in round 16        total fees $32k
  lambda=0.75  settled at $30k in round 10        total fees $20k
  lambda=1.0   settled at $26k in round 8         total fees $16k
```

Every point is Pareto-optimal: more recovery always costs more fees. The dial
turns "how aggressive should we be?" from an argument into a menu — net of
fees, λ=0 nets $74k − $31k (plaintiff's half of $62k) ≈ $43k, while λ=1.0 nets
$26k − $8k = $18k. Holding out pays here because the defense playbook keeps
conceding; change that assumption and the frontier moves.

## Try next

- Give the defense its own dial and sweep both (nested `sweep` calls) to build
  a strategy payoff matrix.
- Replace the plaintiff policy with an LLM negotiator
  (`Agent(..., persona="aggressive litigator", llm=OllamaLLM(...))`) and watch
  `trajectory.warnings` for turns where it produced an unparseable move.
- Add a `trial` event: if `round` exceeds 30 without settlement, the matter
  goes to verdict with a coin-flip outcome encoded in the dynamics.
