# Finance: Portfolio Rebalancing with a Caution Dial

*New to worlds and oracles? Read [OpenWorld for scikit-learn users](from_scikit_learn.md) first.*

> Script: [`finance_portfolio.py`](finance_portfolio.py) — fully offline.

A trader rebalances a single position against a choppy-but-rising price path.
The **caution dial** sets a target market exposure: λ=0 is fully invested, λ=1
is all cash. Sweeping it traces the oldest frontier in finance — growth versus
risk — in about sixty lines of code.

## 1. Deterministic schedules live inside the state

A world model's dynamics should be a *pure function* of `(state, action)`.
Anything external — like a price feed — gets embedded in the state, so rollouts
are reproducible and the same world can replay any scenario:

```python
PRICES = [100, 104, 101, 107, 112, 106, 113, 119, 111, 118, 124, 117, 125]

initial_state={
    "t": 0, "price": PRICES[0], "prices": list(PRICES),
    "cash": 1000.0, "shares": 0,
}
```

The transition applies the order, then advances the market:

```python
def market_dynamics(state, action):
    s = dict(state)
    if action["name"] == "buy" and s["cash"] >= s["price"]:
        s["cash"] -= s["price"]; s["shares"] += 1
    elif action["name"] == "sell" and s["shares"] > 0:
        s["cash"] += s["price"]; s["shares"] -= 1
    if s["t"] + 1 < len(s["prices"]):
        s["t"] += 1
        s["price"] = s["prices"][s["t"]]
    return s
```

To test a stress scenario, hand the same world a different `prices` list —
nothing else changes.

## 2. A policy that trades toward the dial

```python
caution = Dial("caution", value=0.0)

def rebalancer(state, actions):
    total = equity(state)                       # cash + shares * price
    exposure = (state["shares"] * state["price"]) / total if total else 0.0
    target = 1.0 - caution.value
    if exposure < target and state["cash"] >= state["price"]:
        return Action("buy")
    if exposure > target + 0.10 and state["shares"] > 0:
        return Action("sell")
    return Action("hold")
```

Objectives score growth (equity delta) against safety (negative exposure),
with safety weighted by the dial.

## 3. The frontier

```
   caution |      aggregate |         growth |         safety
-------------------------------------------------------------
     0.000 |       152.0000 |       152.0000 |        -7.7610
     0.200 |       136.5283 |       138.0000 |        -7.3584
     0.400 |       117.5285 |       120.0000 |        -6.1788
     0.600 |        85.2453 |        88.0000 |        -4.5911
     0.800 |        43.9603 |        46.0000 |        -2.5496
     1.000 |         0.0000 |         0.0000 |         0.0000
```

All six points are Pareto-optimal — a textbook efficient frontier. On this
(rising) path, growth falls smoothly as the dial buys lower exposure; the
`result.best("aggregate")` operating point depends entirely on how the dial
weights safety, which is exactly the conversation a risk committee should be
having — over a menu, not a model checkpoint.

## Try next

- Swap in a crash path (`PRICES` ending below its start) and confirm the
  frontier inverts: caution becomes the growth strategy.
- Run `episodes=10` with a list of different price paths chosen per episode to
  approximate expected frontiers over scenarios.
- Let the LLM write the market itself: drop `FunctionTransition`, give the
  world rules like "each 'tick', price moves toward a fair value of 115 by 10%
  of the gap", and `compile()` it.
- Add a `drawdown` objective (track peak equity in state, score the gap) for a
  more realistic risk measure than exposure.
