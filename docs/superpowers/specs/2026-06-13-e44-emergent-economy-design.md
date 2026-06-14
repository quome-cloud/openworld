# E44 - Emergent economy capstone

**Date:** 2026-06-13
**Status:** approved

## Goal

Show the headline capability of the framework on a "truly open world": assemble
a multi-agent economy entirely out of small **verified** transitions composed
via `CompositeWorld` / `Route` / `Aggregator`, and demonstrate that
(1) macro-economic phenomena **emerge** from the micro rules without being
hard-coded, and (2) because every rule is an explicit, separable verified
component, we can do **causal attribution** on those macro phenomena by toggling
individual rules on and off - something a black-box simulator cannot do cleanly.

This is the reframed "RuneScape"/open-world experiment: agents gather, craft, and
trade to maximize gold; from those selfish micro incentives, price formation,
inflation, and wealth inequality emerge as measurable macro quantities.

## Why this shows the power of the framework

A monolithic economy simulator can also produce inflation and inequality. What
the framework adds is **separable, verified, toggleable rules**: the faucet
(gathering yield), the sink (crafting cost / market tax), and the redistribution
rule are each their own verified transition. We measure each macro phenomenon
with the rule ON and OFF and attribute the effect causally. The economy is a
`CompositeWorld` of per-agent children plus a shared market child, joined by a
trade `Route`; macro quantities are `Aggregator`s (money supply, mean price,
Gini). Emergence + clean causal counterfactuals is the contribution.

## World structure

- **Agents** (N, default 6): each a child world with `{gold, wood, plank}`.
  Per-tick an agent gathers (wood += yield, the **faucet**), optionally crafts
  (wood -> plank at a wood cost, the **sink** consumes nothing external but
  converts), and trades planks on the market for gold.
- **Market** child: holds a posted `price` and inventory. Price adjusts toward
  supply/demand balance each tick (a verified price-update transition): excess
  sell pressure lowers price, excess buy pressure raises it. **Price is not
  scripted** - it is the fixed point of agent behavior.
- **Trade Route** (`Route`/bridge): an agent crossing to the market executes a
  buy/sell at the posted price; an optional **market tax** (toll on each trade,
  the sink) removes gold from circulation. `on_cross` applies the tax.
- **Aggregators**: `money_supply` (sum of agent gold + market till),
  `mean_price`, `gini` (wealth inequality across agents).

All micro transitions are `FunctionTransition`/`Transition` with invariants
(no negative balances; conservation of gold except at explicit faucet/sink).
Determinism: a seeded per-agent action schedule (no `random` at run time beyond
a fixed seed) so the experiment is fully reproducible offline.

## Claims and the causal toggles that test them

1. **Price formation (emergence).** Starting from an arbitrary posted price,
   the market price converges to a level set by the gather/craft supply and the
   agents' gold-funded demand. Verified by: price trajectory stabilizes, and the
   converged level shifts predictably when the gather yield (supply) changes.

2. **Inflation is faucet-minus-sink (causal).** Run with the tax sink OFF
   (faucet > sink: gold only enters) vs ON (sink drains gold). Money supply rises
   unboundedly and mean price inflates with the sink off; turning the verified
   tax rule on bends the price/money curve down. Attribution = the difference
   between the two runs, isolated to one toggled rule.

3. **Inequality emerges and redistribution controls it (causal).** Under
   selfish gold-max, the Gini coefficient rises over time (compounding
   advantage). Toggle a verified **redistribution** rule (tax proceeds paid back
   as an equal dividend) and Gini falls. Attribution = Gini(redistribution off)
   vs Gini(on).

4. **Selfish vs cooperative dial (ties to E08 morality Pareto).** A policy
   parameter: selfish agents maximize individual gold; cooperative agents
   maximize total welfare (sum of gold). Measure the trade-off: cooperative
   raises total welfare and lowers Gini, while the single greediest selfish
   agent ends richer than any cooperative agent. A Pareto-style tension, now
   emerging from a composed verified economy rather than a scalar reward.

## Metrics

- Mean price trajectory and converged price (per supply level).
- Money supply trajectory; price slope with sink ON vs OFF (inflation control).
- Gini over time; final Gini with redistribution ON vs OFF.
- Total welfare and top-agent gold: selfish vs cooperative.

## Self-checks (asserts in the script)

- Conservation: total gold changes only by faucet inflow minus sink outflow
  (accounted exactly each tick) - the composed economy never silently
  creates/destroys gold.
- Emergence: converged price is higher when gather yield (supply) is lower.
- Inflation: mean-price slope(sink OFF) > slope(sink ON).
- Redistribution: final Gini(redistribution ON) < Gini(OFF).
- Dial: total welfare(cooperative) > total welfare(selfish), AND
  max individual gold(selfish) > max individual gold(cooperative).

## Deliverables

- `experiments/e44_emergent_economy.py` - the composed economy + four claims,
  deterministic/offline/self-checking; writes `results/e44_emergent_economy.json`.
- A multi-panel figure (price formation; money supply & price under sink
  toggle; Gini under redistribution toggle; selfish-vs-cooperative welfare/Gini
  trade-off) via `scripts/make_paper_assets.py`.
- Paper subsection (capstone) with the four claims and the causal-attribution
  framing; `\NumExperiments` 42 -> 43.
- PR.

## Honest boundaries

- This is a stylized economy, not a calibrated market model; the claims are
  about emergence and clean causal attribution within a composed verified world,
  not quantitative macroeconomic realism.
- Agent policies are simple heuristics (greedy gold-max / welfare-max), not
  learned equilibria; we report what emerges from these fixed policies, and the
  selfish/cooperative result is a property of those policies, not a Nash claim.
- Determinism comes from a fixed seed; we note the result is one realization and
  the self-checks are structural (signs of effects), not point estimates.
