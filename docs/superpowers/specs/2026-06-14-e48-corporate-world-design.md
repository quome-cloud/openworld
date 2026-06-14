# E48 - Composite corporate world: optimizing individual, division, and company goals

**Date:** 2026-06-14
**Status:** approved (design); pending spec review

## Goal

Model a tech company (a DigitalOcean-style PaaS org: database, serverless,
storage, compute, networking divisions) as a nested `CompositeWorld`, and show
how agents at different levels of the org chart (senior SWE, director, CEO)
navigate the graph to serve goals at three scales - individual promotion,
division growth, and aggregate company growth - and how those goals align or
conflict. Deterministic, offline, self-checking. The structural showcase is the
framework's composition + aggregation + hierarchical observation; the scientific
core is the individual-vs-collective optimization tension (the E44/E08 Pareto on
an org chart) plus where decision leverage lives in the hierarchy.

## Structure (nested CompositeWorld)

Company (root) -> Divisions -> Teams -> Individuals. Each level is a World; the
parent is a `CompositeWorld` of its children.

- Individual state: `{level, skill, impact, promo_progress}`.
- Division state: `{revenue, headcount, focus}`; `growth_rate` and `revenue` are
  `Aggregator`s over its teams/individuals.
- Company state: `total_revenue`, `growth_rate` as `Aggregator`s over divisions;
  `budget` (the CEO's macro lever).

Revenue has **diminishing returns in effort per division** (concave), so total
company growth is maximized by allocating effort/budget across divisions to
equalize marginal return - the same concavity that drove E44's cooperation
result. This is what makes individual-optimal differ from company-optimal.

## Agents traverse the graph (observe() + roles)

Agents are employees at levels with different **action scope** and **observation
scope** (the framework's `observe()` already returns local detail + ancestor
aggregates - hierarchical attention):

- **Senior SWE (IC):** action = choose which project/division to pour effort into
  -> own impact + local team output. Observes team + division aggregate.
- **Director:** action = allocate effort/headcount across teams in the division,
  hire -> division growth. Observes division + company aggregate.
- **CEO:** action = allocate `budget` across divisions -> macro growth. Observes
  all divisions' aggregates.

## Perception (transcripts + internal data)

Meeting transcripts (1:1s, team standups, division reviews, company all-hands)
and Slack are **generated deterministically from the world state** (templated
text), and agents perceive the org back through a lossy extractor (rounding,
omission, staleness; all-hands aggregates are coarser than a 1:1). An agent may
act on perceived state instead of ground truth. No LLM in the core (the extractor
is a deterministic parser); an optional LLM transcript sample is out of scope.

## Four experiments (all at the agent level)

1. **Individual vs collective (the Pareto).** Sweep a selfishness dial `rho`:
   `rho=1` every agent optimizes its OWN objective (ICs pile onto the hottest
   division for visibility/promotion; directors hoard budget), `rho=0` agents
   optimize toward company growth (allocate by marginal return). Measure company
   aggregate growth AND individual promotions / their distribution. Expect:
   selfish concentration -> lower aggregate growth (diminishing returns) but a
   few big promotions; aligned -> higher company growth, more even promotions.
2. **Value-of-action by level.** Causal attribution: enable each level's optimal
   action while the others stay greedy, and measure the marginal company growth
   it buys (CEO budget reallocation vs director team-allocation vs IC project
   choice). Expect the CEO's portfolio allocation to be the dominant lever, ICs
   the smallest - quantified.
3. **Perception cost.** Each role acts on transcript-derived perceived state vs
   ground truth; measure the growth/promotion-quality gap. Expect the CEO (who
   relies on coarse all-hands aggregates) to lose the most to perception.
4. **Optimal navigation policy.** For each role, compare a principled policy
   (act on marginal value / hierarchical observation) to a greedy/myopic baseline
   on that role's objective; the principled policy wins.

## Self-checks (asserts)

- revenue/growth Aggregators equal the explicit sum over children (composition
  never drifts).
- aligned (`rho=0`) company growth > selfish (`rho=1`) company growth (the
  Pareto: local optimization hurts the aggregate under concave returns).
- selfish regime concentrates promotions (higher promotion Gini) than aligned.
- CEO-action marginal growth > director-action > IC-action (leverage hierarchy).
- ground-truth decisions >= perceived-state decisions on company growth; gap > 0.
- each role's principled policy beats its greedy baseline on its own objective.

## Deliverables

- `experiments/e48_corporate_world.py` (+ `results/e48_corporate_world.json`),
  deterministic/offline/self-checking; builds the org as a real `CompositeWorld`
  with `Aggregator`s, drives quarters + the four analyses with explicit verified
  functions (the E44 pattern).
- Figure (Pareto: company growth vs selfishness with promotion distribution;
  value-of-action bars by level; perception-cost bars by role; policy comparison)
  + table; paper subsection (corporate composite world); `\NumExperiments` 46->47.
- PR based on `main`, targeting `main` (per CLAUDE.md).

## Honest boundaries

- A stylized org model, not a calibrated business simulator; claims are about the
  structure of multi-level optimization (alignment, leverage, perception), not
  forecasting a real company's revenue.
- Agent policies are fixed heuristics (selfish / aligned / greedy / principled),
  not learned equilibria; results are properties of those policies.
- Transcripts are templated synthetic text with a deterministic extractor; the
  perception result is about information loss in aggregation/rounding/staleness,
  not natural-language understanding.
