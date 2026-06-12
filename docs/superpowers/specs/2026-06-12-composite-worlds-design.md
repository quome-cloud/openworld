# Composite, nested, and non-stationary worlds: design

**Date:** 2026-06-12
**Status:** approved (design); three sub-projects, planned individually

## Goal

Extend OpenWorld to three scenarios the paper does not yet cover:

1. **Compositional worlds** — independent worlds merged into one macrostate,
   coupled through *bridges* (trade, migration, flows).
2. **Nested worlds** — worlds within worlds (earth ⊃ country ⊃ state ⊃ city),
   with derived parent summaries and downward parameter flow.
3. **Dynamic worlds** — rules that change over time (regime switches), with
   ahead-of-time verification preserved.

## The unifying principle

Everything stays a pure, verified `(state, action) -> state` function over a
dict state. Composition is a *wrapper* over unmodified child worlds; nesting
is composition applied recursively; rule change is a transition over
transitions. No child world, verifier, simulation loop, judge, or planner
changes behavior. This keeps the paper's central guarantee — bit-exact,
ahead-of-time-verified dynamics — intact at every scale.

## Sub-project A: `openworld/compose.py`

### CompositeWorld

`CompositeWorld(name, children: Dict[str, World], bridges=(), aggregators=(),
bindings=(), timescales=None, description="", rules=None)` — a `World`
subclass (closed under composition: composites nest).

- **State:** `{ns: dict(child.initial_state) for ns, child in children}` plus
  a derived `"_agg"` dict. Child states nest as plain sub-dicts;
  `WorldState.copy()` already deep-copies.
- **Actions:** every child action namespaced `f"{ns}:{action}"`, plus
  `"tick"`.
- **Step semantics** (one composite step, in order):
  1. *Bindings:* for each `Binding(source_path, child_ns, key)`, copy the
     value at `source_path` (e.g. `("_agg", "gdp_total")` or
     `("usa", "rate")`) into the child slice under `key`. One-directional,
     applied before the child steps.
  2. *Route:* `"{ns}:{act}"` slices `state[ns]` into a `WorldState`, steps
     that child's transition with `Action(act, params, agent)`, writes the
     result back. `"tick"` steps every child with its declared default
     action, `timescales[ns]` times (default 1) — cities can tick daily
     while the country ticks quarterly.
  3. *Bridges:* fire in declared order (see below).
  4. *Aggregators:* recompute `state["_agg"]` from child slices.
- Unknown namespace or action → state returned unchanged (matching the
  framework's tolerant transition style).

### Bridge

`Bridge(name, a, b, transition, description="", rules=())` — a coupling
between child namespaces `a` and `b`. Its `transition` is an ordinary
`Transition` over the two-slot dict `{"a": <state of a>, "b": <state of b>}`,
stepped with `Action("flow")`. Because that is just a transition over a dict,
the existing `synthesize_transition`/`Verifier` pipeline applies unchanged:
bridges can be hand-written (`FunctionTransition`) or synthesized from
plain-language rules with cross-world conservation invariants
(`lambda s: s["a"]["pop"] + s["b"]["pop"] == 12_000`). A
`compile_bridge(llm, bridge_spec, invariants)` helper wraps synthesis with a
two-slot smoke state.

### Aggregator

`Aggregator(name, fn)` where `fn(children: Dict[str, dict]) -> value`.
Recomputed after every step into `state["_agg"][name]`. Parent-level
quantities are **derived, never independently simulated** — hierarchy cannot
drift from its leaves, and exactness at the leaves implies exactness of every
summary.

### Binding

`Binding(source_path: Tuple[str, ...], child_ns: str, key: str)` — downward
parameter flow only. Upward influence happens only through aggregators,
sideways only through bridges, so every causal channel in a composite is a
declared, inspectable object.

### Nesting

No additional machinery: a `CompositeWorld` is a `World`, so it can be a
child of another `CompositeWorld`. Earth ⊃ country ⊃ state ⊃ city is three
nested composites; leaf dynamics stay ordinary (possibly synthesized) worlds.

### Tests (`tests/test_compose.py`, all offline, FunctionTransition children)

- routing: namespaced action steps exactly its child; unknown ns is a no-op
- tick + timescales: `timescales={"city": 3}` runs the city default action
  3× per composite tick
- bindings: parent/global value visible in the child slice before its step
- bridges: a hand-written conservation bridge moves quantity, total conserved
  over a rollout; bridge order respected
- aggregators: `_agg` equals recomputation from leaves after every step
- nesting: a 3-level composite (composite-of-composites) routes
  `country:city:work`-style actions and keeps leaf/aggregate consistency
- `reset()` restores the full nested initial state

### Exports

`CompositeWorld, Bridge, Aggregator, Binding` from `openworld/__init__.py`.

## Sub-project B: `PhasedTransition` (in `openworld/transition.py`)

`PhasedTransition(phases: List[Tuple[trigger, Transition]], record_key="_phase")`

- `trigger`: an `int` (becomes active once `state["_phase_steps"] >= n` —
  step-count threshold) or a callable `state -> bool`.
- **Sequential, irreversible advance:** before delegating, check the *next*
  phase's trigger; if true, advance. Regimes do not revert. The active phase
  index is written to `state[record_key]` (and a step counter to
  `"_phase_steps"`), so trajectories are replayable and regime switches are
  visible in the record.
- Every phase transition is constructed (and, when synthesized, verified)
  **before the run** — ahead-of-time verification is preserved. Live mid-run
  re-synthesis is explicitly out of scope.
- Phase 0's trigger is ignored (it is the starting regime).
- Parameter drift that can be encoded in state (scheduled rate changes)
  remains the documented pattern: put the regime variable in state and branch
  in the rules. `PhasedTransition` is for *structural* change only.

Tests (`tests/test_phases.py`): step-count trigger fires once and persists;
predicate trigger; `_phase` recorded; phases verified independently
(constructed from two `FunctionTransition`s with different laws); works as a
`CompositeWorld` child's transition (composes with sub-project A).

## Sub-project C: experiments E30–E32 + paper

- **E30 — composition vs the complexity cliff.** Reuse E20's parametric rule
  machinery: an R=16 oracle partitioned into 4 children × 4 rules with the
  cross-group interactions expressed as bridges. Conditions: monolithic
  synthesis of all 16 rules (E20 showed collapse past R≈8) vs compositional
  synthesis (4 children + bridges, each verified separately), scored on the
  same ground-truth probes. Hypothesis: composition restores accuracy by
  keeping every synthesis task under the cliff.
- **E31 — nested fidelity.** A 3-level region ⊃ 2 countries ⊃ 2 cities world
  with a hand-written whole-system oracle; 20-step rollouts; metrics: exact
  leaf match, aggregator consistency at every step, conservation invariants
  (population, money) never violated.
- **E32 — regime switch.** A two-phase economy (policy change at step 10):
  (a) `PhasedTransition` with two pre-verified phases, (b) monolithic
  synthesis from the full rules-with-change text, (c) LLM next-state proxy.
  Metric: exact match before, across, and after the boundary.
- Results into `experiments/results/e3{0,1,2}_*.json`; `make_paper_assets.py`
  emits a table/figure; `paper/main.tex` gains a Results subsection
  ("Composition, hierarchy, and changing rules") extending the E20
  story, plus Experimental Setup coverage; `NumExperiments` bumped.

## Out of scope

- Live mid-run re-synthesis of dynamics (flag-gated future work).
- Asynchronous/event-queue scheduling between worlds (actor model).
- Bridges spanning more than two children (compose pairwise instead).
- Cross-composite agent planning (agents act through namespaced actions).
