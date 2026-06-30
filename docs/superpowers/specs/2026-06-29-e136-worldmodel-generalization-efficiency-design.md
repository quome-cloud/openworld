# E136 — Generalizing world-model induction + action efficiency: Design

**Goal:** Refocus the ARC-AGI-3 effort from *exploration* (falsified: E135 spectral/graph search gave 0 lift)
onto the two levers that actually win the full benchmark — (1) a predictive world model that **generalizes
to the next level's novel mechanic**, and (2) **action efficiency** measured against a human-style baseline,
not just completion. Both are falsifiable with metrics we can compute today.

**Why this design.** ARC-AGI-3 is engineered against blind exploration: state-space explosion on later
levels makes graph search intractable, and scoring is **Completion + Efficiency vs a median human baseline**
— a graph-mapper that burns hundreds of actions scores near-zero efficiency even when it completes. Humans
take a few exploratory steps, form a symbolic rule, then execute a near-optimal direct path. The EWM agent
(E125/E133/E134) already has this shape — perceive in objects → synthesize `predict(s,a)→s'` → reason the
win → `plan_in_model` the exact path → execute. So the question is **not** "explore smarter" but: *why does
that loop fail on the residual walls?* Two measurable hypotheses:

- **H1 (generalization gap):** the level-N `predict()` does not correctly predict level-(N+1) transitions —
  the novel mechanic isn't captured, so `plan_in_model` searches a wrong model and never contains the win.
- **H2 (efficiency gap):** even when a level is solved, the agent uses far more real-env actions than the
  optimal in-model plan — excessive trial-and-error rather than explore-then-plan.

## Non-goals (YAGNI / on the record)
- No new exploration/search strategy (E135 closed that door). No graph/spectral perceptor.
- No new third-party deps; core stays stdlib, experiments may use numpy.
- We do **not** claim to crack a wall here — E136 first *measures* the two gaps honestly; any solver
  improvement is gated on the measurement showing the gap is the binding one.

## Architecture (three units, each independently testable)

### 1. World-model transfer metric  (`experiments/e136/transfer.py`)
The falsifier for H1. Given a game and a frontier seed at level N:
- `gather(game, level) -> [(s_key, action, s_key', levels)]` — forward-explore one level, collect transitions.
- `induce_predict(transitions) -> predict_fn` — fit the agent's object-relative rule (reuse E125's
  `predict()` synthesis / FunSearch keep-best) on level-N transitions only.
- `transfer_accuracy(predict_fn, next_level_transitions) -> float` — fraction of level-(N+1) observed
  transitions the level-N model predicts correctly (held-out, the env is ground truth).
- Report per-game: in-level accuracy (sanity, should be high) vs **transfer accuracy N→N+1** (the gap).
- **Deterministic, offline-where-possible**; `save_results` before any assert (CLAUDE.md).

### 2. Action-efficiency metric  (`experiments/e136/efficiency.py`)
The falsifier for H2, computed from the captured agent runs (HF traces / `arc3_traces`) + the archive:
- `solution_actions(game) -> int` — length of the banked source-free solution (the executed path).
- `optimal_plan_len(game) -> int` — shortest `plan_in_model` path that reproduces the win in the
  *synthesized* model (the test-time-planning ideal — the "how" length, no exploration).
- `efficiency(game) -> {real_actions, plan_actions, ratio}` — `ratio = real_actions / plan_actions`
  (≈1.0 = explore-then-execute like a human; ≫1 = wasteful trial-and-error). If a human baseline is
  ever available, swap it in as the denominator; until then the in-model optimal is the honest proxy.
- An **efficiency column added to the source-free archive** so every solved game reports completion AND
  an action-efficiency proxy — we stop reporting completion alone.

### 3. Generalization lever (only if H1 confirmed)  (`experiments/e136/induce.py`)
If transfer accuracy is the binding gap, strengthen induction toward simplicity-biased generalization:
- Induce `predict()` as the **simplest object-relative program that reproduces ALL observed transitions**
  (Occam → generalizes), mirroring the verified-shortest-path idea (`tool_graph.graph_search`): among
  rule programs that pass the execution gate on level-N transitions, prefer the minimal one, then measure
  whether minimality *raises* transfer accuracy N→N+1 vs the current keep-best.
- This is a falsifiable ablation: simplicity-biased induction vs E125 keep-best, scored by transfer
  accuracy and by whether `plan_in_model` over the generalized model now *contains* a level-(N+1) win.

## Correctness anchors (TDD spine, `tests/e136/`)
1. `test_transfer_metric.py` — on a synthetic 2-level world where the level-2 rule = level-1 rule
   (transfer should be ~1.0) vs a world where level-2 introduces a new operator (transfer < 1.0). Asserts
   the metric separates the two. The metric must *detect* a generalization gap or it's useless.
2. `test_efficiency_metric.py` — synthetic run with known explore vs plan lengths; assert `ratio`
   equals real/plan and that a pure explore-then-execute run scores ≈1.0.
3. `test_induce_occam.py` — given transitions consistent with two rules (one minimal, one overfit), assert
   the inducer returns the minimal program and that it generalizes to a held-out transition the overfit one misses.

## Honest risks / limitations
- **The goal-as-procedure wall persists.** A faithful, generalizing model is *necessary not sufficient* —
  the win is still an ordered protocol the agent must reason. E136 measures/improves the model half; it
  does not claim to dissolve goal inference. Reported either way.
- **No real human baseline yet.** The in-model optimal plan length is an honest *proxy* for efficiency, not
  the ARC-AGI-3 human median. Flag this clearly in any paper text; do not present the proxy as the official score.
- **Transfer may be low for irreducibly-novel mechanics.** If level-(N+1) introduces a genuinely new
  operator, no amount of level-N induction transfers — that is itself a finding (it bounds the approach and
  points back to the agent's reasoning, not the inducer).

## Paper home
Extends the "strategy landscape" section (where E135's exploration-negative + the forming-vs-verifiable
figure now live). E136 turns the prose claim "the lever is world modeling + planning, not exploration"
into two measured curves: transfer accuracy and action efficiency.
