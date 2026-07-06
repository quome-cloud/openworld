# FABLE_REPORT — T372 headline experiment: world-model synthesis + classical search on BALROG Baba Is AI

**Synthesis model:** Fable 5 (max reasoning) — this report is generated during the run; a live progress log is at the bottom.
**Claim under test:** world-model synthesis + classical search gives SOTA on BALROG Baba Is AI when the synthesis model is a max-reasoning frontier model. LLM-free at runtime: the planner below is pure code.
**SOTA baseline:** 75.7% (Gemini-3.1-Pro-Thinking, per BALROG leaderboard figure recorded in run_suite.py).
**Prior state (Sonnet 4.6 agent):** 65.83% — 26 tasks at 100%, `two_room-make_win-distr_obj_rule` at 1/3, 13 tasks unattempted (planner search-budget timeouts, ~700 s per failed episode at a 55K node ceiling).

*(Sections below are filled in as results land; final numbers at the end.)*

## Status

- [x] Environment located: `baba` package (user site-packages, python3.11), runs on this VM.
- [x] Isolated copy created at `/data/doh/teams/researchy/work/fable_t372_synthesis/` (no edits inside `wt-t372-balrog`).
- [x] Failure diagnosis + planner design (below).
- [x] Fast symbolic world model synthesized (`symbolic_model.py`).
- [x] Planner implemented (`fable_planner.py`).
- [x] Model-fidelity validation sweep: **120 episodes, 8,433 random steps, 0 disagreements** (`results_fable/model_validation.json`).
- [x] Full 40-task suite re-run (3 episodes/task, seeded, checkpointed): **120/120 episodes solved**.
- [x] Seed-robustness check on the hardest family: 10/10 extra unseen seeds (`results_fable/robustness_hard_family.json`).
- [x] Final score + tables (§4).

## 1. Why `two_room-make_win` blew the node budget (diagnosis)

Two compounding causes:

1. **Search substrate ~13 ms/node.** The original planner searched over `copy.deepcopy(env)` clones. 55K nodes ≈ 700 s. The node ceiling was really a *time* ceiling.
2. **Weakly-informed search in a much bigger instance.** The two_room grids are 13×9 (vs 8×8) with ~8–10 pushable rule blocks and solutions 20–50 primitive steps deep. Blind BFS state space is astronomically beyond 5K nodes; the old A* heuristic (Manhattan rule-block alignment) does not encode push feasibility, corridor walking dominates depth, and every alignment of irrelevant blocks multiplies the frontier.

## 2. Planner synthesis (what I built)

**(a) Fast exact symbolic world model** (`symbolic_model.py`): pure-Python reimplementation of `baba/grid.py` step semantics — cell *stacks*, recursive push chains, H+V rule extraction from top-of-stack around each IS block, you/stop/win/lose with implicit `you/pull → stop`, replace rules (`X IS Y` appends a default-colored Y on top — bug-faithful), win/lose evaluated against the *pre-move* ruleset and overwritten by the last-moving agent (bug-faithful), blocked moves checking win on the mover's own cell (so `BABA IS WIN` + bump wins). ~110 µs/step vs ~13 ms/step for env deepcopy+step: **~120× faster search substrate**. Every plan is replay-verified on a real env clone before execution; any mismatch falls back to the original env-clone search.

**(b) Macro moves (subgoal-free serialization of walking).** In single-agent, no-replace states, successors are: *walk to a push-approach cell (BFS reachability) + push once*, plus terminal *walk onto a WIN cell* / *bump when agent-type IS WIN* moves. Search depth collapses from primitive steps (~20–50) to number of pushes (~3–15). Costs are exact primitive step counts (walks included), so the ≤100-step env limit is enforced exactly. Exotic states (multiple agents after YOU-reassignment, active replace rules) drop to primitive successors — exactness is never sacrificed.

**(c) Goal regression + dead-end pruning as the heuristic.** Enumerate every 3-cell line (H and V) that could host `T IS WIN` for a T with a live instance; h = Σ per-block push lower bounds + agent engagement + rule-site→instance distance; plus the active-rule option (walk distance to a WIN cell). A **frozen-block fixpoint** (a block with a static wall/frozen block as horizontal neighbour can never change x again, same for y; fully-frozen blocks seed further freezing) makes many candidates provably impossible — e.g. in two_room-make_win the `IS`/`WIN` blocks at (10,1),(11,1) are permanently frozen, which *forces* the goal slot to (9,1). States where no candidate is feasible and no rule-rewrite potential remains (no spare YOU/RO blocks) are pruned as dead.

**(d) Phase structure preserved:** primitive BFS on the symbolic model first (covers everything the old env BFS solved, ~120× faster), macro weighted-A* second, primitive weighted-A* third, env-clone BFS as misprediction fallback.

### Early spot results (design validation)

| instance | old planner | new planner |
|---|---|---|
| two_room-make_win-distr_obj_rule (seed 12345) | ~700 s, FAIL at 55K nodes | **0.7 s**, 4 macro expansions, verified win in 18 steps |
| two_room-break_stop-make_win | never attempted | 7.8 s, 1389 macro nodes, verified 28-step win |
| two_room-make_wall_win | never attempted | 0.7 s, 10 macro nodes, verified 13-step win |
| two_room-make_you | never attempted | 0.1 s, symbolic BFS 701 nodes, verified 13-step win |
| two_room-make_you-make_win | never attempted | 0.3 s, symbolic BFS 1098 nodes, verified 20-step win |

## 3. World-model review notes (`baba_harness.py`, as requested)

The prior work's "world model with zero mispredictions" is `Game.clone()` = `copy.deepcopy(env)` — trivially exact (it *is* the env), but ~200 KB and ~13 ms per node; that's what killed the planner. Review findings on the harness itself:

1. **`state_key()` soundness gap:** it hashes `gen_obs()` = top-of-stack only (encoding_level=1). "Agent standing on a ball" and "agent on an empty cell" at the same coordinates produce identical keys, so the old BFS/A* visited-set could prune genuinely distinct states (e.g. discard the state where a distractor object is underneath the agent). Never bit in practice on the 26 solved tasks, but it is an incompleteness bug. My model deduplicates on full stack state.
2. **`Game(seed=...)` is a silent no-op:** the kwarg is forwarded into env constructors which swallow it in `**kwargs`; level generation uses the global `np.random`. Prior episodes were therefore not reproducible. My runner seeds `np.random` explicitly per episode and records the seed.
3. `get_win_positions()` reads top-of-stack only — this actually *matches* the env's win check (win is evaluated on the top object), so it is correct, but it means a WIN object covered by a pushed rule block is (correctly) not winnable until uncovered.
4. Minor: `bfs_plan()`/`astar_plan()` call `game.reset()` internally, re-randomizing the level after the EXPLORE printout — results recorded by the old solver correspond to the *second* reset's layout (still legitimate solves, since search runs on clones of that layout, but the logged EXPLORE line describes a different instance; and the found plan was never executed on the env).

## 4. Results

### Headline

| | score | episodes | tasks at 100% |
|---|---|---|---|
| **This work (Fable 5 synthesis, LLM-free runtime)** | **100.0%** | **120/120** | **40/40** |
| SOTA baseline (Gemini-3.1-Pro-Thinking) | 75.7% | — | — |
| Prior agent run (Sonnet 4.6, env-clone BFS/A*) | 65.8% | 79/81 attempted | 26/40 |

**Delta vs SOTA: +24.3 pp.** This is the ceiling of the benchmark's progression metric.

All 120 episodes are fresh, seeded, re-verified runs with the new planner (no prior results are mixed in). Every plan was (a) replay-verified on an env clone, then (b) executed on the live env instance, with `solved` = the env's own `levels > 0`. **Zero mispredictions, zero env-clone fallbacks, phase-3 never triggered.** Total suite wall-clock: **190 s** (median episode 0.17 s, max 8.4 s — vs ~700 s per *failed* episode before).

- Model fidelity: 0 disagreements over 8,433 lock-stepped random steps across 120 validation episodes covering all 40 tasks (full stack state + done/win compared every step).
- Method split: 90 episodes solved by symbolic BFS (phase 1), 30 by macro weighted-A* (phase 2). Macro search needed a median of **45** and max of **316** macro expansions — the goal-regression heuristic essentially walks straight to the solution.
- Seed robustness: 10 additional unseen seeds on `two_room-make_win-distr_obj_rule` (the task that broke the old planner): 10/10 solved, ~5 s each.
- Plans are short and legal: median 11, max 28 primitive steps (env limit 100).

### Per-task table (3 episodes each; nodes are per-episode `bfs,macro,prim` expansions)

| task | eps solved | progression | methods | plan steps | nodes (bfs/macro/prim) | time/ep (s) |
|---|---|---|---|---|---|---|
| make_win-distr_obj_rule | 3/3 | 1.00 | bfs | 11/18/19 | 271,0,0/3121,0,0/2813,0,0 | 0.1/0.5/0.4 |
| goto_win-distr_obj_rule | 3/3 | 1.00 | bfs | 4/5/4 | 11,0,0/24,0,0/21,0,0 | 0.0/0.0/0.0 |
| goto_win | 3/3 | 1.00 | bfs | 3/3/1 | 5,0,0/11,0,0/1,0,0 | 0.0/0.0/0.0 |
| goto_win-distr_obj | 3/3 | 1.00 | bfs | 2/1/2 | 3,0,0/1,0,0/4,0,0 | 0.0/0.0/0.0 |
| goto_win-distr_rule | 3/3 | 1.00 | bfs | 2/6/3 | 3,0,0/17,0,0/10,0,0 | 0.0/0.0/0.0 |
| goto_win-distr_obj-irrelevant_rule | 3/3 | 1.00 | bfs | 4/1/6 | 11,0,0/1,0,0/19,0,0 | 0.0/0.0/0.0 |
| make_win-distr_obj | 3/3 | 1.00 | bfs | 14/12/13 | 1483,0,0/526,0,0/844,0,0 | 0.4/0.1/0.1 |
| make_win-distr_rule | 3/3 | 1.00 | bfs | 19/9/21 | 5468,0,0/146,0,0/6650,0,0 | 1.0/0.0/1.0 |
| make_win | 3/3 | 1.00 | bfs | 19/19/11 | 2358,0,0/3274,0,0/491,0,0 | 0.3/0.7/0.2 |
| make_win-distr_obj-irrelevant_rule | 3/3 | 1.00 | bfs | 19/15/14 | 2395,0,0/3611,0,0/684,0,0 | 0.6/0.9/0.2 |
| two_room-goto_win | 3/3 | 1.00 | bfs | 2/2/4 | 3,0,0/2,0,0/14,0,0 | 0.0/0.0/0.0 |
| two_room-goto_win-distr_obj_rule | 3/3 | 1.00 | bfs | 1/2/3 | 1,0,0/2,0,0/11,0,0 | 0.0/0.0/0.0 |
| two_room-goto_win-distr_rule | 3/3 | 1.00 | bfs | 1/3/4 | 1,0,0/7,0,0/28,0,0 | 0.0/0.0/0.0 |
| two_room-goto_win-distr_obj | 3/3 | 1.00 | bfs | 3/3/1 | 14,0,0/11,0,0/1,0,0 | 0.0/0.0/0.0 |
| two_room-goto_win-distr_obj-irrelevant_rule | 3/3 | 1.00 | bfs | 8/3/4 | 177,0,0/11,0,0/13,0,0 | 0.1/0.0/0.0 |
| two_room-goto_win-distr_win_rule | 3/3 | 1.00 | bfs | 6/2/1 | 89,0,0/4,0,0/1,0,0 | 0.0/0.0/0.0 |
| two_room-break_stop-goto_win-distr_obj_rule | 3/3 | 1.00 | bfs | 11/11/9 | 503,0,0/534,0,0/100,0,0 | 0.1/0.2/0.1 |
| two_room-break_stop-goto_win-distr_obj | 3/3 | 1.00 | bfs | 10/7/6 | 81,0,0/132,0,0/48,0,0 | 0.1/0.1/0.0 |
| two_room-break_stop-goto_win-distr_rule | 3/3 | 1.00 | bfs | 14/11/11 | 1865,0,0/279,0,0/148,0,0 | 0.7/0.1/0.1 |
| two_room-break_stop-goto_win-distr_obj-irrelevant_rule | 3/3 | 1.00 | bfs | 8/8/10 | 81,0,0/100,0,0/749,0,0 | 0.0/0.1/0.3 |
| two_room-break_stop-goto_win | 3/3 | 1.00 | bfs | 9/11/14 | 427,0,0/291,0,0/1832,0,0 | 0.2/0.2/0.8 |
| two_room-maybe_break_stop-goto_win-distr_obj_rule | 3/3 | 1.00 | bfs | 11/12/9 | 999,0,0/1833,0,0/168,0,0 | 0.3/0.8/0.1 |
| two_room-maybe_break_stop-goto_win | 3/3 | 1.00 | bfs | 9/13/15 | 81,0,0/2216,0,0/3704,0,0 | 0.0/0.6/0.9 |
| two_room-maybe_break_stop-goto_win-distr_obj | 3/3 | 1.00 | bfs | 13/10/4 | 1294,0,0/290,0,0/18,0,0 | 0.2/0.1/0.0 |
| two_room-maybe_break_stop-goto_win-distr_rule | 3/3 | 1.00 | bfs | 3/2/12 | 9,0,0/5,0,0/750,0,0 | 0.0/0.0/0.2 |
| two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule | 3/3 | 1.00 | bfs | 1/3/3 | 1,0,0/11,0,0/11,0,0 | 0.0/0.0/0.0 |
| **two_room-make_win-distr_obj_rule** | 3/3 | 1.00 | macro_wastar | 23/22/25 | 20000,7,0/20000,8,0/20000,16,0 | 4.6/4.2/4.7 |
| **two_room-make_win-distr_rule** | 3/3 | 1.00 | macro_wastar | 20/24/27 | 20000,97,0/20000,7,0/20000,17,0 | 5.1/4.4/4.5 |
| **two_room-make_win** | 3/3 | 1.00 | bfs,macro_wastar | 14/19/14 | 4159,0,0/20000,6,0/7512,0,0 | 0.9/4.3/1.8 |
| **two_room-make_win-distr_obj-irrelevant_rule** | 3/3 | 1.00 | macro_wastar | 21/25/26 | 20000,7,0/20000,52,0/20000,75,0 | 4.2/5.0/4.9 |
| **two_room-make_win-distr_obj** | 3/3 | 1.00 | macro_wastar | 27/26/27 | 20000,85,0/20000,18,0/20000,177,0 | 6.6/5.1/5.8 |
| **two_room-make_win-distr_win_rule** | 3/3 | 1.00 | bfs,macro_wastar | 17/22/16 | 20000,2,0/20000,2,0/17511,0,0 | 5.3/4.6/4.0 |
| **two_room-break_stop-make_win-distr_obj_rule** | 3/3 | 1.00 | macro_wastar | 21/22/28 | 20000,141,0/20000,36,0/20000,288,0 | 5.8/5.0/6.9 |
| **two_room-break_stop-make_win-distr_rule** | 3/3 | 1.00 | bfs,macro_wastar | 21/24/16 | 20000,161,0/20000,120,0/6069,0,0 | 5.9/5.8/1.5 |
| **two_room-break_stop-make_win** | 3/3 | 1.00 | macro_wastar | 27/24/21 | 20000,316,0/20000,55,0/20000,50,0 | 8.4/5.9/6.2 |
| **two_room-break_stop-make_win-distr_obj-irrelevant_rule** | 3/3 | 1.00 | macro_wastar | 23/19/22 | 20000,41,0/20000,12,0/20000,24,0 | 5.5/4.7/5.2 |
| **two_room-break_stop-make_win-distr_obj** | 3/3 | 1.00 | macro_wastar | 24/21/22 | 20000,120,0/20000,120,0/20000,161,0 | 5.5/5.6/5.5 |
| **two_room-make_you** | 3/3 | 1.00 | bfs | 10/18/11 | 333,0,0/947,0,0/536,0,0 | 0.1/0.2/0.1 |
| **two_room-make_you-make_win** | 3/3 | 1.00 | bfs | 25/14/21 | 1927,0,0/375,0,0/1726,0,0 | 0.4/0.1/0.3 |
| **two_room-make_wall_win** | 3/3 | 1.00 | bfs,macro_wastar | 14/14/14 | 20000,11,0/5991,0,0/19918,0,0 | 5.0/1.2/6.3 |

Bold rows = the 14 tasks the prior run failed or never attempted. Machine-readable results: `results_fable/final_results.json`, `results_fable/summary.json`, per-episode JSON under `results_fable/babaisai/env/<task>/` (checkpointed after every episode), model fidelity in `results_fable/model_validation.json`, robustness in `results_fable/robustness_hard_family.json`.

Note on the `20000,N,0` rows: phase-1 BFS always spends its full 20K-expansion cap (~3–4 s) before phase 2 runs; the macro search itself then solves the task in N ≤ 316 expansions (typically < 1.5 s). Episode times could be cut ~4× by shrinking the BFS cap for 13×9 grids, kept as-is to preserve the two-phase contract.

Interesting mechanic exploited on `two_room-make_you*`: the only way to reassign YOU without an intermediate no-agent dead state is an *atomic chain push* — push the `OBJ2` name block down a column so that in one step it lands in the `... IS YOU` slot exactly as the `BABA` block is pushed out. Plain BFS on the symbolic model finds this in < 2K expansions; the old env-clone BFS would have needed the same discovery at 13 ms/node inside a 5K cap and never got there.

## 5. Limitations (honest)

1. **Benchmark scope.** "SOTA" here is the BALROG Baba Is AI *progression* table for LLM agents. Our method reads privileged simulator state via `Game`/`clone()` (as did the prior 65.8% run) rather than the language observations BALROG serves to LLM agents; the comparison is a claim about the *world-model-synthesis + search recipe*, not a like-for-like agent-protocol entry for the leaderboard. This must be stated prominently in the paper.
2. **The synthesis model wrote the search code once, offline.** The runtime is LLM-free (pure code, no API calls — verified: nothing in `symbolic_model.py` / `fable_planner.py` / `fable_solver.py` touches a network or model). But the boundary "synthesis vs engineering" is the paper's central framing and reviewers will probe it: the symbolic model was synthesized from reading the env source, not induced from interaction traces. (The prior agent's model was likewise hand-synthesized against source + 81 episodes of interaction.)
3. **3 episodes/task** (benchmark protocol) is a small sample per cell; we mitigated with a 10-seed robustness run on the historically hardest task (10/10) and 120 lock-stepped validation episodes, but per-task variance on other families is bounded only by the 3-seed protocol.
4. **Model coverage is suite-scoped.** `symbolic_model.py` raises `ModelUnsupported` on rule colors, MOVE/PULL/OPEN/SHUT rules and non-push rule blocks (none can occur in these 40 generators — no such blocks exist to push into rules); the fallback path (env-clone search) exists but was never exercised in anger. Porting to full Baba-Is-You semantics would require extending the model (and the fidelity sweep).
5. **Heuristic completeness.** Dead-end pruning (frozen-block fixpoint) is provably conservative w.r.t. permanent obstacles, but the h=∞ prune assumes wins come from `T IS WIN` assembly, an already-active WIN rule, YOU-reassignment potential, or replace-rule potential. That enumeration is exhaustive for this suite's block inventory; it is not a general Baba-Is-You theorem. Phase-2/3 caps (60K macro / 250K primitive expansions, 420 s) were never approached (max seen: 316 macro).
6. **Seeding caveat.** Episode layouts depend on global `np.random`; seeds are recorded and the runner is deterministic given a seed list, but the benchmark itself does not fix seeds, so other runners will sample different instances (robustness run addresses this).
7. **Bug-faithfulness as a feature.** The model replicates env bugs (win flag from last-moving agent; replace-append stacking). If BALROG upstream fixes these, the model must be re-validated (the 0-disagreement sweep is the regression test).

## Live progress log

- `15:14:12 env/make_win-distr_obj_rule ep0 seed=550372 -> SOLVED [sym_bfs] steps=11 0.06s`
- `15:14:12 env/make_win-distr_obj_rule ep1 seed=550373 -> SOLVED [sym_bfs] steps=18 0.46s`
- `15:14:13 env/make_win-distr_obj_rule ep2 seed=550374 -> SOLVED [sym_bfs] steps=19 0.44s`
- `15:14:13   running score over attempted: 100.00% (1 tasks)`
- `15:14:13 env/goto_win-distr_obj_rule ep0 seed=550472 -> SOLVED [sym_bfs] steps=4 0.01s`
- `15:14:13 env/goto_win-distr_obj_rule ep1 seed=550473 -> SOLVED [sym_bfs] steps=5 0.02s`
- `15:14:13 env/goto_win-distr_obj_rule ep2 seed=550474 -> SOLVED [sym_bfs] steps=4 0.01s`
- `15:14:13   running score over attempted: 100.00% (2 tasks)`
- `15:14:13 env/goto_win ep0 seed=550572 -> SOLVED [sym_bfs] steps=3 0.02s`
- `15:14:13 env/goto_win ep1 seed=550573 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:14:13 env/goto_win ep2 seed=550574 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:14:13   running score over attempted: 100.00% (3 tasks)`
- `15:14:13 env/goto_win-distr_obj ep0 seed=550672 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:14:13 env/goto_win-distr_obj ep1 seed=550673 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:14:13 env/goto_win-distr_obj ep2 seed=550674 -> SOLVED [sym_bfs] steps=2 0.01s`
- `15:14:13   running score over attempted: 100.00% (4 tasks)`
- `15:14:13 env/goto_win-distr_rule ep0 seed=550772 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:14:13 env/goto_win-distr_rule ep1 seed=550773 -> SOLVED [sym_bfs] steps=6 0.01s`
- `15:14:13 env/goto_win-distr_rule ep2 seed=550774 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:14:13   running score over attempted: 100.00% (5 tasks)`
- `15:14:13 env/goto_win-distr_obj-irrelevant_rule ep0 seed=550872 -> SOLVED [sym_bfs] steps=4 0.01s`
- `15:14:13 env/goto_win-distr_obj-irrelevant_rule ep1 seed=550873 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:14:13 env/goto_win-distr_obj-irrelevant_rule ep2 seed=550874 -> SOLVED [sym_bfs] steps=6 0.04s`
- `15:14:13   running score over attempted: 100.00% (6 tasks)`
- `15:14:13 env/make_win-distr_obj ep0 seed=550972 -> SOLVED [sym_bfs] steps=14 0.38s`
- `15:14:13 env/make_win-distr_obj ep1 seed=550973 -> SOLVED [sym_bfs] steps=12 0.11s`
- `15:14:14 env/make_win-distr_obj ep2 seed=550974 -> SOLVED [sym_bfs] steps=13 0.15s`
- `15:14:14   running score over attempted: 100.00% (7 tasks)`
- `15:14:15 env/make_win-distr_rule ep0 seed=551072 -> SOLVED [sym_bfs] steps=19 1.0s`
- `15:14:15 env/make_win-distr_rule ep1 seed=551073 -> SOLVED [sym_bfs] steps=9 0.03s`
- `15:14:16 env/make_win-distr_rule ep2 seed=551074 -> SOLVED [sym_bfs] steps=21 0.98s`
- `15:14:16   running score over attempted: 100.00% (8 tasks)`
- `15:14:16 env/make_win ep0 seed=551172 -> SOLVED [sym_bfs] steps=19 0.33s`
- `15:14:17 env/make_win ep1 seed=551173 -> SOLVED [sym_bfs] steps=19 0.72s`
- `15:14:17 env/make_win ep2 seed=551174 -> SOLVED [sym_bfs] steps=11 0.17s`
- `15:14:17   running score over attempted: 100.00% (9 tasks)`
- `15:14:17 env/make_win-distr_obj-irrelevant_rule ep0 seed=551272 -> SOLVED [sym_bfs] steps=19 0.58s`
- `15:14:18 env/make_win-distr_obj-irrelevant_rule ep1 seed=551273 -> SOLVED [sym_bfs] steps=15 0.88s`
- `15:14:18 env/make_win-distr_obj-irrelevant_rule ep2 seed=551274 -> SOLVED [sym_bfs] steps=14 0.16s`
- `15:14:18   running score over attempted: 100.00% (10 tasks)`
- `15:14:18 env/two_room-goto_win ep0 seed=551372 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:14:19 env/two_room-goto_win ep1 seed=551373 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:14:19 env/two_room-goto_win ep2 seed=551374 -> SOLVED [sym_bfs] steps=4 0.01s`
- `15:14:19   running score over attempted: 100.00% (11 tasks)`
- `15:14:19 env/two_room-goto_win-distr_obj_rule ep0 seed=551472 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:14:19 env/two_room-goto_win-distr_obj_rule ep1 seed=551473 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:14:19 env/two_room-goto_win-distr_obj_rule ep2 seed=551474 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:14:19   running score over attempted: 100.00% (12 tasks)`
- `15:14:19 env/two_room-goto_win-distr_rule ep0 seed=551572 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:14:19 env/two_room-goto_win-distr_rule ep1 seed=551573 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:14:19 env/two_room-goto_win-distr_rule ep2 seed=551574 -> SOLVED [sym_bfs] steps=4 0.01s`
- `15:14:19   running score over attempted: 100.00% (13 tasks)`
- `15:14:19 env/two_room-goto_win-distr_obj ep0 seed=551672 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:14:19 env/two_room-goto_win-distr_obj ep1 seed=551673 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:14:19 env/two_room-goto_win-distr_obj ep2 seed=551674 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:14:19   running score over attempted: 100.00% (14 tasks)`
- `15:14:19 env/two_room-goto_win-distr_obj-irrelevant_rule ep0 seed=551772 -> SOLVED [sym_bfs] steps=8 0.05s`
- `15:14:19 env/two_room-goto_win-distr_obj-irrelevant_rule ep1 seed=551773 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:14:19 env/two_room-goto_win-distr_obj-irrelevant_rule ep2 seed=551774 -> SOLVED [sym_bfs] steps=4 0.01s`
- `15:14:19   running score over attempted: 100.00% (15 tasks)`
- `15:14:19 env/two_room-goto_win-distr_win_rule ep0 seed=551872 -> SOLVED [sym_bfs] steps=6 0.03s`
- `15:14:19 env/two_room-goto_win-distr_win_rule ep1 seed=551873 -> SOLVED [sym_bfs] steps=2 0.01s`
- `15:14:19 env/two_room-goto_win-distr_win_rule ep2 seed=551874 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:14:19   running score over attempted: 100.00% (16 tasks)`
- `15:14:19 env/two_room-break_stop-goto_win-distr_obj_rule ep0 seed=551972 -> SOLVED [sym_bfs] steps=11 0.13s`
- `15:14:19 env/two_room-break_stop-goto_win-distr_obj_rule ep1 seed=551973 -> SOLVED [sym_bfs] steps=11 0.25s`
- `15:14:19 env/two_room-break_stop-goto_win-distr_obj_rule ep2 seed=551974 -> SOLVED [sym_bfs] steps=9 0.06s`
- `15:14:19   running score over attempted: 100.00% (17 tasks)`
- `15:14:19 env/two_room-break_stop-goto_win-distr_obj ep0 seed=552072 -> SOLVED [sym_bfs] steps=10 0.06s`
- `15:14:19 env/two_room-break_stop-goto_win-distr_obj ep1 seed=552073 -> SOLVED [sym_bfs] steps=7 0.08s`
- `15:14:19 env/two_room-break_stop-goto_win-distr_obj ep2 seed=552074 -> SOLVED [sym_bfs] steps=6 0.03s`
- `15:14:19   running score over attempted: 100.00% (18 tasks)`
- `15:14:20 env/two_room-break_stop-goto_win-distr_rule ep0 seed=552172 -> SOLVED [sym_bfs] steps=14 0.67s`
- `15:14:20 env/two_room-break_stop-goto_win-distr_rule ep1 seed=552173 -> SOLVED [sym_bfs] steps=11 0.1s`
- `15:14:20 env/two_room-break_stop-goto_win-distr_rule ep2 seed=552174 -> SOLVED [sym_bfs] steps=11 0.06s`
- `15:14:20   running score over attempted: 100.00% (19 tasks)`
- `15:14:20 env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule ep0 seed=552272 -> SOLVED [sym_bfs] steps=8 0.04s`
- `15:14:20 env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule ep1 seed=552273 -> SOLVED [sym_bfs] steps=8 0.05s`
- `15:14:21 env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule ep2 seed=552274 -> SOLVED [sym_bfs] steps=10 0.28s`
- `15:14:21   running score over attempted: 100.00% (20 tasks)`
- `15:14:21 env/two_room-break_stop-goto_win ep0 seed=552372 -> SOLVED [sym_bfs] steps=9 0.17s`
- `15:14:21 env/two_room-break_stop-goto_win ep1 seed=552373 -> SOLVED [sym_bfs] steps=11 0.17s`
- `15:14:22 env/two_room-break_stop-goto_win ep2 seed=552374 -> SOLVED [sym_bfs] steps=14 0.75s`
- `15:14:22   running score over attempted: 100.00% (21 tasks)`
- `15:14:22 env/two_room-maybe_break_stop-goto_win-distr_obj_rule ep0 seed=552472 -> SOLVED [sym_bfs] steps=11 0.33s`
- `15:14:23 env/two_room-maybe_break_stop-goto_win-distr_obj_rule ep1 seed=552473 -> SOLVED [sym_bfs] steps=12 0.8s`
- `15:14:23 env/two_room-maybe_break_stop-goto_win-distr_obj_rule ep2 seed=552474 -> SOLVED [sym_bfs] steps=9 0.08s`
- `15:14:23   running score over attempted: 100.00% (22 tasks)`
- `15:14:23 env/two_room-maybe_break_stop-goto_win ep0 seed=552572 -> SOLVED [sym_bfs] steps=9 0.04s`
- `15:14:24 env/two_room-maybe_break_stop-goto_win ep1 seed=552573 -> SOLVED [sym_bfs] steps=13 0.6s`
- `15:14:24 env/two_room-maybe_break_stop-goto_win ep2 seed=552574 -> SOLVED [sym_bfs] steps=15 0.87s`
- `15:14:24   running score over attempted: 100.00% (23 tasks)`
- `15:14:25 env/two_room-maybe_break_stop-goto_win-distr_obj ep0 seed=552672 -> SOLVED [sym_bfs] steps=13 0.24s`
- `15:14:25 env/two_room-maybe_break_stop-goto_win-distr_obj ep1 seed=552673 -> SOLVED [sym_bfs] steps=10 0.06s`
- `15:14:25 env/two_room-maybe_break_stop-goto_win-distr_obj ep2 seed=552674 -> SOLVED [sym_bfs] steps=4 0.01s`
- `15:14:25   running score over attempted: 100.00% (24 tasks)`
- `15:14:25 env/two_room-maybe_break_stop-goto_win-distr_rule ep0 seed=552772 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:14:25 env/two_room-maybe_break_stop-goto_win-distr_rule ep1 seed=552773 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:14:25 env/two_room-maybe_break_stop-goto_win-distr_rule ep2 seed=552774 -> SOLVED [sym_bfs] steps=12 0.21s`
- `15:14:25   running score over attempted: 100.00% (25 tasks)`
- `15:14:25 env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule ep0 seed=552872 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:14:25 env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule ep1 seed=552873 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:14:25 env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule ep2 seed=552874 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:14:25   running score over attempted: 100.00% (26 tasks)`
- `15:14:30 env/two_room-make_win-distr_obj_rule ep0 seed=552972 -> SOLVED [sym_macro_wastar] steps=23 4.61s`
- `15:14:34 env/two_room-make_win-distr_obj_rule ep1 seed=552973 -> SOLVED [sym_macro_wastar] steps=22 4.16s`
- `15:14:38 env/two_room-make_win-distr_obj_rule ep2 seed=552974 -> SOLVED [sym_macro_wastar] steps=25 4.65s`
- `15:14:38   running score over attempted: 100.00% (27 tasks)`
- `15:14:43 env/two_room-make_win-distr_rule ep0 seed=553072 -> SOLVED [sym_macro_wastar] steps=20 5.06s`
- `15:14:48 env/two_room-make_win-distr_rule ep1 seed=553073 -> SOLVED [sym_macro_wastar] steps=24 4.41s`
- `15:14:52 env/two_room-make_win-distr_rule ep2 seed=553074 -> SOLVED [sym_macro_wastar] steps=27 4.5s`
- `15:14:52   running score over attempted: 100.00% (28 tasks)`
- `15:14:53 env/two_room-make_win ep0 seed=553172 -> SOLVED [sym_bfs] steps=14 0.95s`
- `15:14:58 env/two_room-make_win ep1 seed=553173 -> SOLVED [sym_macro_wastar] steps=19 4.35s`
- `15:14:59 env/two_room-make_win ep2 seed=553174 -> SOLVED [sym_bfs] steps=14 1.8s`
- `15:14:59   running score over attempted: 100.00% (29 tasks)`
- `15:15:04 env/two_room-make_win-distr_obj-irrelevant_rule ep0 seed=553272 -> SOLVED [sym_macro_wastar] steps=21 4.23s`
- `15:15:09 env/two_room-make_win-distr_obj-irrelevant_rule ep1 seed=553273 -> SOLVED [sym_macro_wastar] steps=25 5.0s`
- `15:15:14 env/two_room-make_win-distr_obj-irrelevant_rule ep2 seed=553274 -> SOLVED [sym_macro_wastar] steps=26 4.88s`
- `15:15:14   running score over attempted: 100.00% (30 tasks)`
- `15:15:20 env/two_room-make_win-distr_obj ep0 seed=553372 -> SOLVED [sym_macro_wastar] steps=27 6.6s`
- `15:15:25 env/two_room-make_win-distr_obj ep1 seed=553373 -> SOLVED [sym_macro_wastar] steps=26 5.13s`
- `15:15:31 env/two_room-make_win-distr_obj ep2 seed=553374 -> SOLVED [sym_macro_wastar] steps=27 5.76s`
- `15:15:31   running score over attempted: 100.00% (31 tasks)`
- `15:15:36 env/two_room-make_win-distr_win_rule ep0 seed=553472 -> SOLVED [sym_macro_wastar] steps=17 5.35s`
- `15:15:41 env/two_room-make_win-distr_win_rule ep1 seed=553473 -> SOLVED [sym_macro_wastar] steps=22 4.6s`
- `15:15:45 env/two_room-make_win-distr_win_rule ep2 seed=553474 -> SOLVED [sym_bfs] steps=16 3.97s`
- `15:15:45   running score over attempted: 100.00% (32 tasks)`
- `15:15:51 env/two_room-break_stop-make_win-distr_obj_rule ep0 seed=553572 -> SOLVED [sym_macro_wastar] steps=21 5.76s`
- `15:15:56 env/two_room-break_stop-make_win-distr_obj_rule ep1 seed=553573 -> SOLVED [sym_macro_wastar] steps=22 5.04s`
- `15:16:03 env/two_room-break_stop-make_win-distr_obj_rule ep2 seed=553574 -> SOLVED [sym_macro_wastar] steps=28 6.93s`
- `15:16:03   running score over attempted: 100.00% (33 tasks)`
- `15:16:09 env/two_room-break_stop-make_win-distr_rule ep0 seed=553672 -> SOLVED [sym_macro_wastar] steps=21 5.91s`
- `15:16:14 env/two_room-break_stop-make_win-distr_rule ep1 seed=553673 -> SOLVED [sym_macro_wastar] steps=24 5.75s`
- `15:16:16 env/two_room-break_stop-make_win-distr_rule ep2 seed=553674 -> SOLVED [sym_bfs] steps=16 1.47s`
- `15:16:16   running score over attempted: 100.00% (34 tasks)`
- `15:16:24 env/two_room-break_stop-make_win ep0 seed=553772 -> SOLVED [sym_macro_wastar] steps=27 8.42s`
- `15:16:30 env/two_room-break_stop-make_win ep1 seed=553773 -> SOLVED [sym_macro_wastar] steps=24 5.92s`
- `15:16:36 env/two_room-break_stop-make_win ep2 seed=553774 -> SOLVED [sym_macro_wastar] steps=21 6.18s`
- `15:16:36   running score over attempted: 100.00% (35 tasks)`
- `15:16:42 env/two_room-break_stop-make_win-distr_obj-irrelevant_rule ep0 seed=553872 -> SOLVED [sym_macro_wastar] steps=23 5.47s`
- `15:16:47 env/two_room-break_stop-make_win-distr_obj-irrelevant_rule ep1 seed=553873 -> SOLVED [sym_macro_wastar] steps=19 4.73s`
- `15:16:52 env/two_room-break_stop-make_win-distr_obj-irrelevant_rule ep2 seed=553874 -> SOLVED [sym_macro_wastar] steps=22 5.17s`
- `15:16:52   running score over attempted: 100.00% (36 tasks)`
- `15:16:57 env/two_room-break_stop-make_win-distr_obj ep0 seed=553972 -> SOLVED [sym_macro_wastar] steps=24 5.55s`
- `15:17:03 env/two_room-break_stop-make_win-distr_obj ep1 seed=553973 -> SOLVED [sym_macro_wastar] steps=21 5.63s`
- `15:17:08 env/two_room-break_stop-make_win-distr_obj ep2 seed=553974 -> SOLVED [sym_macro_wastar] steps=22 5.51s`
- `15:17:08   running score over attempted: 100.00% (37 tasks)`
- `15:17:09 env/two_room-make_you ep0 seed=554072 -> SOLVED [sym_bfs] steps=10 0.08s`
- `15:17:09 env/two_room-make_you ep1 seed=554073 -> SOLVED [sym_bfs] steps=18 0.17s`
- `15:17:09 env/two_room-make_you ep2 seed=554074 -> SOLVED [sym_bfs] steps=11 0.1s`
- `15:17:09   running score over attempted: 100.00% (38 tasks)`
- `15:17:09 env/two_room-make_you-make_win ep0 seed=554172 -> SOLVED [sym_bfs] steps=25 0.41s`
- `15:17:09 env/two_room-make_you-make_win ep1 seed=554173 -> SOLVED [sym_bfs] steps=14 0.08s`
- `15:17:10 env/two_room-make_you-make_win ep2 seed=554174 -> SOLVED [sym_bfs] steps=21 0.3s`
- `15:17:10   running score over attempted: 100.00% (39 tasks)`
- `15:17:15 env/two_room-make_wall_win ep0 seed=554272 -> SOLVED [sym_macro_wastar] steps=14 5.03s`
- `15:17:16 env/two_room-make_wall_win ep1 seed=554273 -> SOLVED [sym_bfs] steps=14 1.17s`
- `15:17:22 env/two_room-make_wall_win ep2 seed=554274 -> SOLVED [sym_bfs] steps=14 6.35s`
- `15:17:22   running score over attempted: 100.00% (40 tasks)`
- `15:22 suite complete: 120/120 episodes, 40/40 tasks, FINAL SCORE 100.0% (SOTA 75.7%, +24.3pp). 0 mispredictions, 0 fallbacks. Robustness 10/10 on extra seeds. Final JSONs copied to results/ (fable_final_results.json, fable_summary.json, fable_model_validation.json).`
- `15:29:45 [clean] env/make_win-distr_obj_rule ep0 seed=550372 -> SOLVED [sym_bfs] steps=11 0.04s`
- `15:29:45 [clean] env/make_win-distr_obj_rule ep1 seed=550373 -> SOLVED [sym_bfs] steps=18 0.42s`
- `15:29:46 [clean] env/make_win-distr_obj_rule ep2 seed=550374 -> SOLVED [sym_bfs] steps=19 0.39s`
- `15:29:46 [clean] env/goto_win-distr_obj_rule ep0 seed=550472 -> SOLVED [sym_bfs] steps=4 0.0s`
- `15:29:46 [clean] env/goto_win-distr_obj_rule ep1 seed=550473 -> SOLVED [sym_bfs] steps=5 0.0s`
- `15:29:46 [clean] env/goto_win-distr_obj_rule ep2 seed=550474 -> SOLVED [sym_bfs] steps=4 0.0s`
- `15:29:46 [clean] env/goto_win ep0 seed=550572 -> SOLVED [sym_bfs] steps=3 0.0s`
- `15:29:46 [clean] env/goto_win ep1 seed=550573 -> SOLVED [sym_bfs] steps=3 0.0s`
- `15:29:46 [clean] env/goto_win ep2 seed=550574 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:29:46 [clean] env/goto_win-distr_obj ep0 seed=550672 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:29:46 [clean] env/goto_win-distr_obj ep1 seed=550673 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:29:46 [clean] env/goto_win-distr_obj ep2 seed=550674 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:29:46 [clean] env/goto_win-distr_rule ep0 seed=550772 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:29:46 [clean] env/goto_win-distr_rule ep1 seed=550773 -> SOLVED [sym_bfs] steps=6 0.0s`
- `15:29:46 [clean] env/goto_win-distr_rule ep2 seed=550774 -> SOLVED [sym_bfs] steps=3 0.0s`
- `15:29:46 [clean] env/goto_win-distr_obj-irrelevant_rule ep0 seed=550872 -> SOLVED [sym_bfs] steps=4 0.0s`
- `15:29:46 [clean] env/goto_win-distr_obj-irrelevant_rule ep1 seed=550873 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:29:46 [clean] env/goto_win-distr_obj-irrelevant_rule ep2 seed=550874 -> SOLVED [sym_bfs] steps=6 0.0s`
- `15:29:46 [clean] env/make_win-distr_obj ep0 seed=550972 -> SOLVED [sym_bfs] steps=14 0.27s`
- `15:29:46 [clean] env/make_win-distr_obj ep1 seed=550973 -> SOLVED [sym_bfs] steps=12 0.1s`
- `15:29:46 [clean] env/make_win-distr_obj ep2 seed=550974 -> SOLVED [sym_bfs] steps=13 0.16s`
- `15:29:47 [clean] env/make_win-distr_rule ep0 seed=551072 -> SOLVED [sym_bfs] steps=19 1.01s`
- `15:29:47 [clean] env/make_win-distr_rule ep1 seed=551073 -> SOLVED [sym_bfs] steps=9 0.03s`
- `15:29:48 [clean] env/make_win-distr_rule ep2 seed=551074 -> SOLVED [sym_bfs] steps=21 1.05s`
- `15:29:49 [clean] env/make_win ep0 seed=551172 -> SOLVED [sym_bfs] steps=19 0.32s`
- `15:29:49 [clean] env/make_win ep1 seed=551173 -> SOLVED [sym_bfs] steps=19 0.54s`
- `15:29:50 [clean] env/make_win ep2 seed=551174 -> SOLVED [sym_bfs] steps=11 0.17s`
- `15:29:50 [clean] env/make_win-distr_obj-irrelevant_rule ep0 seed=551272 -> SOLVED [sym_bfs] steps=19 0.45s`
- `15:29:51 [clean] env/make_win-distr_obj-irrelevant_rule ep1 seed=551273 -> SOLVED [sym_bfs] steps=15 0.64s`
- `15:29:51 [clean] env/make_win-distr_obj-irrelevant_rule ep2 seed=551274 -> SOLVED [sym_bfs] steps=14 0.11s`
- `15:29:51 [clean] env/two_room-goto_win ep0 seed=551372 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:29:51 [clean] env/two_room-goto_win ep1 seed=551373 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:29:51 [clean] env/two_room-goto_win ep2 seed=551374 -> SOLVED [sym_bfs] steps=4 0.01s`
- `15:29:51 [clean] env/two_room-goto_win-distr_obj_rule ep0 seed=551472 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:29:51 [clean] env/two_room-goto_win-distr_obj_rule ep1 seed=551473 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:29:51 [clean] env/two_room-goto_win-distr_obj_rule ep2 seed=551474 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:29:51 [clean] env/two_room-goto_win-distr_rule ep0 seed=551572 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:29:51 [clean] env/two_room-goto_win-distr_rule ep1 seed=551573 -> SOLVED [sym_bfs] steps=3 0.0s`
- `15:29:51 [clean] env/two_room-goto_win-distr_rule ep2 seed=551574 -> SOLVED [sym_bfs] steps=4 0.01s`
- `15:29:51 [clean] env/two_room-goto_win-distr_obj ep0 seed=551672 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:29:51 [clean] env/two_room-goto_win-distr_obj ep1 seed=551673 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:29:51 [clean] env/two_room-goto_win-distr_obj ep2 seed=551674 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:29:51 [clean] env/two_room-goto_win-distr_obj-irrelevant_rule ep0 seed=551772 -> SOLVED [sym_bfs] steps=8 0.04s`
- `15:29:51 [clean] env/two_room-goto_win-distr_obj-irrelevant_rule ep1 seed=551773 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:29:51 [clean] env/two_room-goto_win-distr_obj-irrelevant_rule ep2 seed=551774 -> SOLVED [sym_bfs] steps=4 0.01s`
- `15:29:51 [clean] env/two_room-goto_win-distr_win_rule ep0 seed=551872 -> SOLVED [sym_bfs] steps=6 0.02s`
- `15:29:51 [clean] env/two_room-goto_win-distr_win_rule ep1 seed=551873 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:29:51 [clean] env/two_room-goto_win-distr_win_rule ep2 seed=551874 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:29:51 [clean] env/two_room-break_stop-goto_win-distr_obj_rule ep0 seed=551972 -> SOLVED [sym_bfs] steps=11 0.12s`
- `15:29:51 [clean] env/two_room-break_stop-goto_win-distr_obj_rule ep1 seed=551973 -> SOLVED [sym_bfs] steps=11 0.12s`
- `15:29:51 [clean] env/two_room-break_stop-goto_win-distr_obj_rule ep2 seed=551974 -> SOLVED [sym_bfs] steps=9 0.02s`
- `15:29:51 [clean] env/two_room-break_stop-goto_win-distr_obj ep0 seed=552072 -> SOLVED [sym_bfs] steps=10 0.02s`
- `15:29:51 [clean] env/two_room-break_stop-goto_win-distr_obj ep1 seed=552073 -> SOLVED [sym_bfs] steps=7 0.03s`
- `15:29:51 [clean] env/two_room-break_stop-goto_win-distr_obj ep2 seed=552074 -> SOLVED [sym_bfs] steps=6 0.01s`
- `15:29:52 [clean] env/two_room-break_stop-goto_win-distr_rule ep0 seed=552172 -> SOLVED [sym_bfs] steps=14 0.42s`
- `15:29:52 [clean] env/two_room-break_stop-goto_win-distr_rule ep1 seed=552173 -> SOLVED [sym_bfs] steps=11 0.06s`
- `15:29:52 [clean] env/two_room-break_stop-goto_win-distr_rule ep2 seed=552174 -> SOLVED [sym_bfs] steps=11 0.04s`
- `15:29:52 [clean] env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule ep0 seed=552272 -> SOLVED [sym_bfs] steps=8 0.02s`
- `15:29:52 [clean] env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule ep1 seed=552273 -> SOLVED [sym_bfs] steps=8 0.02s`
- `15:29:52 [clean] env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule ep2 seed=552274 -> SOLVED [sym_bfs] steps=10 0.14s`
- `15:29:52 [clean] env/two_room-break_stop-goto_win ep0 seed=552372 -> SOLVED [sym_bfs] steps=9 0.09s`
- `15:29:52 [clean] env/two_room-break_stop-goto_win ep1 seed=552373 -> SOLVED [sym_bfs] steps=11 0.08s`
- `15:29:53 [clean] env/two_room-break_stop-goto_win ep2 seed=552374 -> SOLVED [sym_bfs] steps=14 0.47s`
- `15:29:53 [clean] env/two_room-maybe_break_stop-goto_win-distr_obj_rule ep0 seed=552472 -> SOLVED [sym_bfs] steps=11 0.29s`
- `15:29:53 [clean] env/two_room-maybe_break_stop-goto_win-distr_obj_rule ep1 seed=552473 -> SOLVED [sym_bfs] steps=12 0.48s`
- `15:29:53 [clean] env/two_room-maybe_break_stop-goto_win-distr_obj_rule ep2 seed=552474 -> SOLVED [sym_bfs] steps=9 0.04s`
- `15:29:53 [clean] env/two_room-maybe_break_stop-goto_win ep0 seed=552572 -> SOLVED [sym_bfs] steps=9 0.02s`
- `15:29:54 [clean] env/two_room-maybe_break_stop-goto_win ep1 seed=552573 -> SOLVED [sym_bfs] steps=13 0.52s`
- `15:29:55 [clean] env/two_room-maybe_break_stop-goto_win ep2 seed=552574 -> SOLVED [sym_bfs] steps=15 0.78s`
- `15:29:55 [clean] env/two_room-maybe_break_stop-goto_win-distr_obj ep0 seed=552672 -> SOLVED [sym_bfs] steps=13 0.27s`
- `15:29:55 [clean] env/two_room-maybe_break_stop-goto_win-distr_obj ep1 seed=552673 -> SOLVED [sym_bfs] steps=10 0.11s`
- `15:29:55 [clean] env/two_room-maybe_break_stop-goto_win-distr_obj ep2 seed=552674 -> SOLVED [sym_bfs] steps=4 0.01s`
- `15:29:55 [clean] env/two_room-maybe_break_stop-goto_win-distr_rule ep0 seed=552772 -> SOLVED [sym_bfs] steps=3 0.0s`
- `15:29:55 [clean] env/two_room-maybe_break_stop-goto_win-distr_rule ep1 seed=552773 -> SOLVED [sym_bfs] steps=2 0.0s`
- `15:29:55 [clean] env/two_room-maybe_break_stop-goto_win-distr_rule ep2 seed=552774 -> SOLVED [sym_bfs] steps=12 0.17s`
- `15:29:55 [clean] env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule ep0 seed=552872 -> SOLVED [sym_bfs] steps=1 0.0s`
- `15:29:55 [clean] env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule ep1 seed=552873 -> SOLVED [sym_bfs] steps=3 0.0s`
- `15:29:55 [clean] env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule ep2 seed=552874 -> SOLVED [sym_bfs] steps=3 0.01s`
- `15:30:00 [clean] env/two_room-make_win-distr_obj_rule ep0 seed=552972 -> SOLVED [sym_macro_wastar] steps=23 4.98s`
- `15:30:05 [clean] env/two_room-make_win-distr_obj_rule ep1 seed=552973 -> SOLVED [sym_macro_wastar] steps=22 4.97s`
- `15:30:10 [clean] env/two_room-make_win-distr_obj_rule ep2 seed=552974 -> SOLVED [sym_macro_wastar] steps=25 4.65s`
- `15:30:18 [clean] env/two_room-make_win-distr_rule ep0 seed=553072 -> SOLVED [sym_macro_wastar] steps=20 8.3s`
- `15:30:25 [clean] env/two_room-make_win-distr_rule ep1 seed=553073 -> SOLVED [sym_macro_wastar] steps=24 6.55s`
- `15:30:30 [clean] env/two_room-make_win-distr_rule ep2 seed=553074 -> SOLVED [sym_macro_wastar] steps=27 5.59s`
- `15:30:31 [clean] env/two_room-make_win ep0 seed=553172 -> SOLVED [sym_bfs] steps=14 0.88s`
- `15:30:36 [clean] env/two_room-make_win ep1 seed=553173 -> SOLVED [sym_macro_wastar] steps=19 4.66s`
- `15:30:38 [clean] env/two_room-make_win ep2 seed=553174 -> SOLVED [sym_bfs] steps=14 1.63s`
- `15:30:43 [clean] env/two_room-make_win-distr_obj-irrelevant_rule ep0 seed=553272 -> SOLVED [sym_macro_wastar] steps=21 5.28s`
- `15:30:49 [clean] env/two_room-make_win-distr_obj-irrelevant_rule ep1 seed=553273 -> SOLVED [sym_macro_wastar] steps=25 5.89s`
- `15:30:54 [clean] env/two_room-make_win-distr_obj-irrelevant_rule ep2 seed=553274 -> SOLVED [sym_macro_wastar] steps=26 5.42s`
- `15:30:59 [clean] env/two_room-make_win-distr_obj ep0 seed=553372 -> SOLVED [sym_macro_wastar] steps=27 4.94s`
- `15:31:04 [clean] env/two_room-make_win-distr_obj ep1 seed=553373 -> SOLVED [sym_macro_wastar] steps=26 4.61s`
- `15:31:09 [clean] env/two_room-make_win-distr_obj ep2 seed=553374 -> SOLVED [sym_macro_wastar] steps=27 5.26s`
- `15:31:13 [clean] env/two_room-make_win-distr_win_rule ep0 seed=553472 -> SOLVED [sym_macro_wastar] steps=17 4.48s`
- `15:31:19 [clean] env/two_room-make_win-distr_win_rule ep1 seed=553473 -> SOLVED [sym_macro_wastar] steps=22 5.39s`
- `15:31:24 [clean] env/two_room-make_win-distr_win_rule ep2 seed=553474 -> SOLVED [sym_bfs] steps=16 4.73s`
- `15:31:29 [clean] env/two_room-break_stop-make_win-distr_obj_rule ep0 seed=553572 -> SOLVED [sym_macro_wastar] steps=21 5.39s`
- `15:31:34 [clean] env/two_room-break_stop-make_win-distr_obj_rule ep1 seed=553573 -> SOLVED [sym_macro_wastar] steps=22 4.6s`
- `15:31:40 [clean] env/two_room-break_stop-make_win-distr_obj_rule ep2 seed=553574 -> SOLVED [sym_macro_wastar] steps=28 6.14s`
- `15:31:46 [clean] env/two_room-break_stop-make_win-distr_rule ep0 seed=553672 -> SOLVED [sym_macro_wastar] steps=21 6.06s`
- `15:31:51 [clean] env/two_room-break_stop-make_win-distr_rule ep1 seed=553673 -> SOLVED [sym_macro_wastar] steps=24 5.47s`
- `15:31:52 [clean] env/two_room-break_stop-make_win-distr_rule ep2 seed=553674 -> SOLVED [sym_bfs] steps=16 1.21s`
- `15:31:59 [clean] env/two_room-break_stop-make_win ep0 seed=553772 -> SOLVED [sym_macro_wastar] steps=27 6.65s`
- `15:32:04 [clean] env/two_room-break_stop-make_win ep1 seed=553773 -> SOLVED [sym_macro_wastar] steps=24 4.66s`
- `15:32:08 [clean] env/two_room-break_stop-make_win ep2 seed=553774 -> SOLVED [sym_macro_wastar] steps=21 4.66s`
- `15:32:13 [clean] env/two_room-break_stop-make_win-distr_obj-irrelevant_rule ep0 seed=553872 -> SOLVED [sym_macro_wastar] steps=23 4.4s`
- `15:32:18 [clean] env/two_room-break_stop-make_win-distr_obj-irrelevant_rule ep1 seed=553873 -> SOLVED [sym_macro_wastar] steps=19 4.92s`
- `15:32:24 [clean] env/two_room-break_stop-make_win-distr_obj-irrelevant_rule ep2 seed=553874 -> SOLVED [sym_macro_wastar] steps=22 6.39s`
- `15:32:29 [clean] env/two_room-break_stop-make_win-distr_obj ep0 seed=553972 -> SOLVED [sym_macro_wastar] steps=24 5.08s`
- `15:32:34 [clean] env/two_room-break_stop-make_win-distr_obj ep1 seed=553973 -> SOLVED [sym_macro_wastar] steps=21 4.86s`
- `15:32:39 [clean] env/two_room-break_stop-make_win-distr_obj ep2 seed=553974 -> SOLVED [sym_macro_wastar] steps=22 5.35s`
- `15:32:40 [clean] env/two_room-make_you ep0 seed=554072 -> SOLVED [sym_bfs] steps=10 0.06s`
- `15:32:40 [clean] env/two_room-make_you ep1 seed=554073 -> SOLVED [sym_bfs] steps=18 0.16s`
- `15:32:40 [clean] env/two_room-make_you ep2 seed=554074 -> SOLVED [sym_bfs] steps=11 0.1s`
- `15:32:40 [clean] env/two_room-make_you-make_win ep0 seed=554172 -> SOLVED [sym_bfs] steps=25 0.34s`
- `15:32:40 [clean] env/two_room-make_you-make_win ep1 seed=554173 -> SOLVED [sym_bfs] steps=14 0.09s`
- `15:32:41 [clean] env/two_room-make_you-make_win ep2 seed=554174 -> SOLVED [sym_bfs] steps=21 0.4s`
- `15:32:45 [clean] env/two_room-make_wall_win ep0 seed=554272 -> SOLVED [sym_macro_wastar] steps=14 4.5s`
- `15:32:46 [clean] env/two_room-make_wall_win ep1 seed=554273 -> SOLVED [sym_bfs] steps=14 1.28s`
- `15:32:51 [clean] env/two_room-make_wall_win ep2 seed=554274 -> SOLVED [sym_bfs] steps=14 4.91s`

## 6. Clean test-time protocol (operator follow-up: privileged-access ablation)

The 100% run above used three test-time channels beyond what the benchmark's obs exposes: (1) structured initial-state readout via env attribute access, (2) `verify_on_clone()` replaying each plan on a live env clone pre-execution, (3) an env-clone fallback search (never fired). A clean re-run (`clean_solver.py`, log `clean_run.log`) removed all three:

- **Initial state from `reset()` obs only** — the env's native `observation_space` output, a `(W,H,3)` uint8 array giving each cell's *top* object as `(type_idx, color_idx, 0)`. The parser (`parse_obs`) uses only the published encoding tables (`OBJECT_TO_IDX`/`COLOR_TO_IDX`/`name_mapping`) — the observation format spec, the array analogue of knowing what the words in a text observation mean. No `get_objects()`, no `agent_pos`, no ruleset readout, no `env.grid` access.
- **No clone verification, no fallback** — plans executed **open-loop** on the live env; the only env interaction per episode is `reset()` + the actual episode steps; score = the env's own win signal (`levels > 0`).
- **Same 120 seeds** as the privileged run, per-episode checkpointing, results in `results_fable/clean_protocol_results.json` (+ per-episode JSON under `results_fable/clean_protocol/babaisai/`).

### Result

| protocol | score | episodes | divergences vs privileged run |
|---|---|---|---|
| privileged (clone-verified, closed-loop safety nets) | 100.0% | 120/120 | — |
| **clean (obs-parse, open-loop, no verification)** | **100.0%** | **120/120** | **0** — identical action sequence on every episode |

Zero divergences is the expected-if-and-only-if-the-model-is-exact outcome, and it held: the clone-verification and fallback channels contributed nothing to the score; they were belt-and-suspenders. Wall-clock and node counts match the privileged run (the planner input was byte-identical on 119/120 episodes, see below).

### What the obs could and couldn't reconstruct

The obs encodes only the **top** object per cell (`encoding_level=1`). Offline audit (disclosed; not part of the test-time loop) comparing the obs-parse against privileged extraction on all 120 seeded instances:

- 480 differing cells are border **corners holding two stacked static `Wall` objects** (an artifact of `wall_rect` drawing overlapping edges). Behaviorally inert — static walls never move and can never be uncovered. The parser sees one wall; dynamics are identical.
- **1 gameplay-relevant occlusion in 120 instances**: in one `two_room-goto_win-distr_win_rule` episode (episode 1, seed 551873), the level generator stacked the distractor win-rule's `WIN` property block *on top of* the distractor `RuleObject(key)` at (11,4) (`put_rule` writes to fixed positions without an emptiness check). The obs cannot see the buried block. In principle a plan that pushed the `WIN` block off that cell would uncover state the model didn't know about and could diverge open-loop; in practice the episode is a goto task, the plan (identical to the privileged run's) never touches the cell, and it solved. **This is an irreducible observability gap of the benchmark's obs encoding, not of the method** — no obs-respecting agent can see under that block at episode start.

Not needed from the obs: agent direction (not encoded; dynamically irrelevant — verified in the fidelity sweep) and step count (0 at reset). `max_steps=100` and the 4-action interface are benchmark constants.

### The one remaining asterisk

The symbolic world model itself was synthesized by the frontier model **from reading the environment source code**, not induced from interaction traces or from observations alone. Runtime is fully LLM-free and, in this clean protocol, obs-only and open-loop — but "the model knew the rules because it read them" is a different (and weaker) claim than "the model learned the rules from play." The 0-disagreement fidelity sweep (8,433 lock-stepped steps) and the 0-divergence open-loop suite bound the *correctness* of the synthesized model; they do not change its *provenance*. A source-blind synthesis run (model induced purely from interaction transitions) is the natural next experiment and should be the paper's stated future work.
