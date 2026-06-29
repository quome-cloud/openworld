# E119 — SLM macro/procedure slot (design)

**Date:** 2026-06-28
**Branch:** `jenia/e119-slm-solver` (off `arc3-runner-fix`)
**Status:** design — **gated on a pre-experiment** (Phase 0 below). Do not build the macro
loop until Phase 0 passes (GO).

This spec merges two independently-developed designs. Structure (Phase 0 gate, post-stall
integration, driver modes, 3-arm measurement, reproducibility protocol) is shared. Three
decisions follow the brainstorm in this session and override the sibling draft where they differ:
**object-referential macro form** (not raw pixel coords), **selection signal decided by Phase 0**
(not pre-locked to the subgoal proxy), and **`sc25` as a confirmed wall** (not a build-blocking
"suspected bug").

## Goal

After the first real E119 run (`experiments/e119/RESULTS.md`), the SLM-vs-search delta is **0** on
the pilot, and step-(a) classification (`experiments/e119/PROGRESS.md`) of the 15 unsolved games
shows the bottleneck is **not** branching (only `bp35` is width-bound → pruner) but
**goal-as-procedure** (`g50t`, the deep-walls). This spec adds the design's primary-but-unimplemented
**macro slot**: the SLM proposes a short *action procedure* when blind search stalls, replay-verified
against the env. It targets the ~13 signal-bearing walls a pruner cannot help, and matches the main
ARC-3 paper's finding (E102/103/104: the win is goal-as-procedure, not goal-as-state).

## The hard constraint the data imposes (read first)

All 15 unsolved games show **zero reward observed** (`solved=False`, every game). So:

- Strict level-up banking has **nothing to commit** until the *first* reward is found.
- For a deep procedure-wall, that first reward may be many actions deep with no intermediate ticks.
  A 2–8-action macro cannot reach it in one shot, and with no intermediate level-ups it **cannot
  chain**. (`g50t` is the sharp case: blind BFS *frontier-exhausted* at 843 masked states / depth 60
  with zero reward — its wall looks like state-dedup/masking collapsing the path, not raw depth.
  Any "reward is ~N actions deep" figure is **unverified — confirm in Phase 0**.)
- Therefore the entire burden of directing the macro phase falls on the **selection signal** — and
  whether *any* candidate signal carries directional information is **unmeasured**.

This is the E89/E89b bootstrap wall restated: *we need one positive signal to aim at.* Hence Phase 0
is a gate, not a formality.

---

## Phase 0 — gating pre-experiment: does a selection signal carry directional information?

**Build NOTHING in the macro slot until this passes.**

**Question.** On the zero-reward procedure-walls, does any candidate selection signal *move search
where blind BFS does not*? We measure **both** signals brainstormed for the macro ranker:
- **subgoal-proxy** — pursue a satisfiable `reach/count/align` predicate (the existing best-first path);
- **novelty** — seed search from states blind BFS has not reached (the brainstorm's chosen default).

If neither carries signal, the macro slot would propose blind → predicted negative, and we report
that boundary instead of building.

**Headroom set (Phase 0 + experiment):** `g50t` (primary, falsifiable) + `tr87`, `re86`, `sb26`,
`cn04` (high-novelty deep-walls). **Exclude** `bp35` (pruner, not macro) and `sc25` (confirmed wall —
below).

**`sc25` — confirmed wall, NOT a precondition.** Empirically ruled out as a perception bug: on the
**raw (unmasked)** frame, all 4 directionals + all 18 inferred targets + a full-board 8×8 click grid
(~100 cells) produce **0** frame change. So it is neither over-masking (raw frames are identical) nor
`click_candidates` coverage (a full-board sweep would have hit real targets). It is inert to every
available single action → excluded as a zero-signal wall. *Residual caveat:* the harness `_g(o)` reads
only the last frame-layer, so a layered effect could be invisible — flag as a **separate harness
investigation**, do not block the macro work on it.

**Procedure (deterministic, no macro code, no new infra):**

1. For each headroom game, run the existing bounded explorer (`probe` + a BFS rollout to the solver's
   budget) and collect observed frames `F` and distinct masked states.
2. **Satisfiability scan.** Enumerate the predicate DSL (`reach(c)`, `count(c,op,k)`, `align(a,b)`)
   over colors/objects actually present in `F`. Record `n_satisfiable` per game.
3. **Directionality test.** Against the blind-BFS control at a *matched* node/depth budget:
   - *subgoal-proxy:* for each satisfiable predicate `p`, run `search_level` best-first with
     `score_fn = 1.0 if p(frame) else 0.0`; record `depth_gain(p)` = max depth(guided) − max
     depth(blind), and `subgoal_novel_gain(p)` = states reached only under guidance.
   - *novelty:* record blind BFS's reachable-state count and whether it *frontier-exhausts* within
     budget (if it exhausts, novelty has no headroom — there are no unseen states to seed from).
4. Save to `experiments/results/e119_proxy_probe.json` **before** any assert (CLAUDE.md).

**Go / No-Go gate:**

- **GO** (build the macro loop) iff, on `g50t`, **either** signal is non-flat:
  `n_satisfiable ≥ 1 AND max_p depth_gain(p) ≥ +2` (or `subgoal_novel_gain ≥ 10%` of blind's states),
  **or** blind BFS does **not** frontier-exhaust (novelty headroom exists). Apply the same test to the
  four deep-walls to size the experiment and to pick the ranker (whichever signal is non-flat;
  default to **novelty** if both qualify, per the brainstorm).
- **NO-GO** (stop, report negative) if on `g50t` no predicate is ever satisfiable / guided ≈ blind on
  every predicate **and** blind BFS frontier-exhausts (no novelty headroom). The *finding* — "selection
  signals are flat on procedure-walls; first-reward capture needs a richer target space than the
  `reach/count/align` DSL or pure novelty" — is itself a publishable boundary result.

**Honest caveat on the gate.** Depth/novelty gain is a *necessary-not-sufficient* proxy for "closer to
the win" (no reward exists to measure true progress). Passing means a signal isn't flat; it does not
promise a solve. Failing means a solve is very unlikely — the gate's job is to kill a doomed build,
not to predict success.

---

## The macro slot (build only on GO)

### Decisions (this session's brainstorm)
- **Integration:** post-stall fallback (not super-actions in the frontier).
- **Macro form:** **object-referential ops** compiled to primitives — e.g. `["up","up","click #3",
  "left"]`. The compiler resolves object refs to sprite centroids (via `perceive.object_json`) and
  directions to `avail` action codes; `op x k` = repetition. This grounds the small model in the
  relational scene it already perceives and **avoids blind pixel-coordinate guessing** (the failure
  mode behind no-op clicks). Raw primitive `(a,)`/`(6,x,y)` is a fallback only.
- **Grading:** strict level-up **banks** (the only "solve"); the **selection signal** (which macros
  get real-env steps / which endpoints seed continued search) is **whichever Phase 0 validated** —
  default **novelty** (unseen masked state), subgoal-proxy if Phase 0 shows it is the stronger signal.
- **Macro length:** 2–8 ops (revisit the cap only if Phase 0 shows the primary win needs more).

### Hook point (`experiments/e119/solve.py`)
Per-level loop unchanged until search stalls. **Stall** = `planner.search_level(...)` returns `None`
(budget spent, no level-up). On stall:

```
1. metas/frames already collected this iteration (probe).
2. macros = macro.propose_macros(llm, obj_json, diffs, facts, stall_ctx, k=2..8, n=N, tau=τ)
     - relational prompt only (object-JSON + contrastive diffs + facts ledger + stall context);
       NEVER the raw grid.
     - object-referential ops -> compiled to primitive tuples (unresolvable ref -> drop op;
       empty/invalid macro -> discard).
     - behavioral best-of-N + abstention: cluster macros by OBSERVED effect
       (replayed endpoint masked-state + levels delta), not text; abstain if none agree.
3. rank surviving macros by the Phase-0-validated selection signal
     (novelty: endpoint masked-state unseen; and/or subgoal-proxy: endpoint satisfies a subgoal).
4. for each ranked macro: replay from the verified prefix on a FRESH env (arc3_harness.replay
       pattern, per Bug #2 fix):
         - levels increases  -> BANK: actions += macro; resume search from the new prefix.
         - else novel endpoint -> seed search from it (frontier jump), continue.
         - else                -> discard.
5. if no macro raises levels and none opens novelty -> honest stop (record attempts in the log).
```

Invariant preserved: **the env decides correctness.** A macro is replay-verified before banking; a
wrong/blind macro costs real-env steps, never a false solve. Macros only ever *add* reachable states,
so `macro-search ⊒ control`. Reuses `_PrefixGame` chaining, so no single macro need reach the full win.

### Modes (driver `e119_slm_solver.py`)
Add `--mode macro` (search + macro fallback, no subgoal ordering) and `--mode macro+slm` (adds subgoal
best-first ordering inside `search_level`). Keep `search` and `slm` as is. `llm` is constructed for any
mode ≠ `search`.

### New code surface (minimal)
- `experiments/e119/macro.py` — `propose_macros(...)` (proposer + object-referential op grammar),
  the op→primitive compiler, the behavioral grader, abstention. Mirrors `slm.propose_subgoal`.
- `experiments/e119/solve.py` — the stall-hook block above (≈20 lines), behind the mode flag.
- `tests/test_e119_macro.py` — MockLLM + an in-memory `MacroGame` (a short win no blind BFS finds
  within a tight budget but a banked macro does); compiler tests (op resolution, unresolvable→discard);
  grader branch tests (bank / novelty-seed / discard); and the safety test (an unverifiable macro never
  banks; abstain → no bank). numpy/MockLLM only — no env/Ollama.

### Measurement (headroom set only — solved games show no lift by construction)
Three arms at matched node/depth budget:

| arm | what |
|---|---|
| `search` | blind BFS (control) |
| `random-macro` | inject random 2–8-op macros on stall (matched count/length, seeded) |
| `macro` (SLM) | SLM-proposed macros on stall |

- **Model contribution** = `macro` − `random-macro` (isolates the SLM from "just try short seqs").
- **Mechanism value** = either − `search`.
- Run across the four SLMs (qwen2.5-coder:7b, qwen2.5:7b, gemma3, llama3.1:8b) for the model-diversity
  corollary, **per the reproducibility protocol below** (seeds + variance).

### Success criterion (falsifiable)
The macro slot solves **≥ 1 procedure-wall** (replay-verified, `levels ≥ 1`) that blind search at
*matched* budget cannot — ideally `g50t`. **A clean negative is an accepted, reportable result** (it
sharpens the goal-as-procedure boundary for SLMs).

## Reproducibility

Determinism splits cleanly by layer. State results accordingly — do not report a stochastic number as
if it were fixed.

**Deterministic & exactly reproducible (report as point facts):**
- **Env dynamics** — ARC-AGI-3 is replay-deterministic (repo determinism = 1.00); replay-verify is
  exact, 0% noise.
- **Control arm, reachability, classification, and all of Phase 0** — env + fixed deterministic
  perception (`status_mask`, `click_candidates`, search order), **no LLM**. Same inputs → same outputs.
- **Every banked solution** — a fixed action sequence that replays to the same `levels` on any machine.
  The canonical, auditable artifact; the success criterion is stated against it.

**NOT bit-for-bit reproducible (report as distributions): the SLM arm.** Two sources: (1) sampling
(temperature 0.6–1.0 / top_p / top_k); (2) Metal nondeterminism — per CLAUDE.md, Ollama on Metal is not
fully deterministic even with a fixed seed. Strict replay-banking contains this: a banked solve is
genuine regardless of how it was reached, so result *validity* is deterministically checkable even when
the *trajectory* is not. "A replay-verified g50t solution exists" is permanent; "this run's SLM arm
solves g50t" is a draw from a distribution.

**Protocol (bake into the runner before reporting numbers):**
- **Pin** the exact Ollama model *digest* (not just the tag), `num_ctx`, `num_predict`, seed, and
  decoding params; record them in the results JSON `env` block.
- **Repeat each SLM arm over m seeds (m ≥ 5)**; report **mean ± variance** and **k/m games solved**,
  never a single run (matches the E62 stats stance and "keep the best, verified by reproduction").
- **Log every prompt + raw completion** per call in the `arc3_traces` schema (prompt, transcript,
  model/version, token usage, verified outcome) for auditability.
- **Fix all deterministic knobs** shared with control: budget (`max_nodes`/`max_depth`), masking
  threshold, candidate `max_size`, RNG seed for `random-macro`.
- **Pin the env**: arc-agi / arcengine versions and Python (≥3.12) in the results `env` block.

**Bottom line:** the *claims* (a solve exists; the control numbers; the Phase 0 gate) are fully
reproducible; the *SLM-arm magnitudes* (delta, levels, which macro won) are reproducible only in
distribution — report them with seeds and variance, anchored to replayable action sequences.

## Risks / honest priors
- Procedure-walls defeated Claude-grade goal-discovery *with perfect world models* (E102/3/4); a local
  SLM cracking many of the ~13 is unlikely. Realistic target: 0–1 games + a clean boundary.
- Zero reward everywhere ⇒ success hinges on a selection signal having directionality (Phase 0). If
  Phase 0 is NO-GO, do not build.
- `random-macro` may occasionally get lucky on a shallow procedure — that is why it is a baseline, not
  noise to hide.
- `g50t` frontier-exhausts within budget; if novelty has no headroom there and the subgoal proxy is
  flat, it is a clean No-Go regardless of macro quality.

## Sequencing
1. (Separate) flag `sc25` harness-layer question; it is **not** a blocker here.
2. **Phase 0** probe → `experiments/results/e119_proxy_probe.json` + GO/No-Go.
3. (GO only) build `macro.py` (`propose_macros` + compiler + grader) + the `solve.py` stall hook + tests.
4. 3-arm sweep on the headroom set, per the reproducibility protocol (seeds + variance).
5. Write up (`RESULTS.md` / `PROGRESS.md`): delta, attribution, and the boundary — positive or negative.
