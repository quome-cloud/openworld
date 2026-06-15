# Instance generator — scope (thesis-testing MVP)

**Date:** 2026-06-15
**Status:** scope (pre-plan)
**Parent design:** `docs/superpowers/specs/2026-06-11-dataset-factory-design.md` (the full factory).
This is a deliberately **smaller slice** of that spec — only what's needed to unblock the
distillation thesis. Most of the grand factory is YAGNI until the thesis is proven.

## Purpose (why this exists)

The whole OpenWorld effort is testing one claim — **the param-efficiency / verified-trace
distillation thesis**: *a small model (qwen 1.5b/3b) fine-tuned on verified (prompt→passing-patch)
traces harvested from a larger teacher will out-solve the same small model's base, single-shot.*
If true, you get teacher-level fixes at a fraction of the params/cost — the headline.

The generator is **not** the goal. It exists because, as of tonight (2026-06-15), the thesis is
**stuck for a data reason, not a model reason** — and the generator is the only remaining lever.

## What tonight proved (the constraint the generator must fix)

Cheap experiments ruled out both model-side knobs (see
`LRN-openworld-teacher-scaling-not-the-lever-2026-06-15`):

| lever tested | result |
|---|---|
| teacher 7B → 14b | distillation Δ stayed **+1/10**, not significant |
| student 1.5b → 3b | distillation Δ stayed **+1/10** (bigger base, same lift) |

Difficulty profile of the 35 existing instances: base solves 3, **learnable band (base-fail +
14b-solve) = 27**, both-fail 5. So the band is *rich* — yet the student still learns ~1. Both
genuine transfers across all runs landed on the **same** instance (`interval-merge`), implying
it's the lone *absorbable* one. Conclusion: the binding constraints are **(a) absorbability** —
the existing band bugs are algorithmically too hard for a 1.5b to learn from one trace — and
**(b) power** — n=10 heldout can't show significance.

## Design decisions

1. **Parametric generator first** (not `llm`, not `mined`). The spec lists parametric as "the only
   source with exact difficulty knobs." That's exactly what we need: dial difficulty *down* and
   produce *volume*. `make_instance(seed, difficulty)` families emit endless controlled variants
   of simple bugs (off-by-one, wrong comparator, swapped operands, missing clamp, etc.) where the
   fix is a 1–2 line obvious change. Absorbability is then controlled **by construction**, not
   hoped for. (`llm` stays a later option for variety; `mined`/QuixBugs for external validity.)

2. **Two-sided difficulty band in the gate** — this is the refinement tonight adds to the spec.
   The spec's gate v2 calibration is one-sided (reject if the base *solves* it → preserve
   headroom / upper bound). We also need a **lower bound on triviality** is not the issue — the
   real second bound is *absorbability*, which can't be measured on a single candidate without
   training. Parametric construction is how we control it: target the band
   **base-1.5b fails AND the fix is ≤2 lines / single edit-site**. Operationally the gate keeps a
   candidate iff: reference solves both suites; buggy fails f2p / passes p2p; **base 1.5b fails it
   single-shot** (headroom, existing calibration idea); **patch diff is ≤ N lines / 1 hunk**
   (absorbability proxy); leak check; AST-dedup across all datasets.

3. **Reuse the existing pipeline end-to-end.** Generated instances emit the **unchanged**
   `SWEBenchInstance` schema and pass through the **existing gate** (`build_tasks.py` validation +
   `run_instance_tests`), the **existing harvest** (`bench … run --log-traces`), and the
   **existing chain** (`format_traces` → `to_mlx_data` → LoRA → `eval_heldout`). No new training
   or eval infra. This is the laziest correct path and keeps results comparable to tonight's.

## MVP slice — build this, skip the rest

**Build:**
- `datasets/owsb-param-easy/build_tasks.py` (or a `gen_parametric.py`) with 3–5 simple bug
  families, each `make_instance(seed, difficulty)`, emitting `SWEBenchInstance` dicts.
- A thin **band gate** wrapper: run reference + buggy validation (exists) + base-1.5b single-shot
  calibration + ≤N-line-diff check; keep survivors. Target output: **~80–150 instances**, big
  enough for a heldout of 30–50 (real power) plus a training set.
- Run them through the existing harvest+chain; produce `heldout_param.json`.

**Skip (YAGNI until thesis proven):** the recipe-YAML standard, auto-`CARD.md`, N-stage staging
verification, `mined` adapters, `llm` generator, Wilson-CI result schema, contextbench/runner
unification. All live in the parent spec; pull them in only after the thesis shows lift.

## Success criterion (what makes this worth it)

On the generated easy band, with a heldout of ≥30:
- **Primary:** distilled 1.5b Δ over base is **clearly > +1 and McNemar-significant** (p<0.05).
  That confirms the thesis on absorbable data — the generator was the missing piece.
- **Negative result is also valuable:** if Δ stays flat even on easy, plentiful, gated instances,
  the thesis itself (not the data) is weak at this scale — a real finding that saves further spend.

## Open question for Anderson before planning

- **Easy-bug source:** pure-parametric families (fastest, fully controlled, my recommendation) vs.
  an `llm`-generated easy set (more natural variety, noisier, needs the leak check to earn its
  keep). Recommend parametric for the first pass; add llm only if variety turns out to matter.
