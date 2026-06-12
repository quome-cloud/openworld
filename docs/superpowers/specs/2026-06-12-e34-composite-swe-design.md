# E34 — The sprint world: composite allocation on openworld-swebench

**Date:** 2026-06-12
**Status:** approved

## Goal

Benchmark the composite-world structure on real software-engineering tasks:
a CompositeWorld whose 20 children are the owsb-atomic repair worlds, where
a global allocation policy decides which task receives each repair attempt.
Same model, same total budget as the standard protocol — the measured
quantity is what the composite's global view is worth.

## Design

- **World:** children = `build_swebench_world(inst)` for all 20 owsb-atomic
  v1 instances (children unmodified — the composition contract). Aggregators:
  `open_tasks` (unsolved count), `total_failing` (sum of failing tests).
  Actions route as `<instance_id>:submit_patch`.
- **Repair model:** qwen2.5:7b, temperature/seed from
  `recipes/owsb-atomic-v1.json`; prompts identical to the standard in-world
  condition (`openworld.swebench._feedback_prompt` / `_safe_ask`).
- **Conditions** (total budget B = 80 attempts each):
  1. `fixed` — the standard protocol: 4 attempts per task in isolation
     (control; also the first real-model numbers for owsb-atomic v1);
  2. `round_robin` — composite: cycle unsolved tasks, skipping solved ones
     (recycles stranded attempts);
  3. `greedy` — composite: next attempt to the unsolved task with the
     fewest failing tests (ties: fewest attempts spent, then dataset order).
- **Metrics:** solved@B per condition; solved-vs-budget curve (every
  attempt); per-attempt records (task, fail counts before/after); task
  switches. Wilson CIs on solved@B.
- **Honesty:** if allocation does not beat the fixed protocol, that is the
  published result. The fixed condition's per-task records are also saved in
  the frozen bench result schema spirit (per-instance paired-style rows).

## Deliverables

- `experiments/e34_composite_swe.py` (+ results JSON).
- Figure `paper/figs/sprint.png`: solved-vs-budget curves for the three
  conditions (house palette), in the regeneration pipeline.
- Paper: paragraph in the composition subsection + figure;
  `NumExperiments` 32 → 33.

## Out of scope

Context-switch penalties (switches are counted, not priced); the staged
dataset (atomic only); multi-model ladder (7b only this round).
