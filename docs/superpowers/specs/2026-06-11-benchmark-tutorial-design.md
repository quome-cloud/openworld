# Benchmark-dataset tutorial: design

**Date:** 2026-06-11
**Status:** approved

## Goal

A fifth tutorial teaching the new dataset machinery end to end: author a
program-repair instance with a world spec, validate it through the gate, run
the paired single-shot vs in-world ablation, and make it reproducible with a
recipe. Existing tutorials are untouched except for the README index.

## Deliverables

1. **`tutorials/benchmark_dataset.md`** (~120 lines, house voice — matches
   the other four guides) titled "Build Your Own Benchmark Dataset", sections:
   1. *An instance is a tiny world* — the `RepairBenchInstance` schema (issue,
      buggy/reference source, hidden `fail_to_pass` / `pass_to_pass` suites,
      world spec); solving requires zero failures in BOTH suites, so
      regressions count.
   2. *Author one instance* — a small stateful bug with a regression trap,
      issue written as a user report; the exact instance the script defines.
   3. *The gate is the trust layer* — `run_instance_tests` oracle and
      bug-reality checks; why a validated instance can be trusted regardless
      of who (or what) authored it.
   4. *The paired ablation* — `solve_single_shot` vs `solve_in_world`; the
      only difference is the feedback loop; one paragraph on the E28/E29
      finding (atomic Δ≈0, staged Δ grows with model scale) with a pointer
      to the paper and the staged dataset.
   5. *Recipes make it reproducible* — anatomy of
      `recipes/owrb-atomic-v1.json`, the
      `python -m openworld.bench <recipe> all --mock` flow, frozen artifact
      hashes, auto-generated cards, the three reproducibility tiers.
2. **`tutorials/benchmark_dataset.py`** (~140 lines, runnable offline):
   - defines the §2 instance inline (`RepairBenchInstance(...)`);
   - asserts the gate checks (reference solves both suites; buggy fails every
     `fail_to_pass`, passes every `pass_to_pass`) and prints what passed;
   - builds the world via `build_repairbench_world`, steps it once with a wrong
     patch and once with the reference to show exact dynamics;
   - runs both conditions — `MockLLM` scripted garbage-then-oracle by
     default, or a live `OllamaLLM` when a model name is passed as argv[1] —
     and prints a one-row markdown comparison table via
     `openworld.bench.markdown_table`-compatible formatting;
   - loads `recipes/owrb-atomic-v1.json` with `openworld.bench.load_recipe`
     and prints the pinned fields (dataset path, frozen sha256 prefix,
     ladder, budget), then points at the CLI for the full flow.
   - Exits non-zero on any assertion failure (the script is its own test).
3. **`tutorials/README.md`** — one new table row (Benchmarking domain,
   "instance schema, validation gate, paired single-shot vs in-world
   ablation, recipes") and one line in the commands block
   (`python tutorials/benchmark_dataset.py`).

## Constraints

- Offline by default; no new dependencies; no pytest additions (tutorials
  are not in the test suite, matching the existing four).
- Uses only public/canonical APIs: `openworld.repairbench`
  (`RepairBenchInstance`, `run_instance_tests`, `build_repairbench_world`,
  `solve_single_shot`, `solve_in_world`) and `openworld.bench`
  (`load_recipe`).
- The inline instance must satisfy the same invariants the gate enforces on
  shipped datasets (verified by the script's own assertions at runtime).

## Out of scope

Rewriting the four existing tutorials; generalizing the instance schema to
non-coding domains; adding the tutorial to pytest or CI.
