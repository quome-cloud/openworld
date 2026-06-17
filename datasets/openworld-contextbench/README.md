# OpenWorld-ContextBench

In-context learning for program repair — the SWE-ContextBench analogue, built on
the openworld-repairbench harness. Each instance is a repair **task** plus a
`context_history`: related bugs already fixed on *different* modules that share the
same underlying **fix pattern**. The model must **transfer the pattern**, not copy
code.

**Ablation:** with-context vs. without-context. Does prepending the related solved
examples help the model fix a new module? This is a *different axis* from
openworld-repairbench's single-shot-vs-in-world loop — there the extra signal is
*test results* (iterative feedback); here it's *prior solved examples*
(in-context learning). Scoring is identical (`solved` = zero failures in both
hidden suites), reusing `openworld.repairbench.run_instance_tests`.

## Instances (v0, 3)

| id | task module | shared fix pattern | context example |
|---|---|---|---|
| 000 | scoreboard | cap a running value with `min()` | rate limiter refill cap |
| 001 | warehouse | reject an out-of-range update | bank overdraft rejection |
| 002 | stats | sort before indexing | interval-merge sort-first |

Each context example is a *different module* with the same bug family, so a model
that simply pattern-matches the fix should improve with context — while one that
can't generalize won't. Oracle solves both suites; each bug is real (fails every
`fail_to_pass`, passes every `pass_to_pass`).

## Schema

Each JSONL line: `{instance_id, task, context_history, pattern}` where `task` is a
full openworld-repairbench instance (`module_name, issue, buggy_source,
reference_source, test_preamble, fail_to_pass, pass_to_pass, world`) and
`context_history` is a list of `{module_name, issue, buggy_source,
reference_source}`. Only `task.{module_name,issue,buggy_source}` and the context
examples are shown to the model; the task's answer key stays hidden.

## Running

```bash
python datasets/openworld-contextbench/build_tasks.py        # regenerate tasks.jsonl
python datasets/openworld-contextbench/run_comparison.py --mock        # offline smoke
python datasets/openworld-contextbench/run_comparison.py qwen2.5:3b    # real model
```

Output:

| model | without-context pass@1 | with-context pass@1 | Δ (with − without) |
|---|---|---|---|

with 95% Wilson CIs and per-task paired records saved to `results/comparison.json`.

## Status

v0 — 3 instances, harness validated offline (7 tests). Stacks on the
openworld-repairbench package (`openworld/repairbench.py`); merge after that lands.
Scaling instances + the real model-ladder run are tracked next.
