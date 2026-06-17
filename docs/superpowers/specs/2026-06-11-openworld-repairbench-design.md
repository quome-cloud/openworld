# OpenWorld-SWE-bench: design

**Date:** 2026-06-11
**Status:** approved

## Goal

A SWE-bench-style program-repair dataset where every instance carries an explicit
world-model representation, plus a harness that compares each model **single-shot**
(one patch, no feedback) against the **same model operating inside the world model**
(iterative patches with exact failing-test feedback). Side-by-side, paired, per-task.

## Why

The existing 20-task `openworld.coding.BENCHMARK` is saturated by capable local
models (e05: 7B/3B hit pass@1 = 100%). It tests single functions with all tests
visible as fail-to-pass. SWE-bench's difficulty comes from larger contexts, natural
language issue reports, and regression risk: a fix must repair the reported bug
*without breaking everything else*. This dataset reproduces those properties
offline and zero-dependency, and makes the world-model framing explicit so the
single-shot vs. in-world comparison is an apples-to-apples ablation of the world
model itself.

## Dataset

**Location:** `datasets/openworld-repairbench/`

- `tasks.jsonl` — 20 instances, one JSON object per line.
- `README.md` — dataset card: schema, design rationale, how to run the comparison,
  results table template.

**Instance schema:**

| field | type | meaning |
|---|---|---|
| `instance_id` | str | `openworld-repairbench-NNN-<slug>` |
| `module_name` | str | name of the buggy module (for prompts) |
| `issue` | str | GitHub-style natural-language bug report (the only problem statement the model sees) |
| `buggy_source` | str | full Python module, 30–80 lines, 2–5 functions or a stateful class |
| `reference_source` | str | known-good module (oracle; never shown to models) |
| `test_preamble` | str | hidden driver helpers, exec'd in the namespace after the submission (e.g. multi-step stateful scenarios) |
| `fail_to_pass` | [[expr, expected_repr]] | tests that fail on `buggy_source` and pass on `reference_source` |
| `pass_to_pass` | [[expr, expected_repr]] | regression tests that pass on both |
| `world` | object | world-model spec: `name`, `description`, `initial_state`, `actions` (`["submit_patch"]`), `rules`, `invariants` (strings, mirrored as executable checks in the package) |

**Content principles:**

- Fully offline and deterministic; tests run in the restricted sandbox
  (`_TEST_BUILTINS` + `math`), via the hardened fork+SIGKILL runner.
- Harder than the existing benchmark: cross-function bugs, stateful classes,
  bugs whose naive fix breaks a `pass_to_pass` test.
- Varied domains: e.g. rate limiter, LRU cache, text wrapping, bank ledger,
  graph traversal, tokenizer/parser, interval merging, inventory manager.
- `issue` is written as a user would write it (symptoms, repro), not as a spec
  of the fix.

## Package: `openworld/repairbench.py`

Follows the `openworld/coding.py` pattern.

- `RepairBenchInstance` dataclass mirroring the schema.
- `load_dataset(path=DEFAULT_PATH) -> List[RepairBenchInstance]` — reads the JSONL;
  default path resolves `datasets/openworld-repairbench/tasks.jsonl` relative to the
  repo root.
- `run_instance_tests(source, instance, timeout_seconds=5.0)` — execs the
  submission, then the `test_preamble`, then evaluates both suites. Returns
  `{"fail_to_pass": {"passed", "failed", "errors"}, "pass_to_pass": {...},
  "solved": bool}` where `solved` means zero failures in **both** suites.
  Reuses the fork+SIGKILL pattern from `coding.run_tests`.
- `RepairBenchTransition(Transition)` — exact dynamics: `submit_patch` runs
  `run_instance_tests`, updates state (`source`, per-suite pass/fail counts,
  `last_errors` capped at 3, `attempts`, `solved`). No-op once solved.
- `build_repairbench_world(instance) -> World` — instantiates the instance's
  `world` spec with a `RepairBenchTransition`.
- Episode runners:
  - `solve_single_shot(instance, llm) -> record` — one prompt (issue +
    buggy module), one completion, `extract_code`, one hidden-suite run.
  - `solve_in_world(instance, llm, budget=4) -> record` — iterative loop in the
    world; each prompt includes the issue, current source, per-suite pass/fail
    counts and `last_errors`. Stops on solve or budget.
  - Both share the same system prompt and base prompt text so the only
    difference is the feedback loop. Unparseable output = failed attempt.
  - Records: `{"instance_id", "solved", "solved_first_attempt", "attempts",
    "regression_failures_seen"}`.

## Runner: `datasets/openworld-repairbench/run_comparison.py`

CLI: `python run_comparison.py [model ...]`, default models
`qwen2.5:1.5b qwen2.5:3b qwen2.5:7b llama3.2` (the e05/e16/e19 ladder).
`--budget N` (default 4), `--mock` for an offline smoke run with `MockLLM`.

For each model × instance it runs both conditions (paired), then writes
`datasets/openworld-repairbench/results/comparison.json` and prints a markdown
table:

| model | single-shot pass@1 | in-world pass@1 | in-world pass@4 | Δ (pass@4 − SS) | mean attempts |

Wilson 95% CIs on all rates (same convention as `experiments/common.py`;
the helper is vendored into the runner so the dataset folder stays
self-contained for execution while logic stays importable for tests).
Per-task paired records are saved so paired significance tests can be run later.
Requires Ollama only for real models; checks model availability up front.

## Tests: `tests/test_repairbench.py`

All offline, no Ollama:

1. **Schema:** every instance loads, all required fields non-empty, IDs unique.
2. **Oracle:** `reference_source` passes both suites for every instance.
3. **Bug reality:** `buggy_source` fails **all** `fail_to_pass` and passes
   **all** `pass_to_pass` tests for every instance.
4. **World spec:** `build_repairbench_world` instantiates; submitting
   `reference_source` flips `solved`; submitting garbage increments `attempts`
   without solving; solved worlds ignore further steps.
5. **Harness:** `solve_single_shot` and `solve_in_world` end-to-end with
   `MockLLM` scripted to fail then succeed; verify paired record shapes and
   that in-world recovers on attempt 2.

## Error handling

- Model output with no extractable code block: counts as a failed attempt
  (in-world: the loop continues with prior state; single-shot: unsolved).
- Infinite loops / pathological patches: fork+SIGKILL timeout marks all tests
  failed with a timeout error message.
- Missing Ollama model: runner fails fast with the model name before any work.

## Out of scope

- Real SWE-bench instances, git/diff-format patches, multi-file repos.
- LLM-synthesized dynamics for these worlds (dynamics are exact by
  construction, as in `coding.py`).
- Judge-based selection (e06) — the harness records enough to add it later.
