# OpenWorld-SWE-bench

SWE-bench-style program repair where every instance carries an explicit
world-model spec. The dataset is designed for one comparison: run a model
single-shot against the same model operating inside the world model. System
prompt and base prompt are identical in both conditions; the only difference
is exact executable feedback from the world's dynamics — failing test names,
error messages, and pass/fail counts — returned after each patch submission.
20 offline, deterministic instances.

## Instance schema

| field | type | notes |
|---|---|---|
| `instance_id` | str | unique identifier, e.g. `openworld-repairbench-002-pagination-last-page` |
| `module_name` | str | Python module name the patch must replace |
| `issue` | str | the only problem statement models see — symptoms and repro, no fix named |
| `buggy_source` | str | the broken module given to the model |
| `reference_source` | str | oracle solution, never shown to the model |
| `test_preamble` | str | hidden driver helpers appended to the module before tests run |
| `fail_to_pass` | list[tuple[str, str]] | `(expression, expected_repr)` pairs that must flip from failing to passing |
| `pass_to_pass` | list[tuple[str, str]] | regression guards that must stay passing throughout |
| `world.name` | str | world identifier |
| `world.description` | str | plain-language description of the repair task |
| `world.initial_state` | dict | buggy module's test results at build time |
| `world.actions` | list[str] | `["submit_patch"]` |
| `world.rules` | list[str] | submission semantics and solve conditions |
| `world.invariants` | list[str] | structural guarantees on the state |

**Abridged example — pagination instance:**

```json
{
  "instance_id": "openworld-repairbench-002-pagination-last-page",
  "module_name": "pagination",
  "issue": "Items vanish from the last page of results.\nWith 7 items and per_page=3, page_count says 2 (should be 3) and\nthe 7th item is never returned by get_page. Asking for the page\nafter the last one raises IndexError instead of returning [].\n",
  "buggy_source": "def page_count(total, per_page):\n    return total // per_page\n...",
  "reference_source": "def page_count(total, per_page):\n    return (total + per_page - 1) // per_page\n...",
  "test_preamble": "",
  "fail_to_pass": [
    ["page_count(7, 3)", "3"],
    ["get_page([1, 2, 3, 4, 5, 6, 7], 3, 3)", "[7]"],
    ["get_page([1, 2], 2, 3)", "[]"],
    ["page_summary([1, 2, 3, 4, 5, 6, 7], 3)", "[3, 3, 1]"]
  ],
  "pass_to_pass": [
    ["page_count(6, 3)", "2"],
    ["get_page([1, 2, 3, 4], 1, 2)", "[1, 2]"],
    ["page_summary([1, 2, 3, 4], 2)", "[2, 2]"]
  ],
  "world": {
    "name": "repairbench:openworld-repairbench-002-pagination-last-page",
    "description": "Program repair as a world model for module 'pagination'. Submit a corrected module via submit_patch(params={'source': ...}).",
    "initial_state": {
      "instance": "openworld-repairbench-002-pagination-last-page",
      "fail_to_pass_passed": 0, "fail_to_pass_failed": 4,
      "pass_to_pass_passed": 3, "pass_to_pass_failed": 0,
      "last_errors": [
        "page_count(7, 3) -> 2, expected 3",
        "get_page([1, 2, 3, 4, 5, 6, 7], 3, 3) raised IndexError('list index out of range')",
        "get_page([1, 2], 2, 3) raised IndexError('list index out of range')"
      ],
      "attempts": 0, "solved": false, "source": "..."
    },
    "actions": ["submit_patch"],
    "rules": [
      "submit_patch(params={'source': ...}) replaces the module and runs both hidden suites bit-exactly in a sandbox.",
      "The instance is solved when zero tests fail in both the fail_to_pass and pass_to_pass suites.",
      "Once solved, further actions are no-ops.",
      "Every submit_patch increments attempts by exactly one."
    ],
    "invariants": [
      "attempts never decreases",
      "solved implies zero failing tests in both suites",
      "state always carries the most recently submitted source"
    ]
  }
}
```

## Design notes

**Fully offline and deterministic.** All 20 instances are self-contained
Python modules. No network calls, no external dependencies. The sandbox
whitelists a restricted set of builtins plus `math`; class definitions are
allowed (several instances are class-based), but the sandbox is a guard
against accidental misuse, not a security boundary. Tests run under
fork+SIGKILL timeouts enforced by `openworld.coding.run_tests`.

**Solving requires zero failures in both suites.** A patch must fix every
`fail_to_pass` test AND preserve every `pass_to_pass` test. Regressions
count against the solver — this is the key difficulty increase over the
`openworld.coding.BENCHMARK`, which has no regression suite.

**Issues are written as user reports.** Each issue describes observable
symptoms and a minimal repro. The fix is never named. Several instances are
designed so the naive local fix breaks a `pass_to_pass` test: the matrix
instance (`-015-matrix-multiply-index`) is the clearest example — its buggy
index expression `b[j][k]` produces correct results for symmetric matrices
(the smoke tests that shipped with the code), so the regression suite catches
the naive transpose-only patch.

**Build-time validation.** `build_tasks.py` verifies every instance before
writing `tasks.jsonl`: the reference source must solve both suites, the buggy
source must fail all `fail_to_pass` tests, and the buggy source must pass all
`pass_to_pass` tests. The world's `initial_state` is computed by actually
running the suites on the buggy source, so `last_errors` in the dataset
record reflects real test output.

## The two conditions

**Single-shot (condition A).** The model receives the issue and the buggy
module in one prompt and produces one completion. No feedback. Judged on both
hidden suites: pass@1.

**In-world (condition B).** The model iterates via `submit_patch` against the
world's exact dynamics. After each submission the state carries updated
pass/fail counts and up to three failing-test strings. The model receives this
feedback as part of its next prompt and may submit again up to a configurable
budget (default 4). Judged as pass@1 (solved on the first attempt) and
pass@budget (solved within the budget).

The information asymmetry is explicit and intentional: in-world attempt 1
already sees the initial failing-test feedback embedded in the world's initial
state — that is the world model's contribution. The comparison is designed to
measure how much that feedback channel is worth, holding the model and all
prompt text constant.

## How to run

All operations go through the recipe (`recipes/owrb-atomic-v1.json`):

```bash
python -m openworld.bench recipes/owrb-atomic-v1.json run --mock   # offline smoke
python -m openworld.bench recipes/owrb-atomic-v1.json run          # Ollama ladder
python -m openworld.bench recipes/owrb-atomic-v1.json all --mock   # build+validate+run+card
```

Results land in `results/<model>.json` (frozen result schema v1, one file
per model); the dataset card is `CARD.md`.

To rebuild `tasks.jsonl` directly: `python datasets/openworld-repairbench/build_tasks.py`
(the recipe's `build` step calls this for you: `python -m openworld.bench recipes/owrb-atomic-v1.json build`).

To validate the dataset: `python -m pytest tests/test_repairbench.py`

## Results

Fill from `results/comparison.json` after running the comparison.

| model | single-shot pass@1 | in-world pass@1 | in-world pass@4 | Δ (pass@4 − SS) | mean attempts |
|---|---|---|---|---|---|
| qwen2.5:1.5b | | | | | |
| qwen2.5:3b | | | | | |
| qwen2.5:7b | | | | | |
| llama3.2 | | | | | |
