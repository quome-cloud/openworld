# OpenWorld-SWE-bench-STAGED

A **two-stage** program-repair dataset, companion to
[`openworld-swebench`](../openworld-swebench/) (the atomic set). Same schema,
same `openworld.swebench` harness, same single-shot-vs-in-world ablation — built
to answer the one question the atomic set couldn't.

## Why this exists

The atomic set's 7b-ladder run found **no in-world lift**: single-shot pass@1 and
in-world pass@4 were equal at every model size (1.5b 0.17 / 3b 0.50 / 7b 1.00).
Root cause: each atomic bug is repaired by a **single edit fully described by the
issue text**. The model's first patch is either right or wrong; iterating with
failing-test feedback has nothing to add, so the feedback loop never does work
and the two conditions converge on attempt 1.

That's a property of the *instances*, not of the loop. To measure what the
in-world loop is for, the instances have to require **more than the issue text**.

## The design: two stages per instance

Every instance encodes two defects:

| | What | Visible from the issue? | Surfaces as |
|---|---|---|---|
| **Stage 1** | the symptom the user reports | **Yes** — the issue describes it | the 1st `fail_to_pass` test |
| **Stage 2** | a second, related defect, latent until stage 1 is fixed | **No** | the 2nd `fail_to_pass` test — only as a concrete failing-test error |

The "obvious" patch a model writes from the issue alone fixes **stage 1 and
passes the first test, but fails the second**, while keeping the regression
(`pass_to_pass`) suite green. (`tests/test_swebench_staged.py::test_staging_is_real`
asserts exactly this for all six instances, using the stage-1 patches in
`build_tasks.py::STAGE1_PATCHES`.)

**Predicted result:** single-shot solves stage 1 and stalls on stage 2; in-world
reads the stage-2 error in its feedback and finishes the repair → a measurable
**Δ = in-world − single-shot > 0**. That Δ is the artifact the paper needs to
show the in-world loop is load-bearing.

> Note the dataset contract is unchanged: `buggy_source` still fails **every**
> `fail_to_pass` test and passes **every** `pass_to_pass` test. The staging lives
> in the *model's* intermediate patches, which the harness exercises at run time —
> not in the buggy source.

## Instances (6)

| id | module | stage 1 (issue) | stage 2 (latent) |
|---|---|---|---|
| `…-000-config-parser-staged` | `config_parser` | skip blank lines | split on the **first** `=` only (values may contain `=`) |
| `…-001-discount-clamp-staged` | `discount` | floor a negative pct at 0 | cap pct at 100 (else price goes negative) |
| `…-002-histogram-staged` | `histogram` | `add()` increments, not overwrites | `count()` of an unseen key returns 0, not KeyError |
| `…-003-median-staged` | `median` | average the two middle elements (even N) | empty list returns `None`, not IndexError |
| `…-004-deep-get-staged` | `deep_get` | missing key returns `None` (not KeyError) | non-dict intermediate returns `None` (not TypeError) |
| `…-005-format-duration-staged` | `format_duration` | zero-pad seconds (`1:5` → `1:05`) | break out hours past 3600s (`H:MM:SS`) |

Four pure functions + one stateful class + one parser — same spread of shapes as
the atomic set, so cross-dataset comparison isn't confounded by task type.

## Run it

All operations go through the recipe (`recipes/owsb-staged-v1.json`):

```bash
python -m openworld.bench recipes/owsb-staged-v1.json run --mock   # offline smoke
python -m openworld.bench recipes/owsb-staged-v1.json run          # Ollama ladder
python -m openworld.bench recipes/owsb-staged-v1.json all --mock   # build+validate+run+card
```

Results land in `results/<model>.json` (frozen result schema v1, one file
per model); the dataset card is `CARD.md`.

To rebuild `tasks.jsonl` directly: `python datasets/openworld-swebench-staged/build_tasks.py`
(the recipe's `build` step calls this for you: `python -m openworld.bench recipes/owsb-staged-v1.json build`).

To validate the dataset: `pytest tests/test_swebench_staged.py`

Results (with per-task paired records for significance testing) land alongside
a printed markdown table — same format as the atomic set, so the two are read
side by side: **flat Δ on atomic, positive Δ here** is the story.

## Files

| File | Role |
|---|---|
| `build_tasks.py` | source of truth (`RAW` instances + `STAGE1_PATCHES`); writes `tasks.jsonl` |
| `tasks.jsonl` | generated artifact the harness loads |
| `recipes/owsb-staged-v1.json` | recipe — use `python -m openworld.bench recipes/owsb-staged-v1.json <cmd>` to build/run/card |
| `results/comparison.json` | original E29 ladder run cited by the paper — do not modify |
