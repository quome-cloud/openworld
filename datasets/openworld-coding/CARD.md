# OpenWorld-Coding

A family of **verified coding worlds** — function-implementation tasks, each a world whose
**oracle is its test suite** — for the world-time-compute realism check on a *different* use
case than diagnosis (OpenWorld experiment E77; paper §"World-time compute").

## What it is

Each task is a tiny verified-code world: a function to implement (prompt = signature +
docstring) and a set of `assert`-based unit tests that define correctness. A solution is
"right" iff it passes **all** tests — the cleanest possible oracle (this is the
HumanEval/MBPP setup). The transferable skill is *coding*; held-out tasks measure
generalization.

## Provenance

Tasks are **LLM-authored** (Gemini 2.5 Flash) across 12 topics (strings, arrays, dicts,
math, recursion, sorting, parsing, matrices, intervals, stacks/queues, greedy, simple DP),
then **verified in a sandboxed subprocess** — the reference solution must pass its own tests
before the task is admitted (~58% of generated candidates passed verification and were
kept). This contrasts with the synthetic-parametric `openworld-diagnosis` family: here the
worlds are *authored by a model* (the "Claude-Code-style" realism check), and the oracle is
executable tests rather than a Bayes-optimal classifier.

## Contents (JSONL)

| File | Rows | Schema |
|---|---|---|
| `tasks.jsonl` | 219 | `{name, topic, prompt, solution, tests[]}` (all verified) |
| `sft_train.jsonl` | 164 | `{prompt, completion}` — `prompt` = instruction + signature/docstring; `completion` = reference solution |
| `test_tasks.jsonl` | 55 | `{id, prompt, tests[], kind}` — held-out tasks for pass@k |

Task-level (world-level) train/test split: the 55 test tasks are held out from fine-tuning.

## How to use

Fine-tune on `sft_train.jsonl`; evaluate **pass@1 / pass@k** on `test_tasks.jsonl` (run the
model's code against each task's `tests` in a sandbox). For real-benchmark transfer, also
evaluate on **HumanEval / MBPP** (fetched by `experiments/e77_gen.py`'s benchmark step;
adapters in `experiments/e77_eval.py`).

## Reproduce

```
python experiments/e77_gen.py     # author + verify tasks (needs GEMINI_API_KEY in .env)
python experiments/e77_data.py    # split + SFT
```
Generation uses an LLM, so the exact task set is not bit-reproducible (unlike the seeded
diagnosis family); the committed `tasks.jsonl` is the canonical set used in E77.

## Results (E77, paper §world-time compute)

`experiments/results/e77_coding.json`. Headline: world-time compute helps **in-domain
pass@k at every model size** (e.g. 7B pass@5 0.84→0.95) and transfers **positively to
HumanEval at pass@5** (7B 0.866→0.909) from just 164 worlds — though it *hurts* greedy
pass@1 on HumanEval. Consistent with E76's world-count law (more worlds → more gain), 164
is below the threshold where transfer becomes strong.

## License

Apache 2.0 (same as the OpenWorld repository). Tasks/tests are LLM-generated; treated as synthetic.
