# Benchmarking: Build Your Own World-Model Dataset

> Script: [`benchmark_dataset.py`](benchmark_dataset.py) — runs offline;
> pass an Ollama model name to go live.

The other four tutorials build worlds to *simulate*. This one builds a world
to *measure*: a program-repair benchmark instance whose dynamics are exact
test execution, the validation gate that makes it trustworthy, and the paired
ablation — the same model single-shot versus operating inside the world —
that the OpenWorld-SWE-bench datasets and the paper's E28–E29 results use.
By the end you can author instances of your own and pin a whole dataset
behind a recipe.

## 1. An instance is a tiny world

A `SWEBenchInstance` packages one repair task:

| field | the model under test sees it? | what it is |
|---|---|---|
| `module_name`, `issue` | **yes** | a user-style bug report: symptoms + repro, never the fix |
| `buggy_source` | **yes** | the broken module |
| `reference_source` | no | the oracle solution |
| `test_preamble` | no | hidden driver helpers exec'd after the submission |
| `fail_to_pass` | no | `(expression, expected_repr)` tests that expose the bug |
| `pass_to_pass` | no | regression tests that already pass and must keep passing |
| `world` | no | the world spec: initial state, `submit_patch` action, rules, invariants |

The world's single action runs both hidden suites bit-exactly in a sandbox,
and `solved` requires **zero failures in both** — a patch that fixes the bug
while breaking a passing test does not count. That one rule is what separates
this from a unit-test kata: regressions are first-class.

## 2. Author one instance

The script defines a stateful wallet whose `spend()` happily overdraws:

```python
def spend(self, amount_cents):
    self.balance_cents -= amount_cents
    return True
```

The issue reads like a real report — symptoms, a repro, and one load-bearing
sentence of context, never the fix:

> Wallets are going negative. Repro: `Wallet(100)`, `spend(60)` twice — both
> calls return True and `balance_cents` ends at −20. The second spend should
> have been declined. Spending your exact remaining balance has always worked
> and must keep working.

The design move is in `pass_to_pass`:

```python
fail_to_pass=[
    ("run(100, [60, 60])", "[(True, 40), (False, 40)]"),
    ("run(50, [80])", "[(False, 50)]"),
    ("run(30, [10, 30])", "[(True, 20), (False, 20)]"),
],
pass_to_pass=[
    ("run(100, [40, 40])", "[(True, 60), (True, 20)]"),
    ("run(75, [75])", "[(True, 0)]"),  # exact balance: the trap
],
```

The "obvious" guard a model writes from the issue alone is
`if amount_cents >= self.balance_cents: return False` — it fixes every
overdraft test and silently declines spending your exact balance. The
regression suite is where that mistake gets caught.

## 3. The gate is the trust layer

Before an instance ships it must prove three things, and the script asserts
all of them:

```python
ref = run_instance_tests(INSTANCE.reference_source, INSTANCE)
assert ref["solved"]                                  # oracle solves both suites
buggy = run_instance_tests(INSTANCE.buggy_source, INSTANCE)
assert buggy["fail_to_pass"]["passed"] == 0           # every f2p test exposes the bug
assert buggy["pass_to_pass"]["failed"] == 0           # regressions intact on buggy
```

This gate is the reason the dataset factory can scale: once every instance is
machine-verified, it stops mattering whether a human, an LLM, a parametric
template, or a mined corpus authored it. `python -m openworld.bench
recipes/<r>.json validate` runs this same gate over an entire dataset.

## 4. The paired ablation

Step the world by hand and the trap fires exactly as designed — the naive
patch clears the bug-report tests and trips the regression, and the world
says so in its feedback:

```
[world] naive '>=' patch: bug tests pass, but regression broke:
        run(75, [75]) -> [(False, 75)], expected [(True, 0)]
```

The evaluation runs every instance through two conditions with the same
prompts: `solve_single_shot` (issue + module, one completion, no feedback)
and `solve_in_world` (iterative `submit_patch`, exact failing-test feedback,
a fixed budget). The only difference is the feedback loop, so the gap is the
measured value of the world model. Offline, the script scripts a model that
writes the naive patch first and the truth second:

```
| model                  | single-shot pass@1 | in-world pass@1 | in-world pass@4 | Δ (pass@4 − SS) | mean attempts |
| mock-naive-then-oracle | 0%                 | 0%              | 100%            | +100%           | 2.0           |
```

At benchmark scale this gap is the paper's E28–E29 story: on *atomic*
single-edit bugs the loop adds nothing at any model size (Δ ≈ 0), while on
*staged* bugs — a latent second defect that surfaces only as a failing test —
the loop pays and the payoff grows with capability: Δ = +0.13 at 3B,
+0.33 at 7B, flat at 1.5B. See the "When does the feedback loop pay off?"
section of the paper and `datasets/openworld-swebench-staged/`.

## 5. Recipes make it reproducible

A dataset of these instances ships behind a recipe
(`recipes/owsb-atomic-v1.json`) that pins everything a rerun needs: the
builder and its seed, the frozen `tasks.jsonl` sha256, the eval ladder,
budget, temperature, and sampling seed. One command drives the whole flow:

```bash
python -m openworld.bench recipes/owsb-atomic-v1.json build      # regenerate, verify hash
python -m openworld.bench recipes/owsb-atomic-v1.json validate   # the gate, dataset-wide
python -m openworld.bench recipes/owsb-atomic-v1.json run --mock # paired ablation
python -m openworld.bench recipes/owsb-atomic-v1.json card       # dataset card
python -m openworld.bench recipes/owsb-atomic-v1.json all --mock # everything
```

Results land one file per model in a frozen schema (per-instance paired
records included, so exact tests like McNemar stay possible), and `CARD.md`
documents provenance plus three reproducibility tiers: **Tier 0** — the mock
path runs in pytest on every commit; **Tier 1** — `build` regenerates the
artifact byte-identically; **Tier 2** — reruns at the same pinned Ollama
digests reproduce results within the stated Wilson intervals.

## Where next

Author five instances of your own, give them a `build_tasks.py` and a recipe,
and `bench all --mock` them. The dataset-factory design
(`docs/superpowers/specs/2026-06-11-dataset-factory-design.md`) describes
what arrives next on this foundation: LLM/parametric/mined generators behind
the same gate, N-stage verification, and difficulty calibration.
