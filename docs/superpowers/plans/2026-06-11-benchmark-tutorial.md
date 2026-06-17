# Benchmark-Dataset Tutorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fifth tutorial (guide + offline-runnable script) teaching the dataset machinery: author an instance, pass the gate, run the paired ablation, pin it with a recipe.

**Architecture:** Two files in `tutorials/` matching the existing guide+script house pattern, plus one README row. The script is its own test (asserts throughout, offline by default, live model via argv).

**Tech Stack:** stdlib + `openworld.repairbench` / `openworld.bench` public APIs. Spec: `docs/superpowers/specs/2026-06-11-benchmark-tutorial-design.md`.

---

### Task 1: The runnable script

**Files:**
- Create: `tutorials/benchmark_dataset.py`

- [ ] **Step 1: Write the script** with exactly this content:

```python
"""Build Your Own Benchmark Dataset - runnable companion.

Authors one program-repair instance with a world spec, pushes it through the
validation gate, demonstrates the regression trap inside the world, runs the
paired single-shot vs in-world ablation (MockLLM offline; pass an Ollama
model name to go live), and shows what a recipe pins.

    python tutorials/benchmark_dataset.py              # offline
    python tutorials/benchmark_dataset.py qwen2.5:7b   # live model
"""

import sys

from openworld import Action, MockLLM, OllamaLLM
from openworld.bench import load_recipe, markdown_table, summarize
from openworld.llm import OllamaConnectionError
from openworld.repairbench import (
    RepairBenchInstance,
    build_repairbench_world,
    initial_world_state,
    run_instance_tests,
    solve_in_world,
    solve_single_shot,
)

# ---------------------------------------------------------------------------
# 1. Author one instance: a stateful bug with a regression trap.
# ---------------------------------------------------------------------------

BUGGY = """\
class Wallet:
    \"\"\"A prepaid wallet; spend() must never overdraw the balance.\"\"\"

    def __init__(self, balance_cents):
        self.balance_cents = balance_cents

    def spend(self, amount_cents):
        self.balance_cents -= amount_cents
        return True
"""

REFERENCE = """\
class Wallet:
    \"\"\"A prepaid wallet; spend() must never overdraw the balance.\"\"\"

    def __init__(self, balance_cents):
        self.balance_cents = balance_cents

    def spend(self, amount_cents):
        if amount_cents > self.balance_cents:
            return False
        self.balance_cents -= amount_cents
        return True
"""

# The "obvious" patch a model writes from the issue alone. It fixes every
# overdraft test - and breaks the exact-balance regression test (>= vs >).
NAIVE_PATCH = REFERENCE.replace(
    "if amount_cents > self.balance_cents:",
    "if amount_cents >= self.balance_cents:",
)

INSTANCE = RepairBenchInstance(
    instance_id="tutorial-000-wallet-overdraft",
    module_name="wallet",
    issue=(
        "Wallets are going negative. Repro: Wallet(100), spend(60) twice -\n"
        "both calls return True and balance_cents ends at -20. The second\n"
        "spend should have been declined. Spending your exact remaining\n"
        "balance has always worked and must keep working."
    ),
    buggy_source=BUGGY,
    reference_source=REFERENCE,
    test_preamble=(
        "def run(balance, spends):\n"
        "    w = Wallet(balance)\n"
        "    return [(w.spend(a), w.balance_cents) for a in spends]\n"
    ),
    fail_to_pass=[
        ("run(100, [60, 60])", "[(True, 40), (False, 40)]"),
        ("run(50, [80])", "[(False, 50)]"),
        ("run(30, [10, 30])", "[(True, 20), (False, 20)]"),
    ],
    pass_to_pass=[
        ("run(100, [40, 40])", "[(True, 60), (True, 20)]"),
        ("run(75, [75])", "[(True, 0)]"),  # exact balance: the trap
    ],
    world={},
)

INSTANCE.world = {
    "name": f"repairbench:{INSTANCE.instance_id}",
    "description": (
        "Program repair as a world model for module 'wallet'. Submit a "
        "corrected module via submit_patch(params={'source': ...})."
    ),
    "initial_state": initial_world_state(INSTANCE),
    "actions": ["submit_patch"],
    "rules": [
        "submit_patch replaces the module and runs both hidden suites bit-exactly.",
        "Solved means zero failures in fail_to_pass AND pass_to_pass.",
        "Once solved, further actions are no-ops.",
    ],
    "invariants": ["attempts never decreases",
                   "solved implies zero failing tests in both suites"],
}

# ---------------------------------------------------------------------------
# 2. The gate: what every shipped instance must prove.
# ---------------------------------------------------------------------------

ref = run_instance_tests(INSTANCE.reference_source, INSTANCE)
assert ref["solved"], ref
buggy = run_instance_tests(INSTANCE.buggy_source, INSTANCE)
assert buggy["fail_to_pass"]["passed"] == 0, "every fail_to_pass test must expose the bug"
assert buggy["pass_to_pass"]["failed"] == 0, "the buggy module must not break regressions"
print("[gate] reference solves both suites; bug is real; regressions intact")

# ---------------------------------------------------------------------------
# 3. The world: exact dynamics, and why the naive fix fails.
# ---------------------------------------------------------------------------

world = build_repairbench_world(INSTANCE)
state = world.step(Action("submit_patch", params={"source": NAIVE_PATCH}))
assert state["fail_to_pass_failed"] == 0      # the naive patch fixes the symptom...
assert state["pass_to_pass_failed"] == 1      # ...and trips the regression trap
assert not state["solved"]
print(f"[world] naive '>=' patch: bug tests pass, but regression broke: "
      f"{state['last_errors'][0]}")
state = world.step(Action("submit_patch", params={"source": INSTANCE.reference_source}))
assert state["solved"] and state["attempts"] == 2
print("[world] reference patch solves on attempt 2; world freezes once solved")

# ---------------------------------------------------------------------------
# 4. The paired ablation: same model, only the feedback loop differs.
# ---------------------------------------------------------------------------

model_name = sys.argv[1] if len(sys.argv) > 1 else None
live = None
if model_name:
    live = OllamaLLM(model=model_name, temperature=0.2, options={"seed": 41})
    try:
        live.ask("Reply with OK.")
    except OllamaConnectionError as exc:
        print(f"[warn] {model_name} unavailable ({exc}); falling back to MockLLM")
        live = None

def factory(condition):
    """Fresh scripted model per condition: the naive patch, then the truth."""
    if live is not None:
        return live
    return MockLLM([f"```python\n{NAIVE_PATCH}```",
                    f"```python\n{INSTANCE.reference_source}```"])

single = solve_single_shot(INSTANCE, factory("single_shot"))
in_world = solve_in_world(INSTANCE, factory("in_world"), budget=4)
rows = [{"instance_id": INSTANCE.instance_id,
         "single_shot": single, "in_world": in_world}]
label = model_name or "mock-naive-then-oracle"
print("\n" + markdown_table([{"model": label, "budget": 4,
                              "summary": summarize(rows, budget=4)}]))
if live is None:
    assert not single["solved"], "single-shot saw only the issue; the naive patch fails"
    assert in_world["solved"] and in_world["attempts"] == 2, (
        "in-world read the regression error and recovered")

# ---------------------------------------------------------------------------
# 5. Recipes: how a dataset of these becomes reproducible.
# ---------------------------------------------------------------------------

recipe = load_recipe("recipes/owrb-atomic-v1.json")
frozen = recipe["artifacts"]["tasks_jsonl_sha256"]
print(f"\n[recipe] {recipe['dataset']['name']} {recipe['dataset']['version']}: "
      f"{recipe['dataset']['path'].name} pinned at sha256 {frozen[:12]}…, "
      f"ladder {', '.join(recipe['eval']['models'])}, budget {recipe['eval']['budget']}")
print("[recipe] full flow: python -m openworld.bench recipes/owrb-atomic-v1.json all --mock")
```

- [ ] **Step 2: Run it offline**

Run: `python tutorials/benchmark_dataset.py`
Expected output lines, in order: `[gate] …`, `[world] naive '>=' patch …`,
`[world] reference patch solves on attempt 2 …`, a one-row markdown table
(`| mock-naive-then-oracle | 0% | 0% | 100% | +100% | 2.0 |`), and the two
`[recipe]` lines. Exit code 0. If any assertion fires, fix the instance/test
expectations (do not weaken assertions).

- [ ] **Step 3: Commit**

```bash
git add tutorials/benchmark_dataset.py
git commit -m "Add benchmark-dataset tutorial script"
```

---

### Task 2: The guide

**Files:**
- Create: `tutorials/benchmark_dataset.md`

- [ ] **Step 1: Write the guide** (~120 lines, voice and structure matching `tutorials/software_engineering_sprint.md`: title, script callout line, intro paragraph, numbered sections with short code excerpts, closing "where to go next").

Title: `# Benchmarking: Build Your Own World-Model Dataset`. Callout: script link + "runs offline; pass an Ollama model name to go live." Sections (each excerpts the corresponding script code):

1. **An instance is a tiny world** — the `RepairBenchInstance` fields table (issue / buggy_source / reference_source / test_preamble / fail_to_pass / pass_to_pass / world); only `module_name`+`issue`+`buggy_source` are shown to a model under test; `solved` requires zero failures in BOTH suites, so a fix that breaks a passing test does not count.
2. **Author one instance** — the Wallet bug verbatim (issue text, both sources); call out the design move: `pass_to_pass` includes spending the *exact* balance, so the obvious `>=` guard is wrong. Issues describe symptoms and a repro, never the fix.
3. **The gate is the trust layer** — the three assertions from script §2 with one sentence each; the punchline from the factory spec: once the gate is the quality bar, instance provenance (human, LLM, template, mined corpus) stops mattering — `python -m openworld.bench <recipe> validate` runs this same gate over a whole dataset.
4. **The paired ablation** — script §3–4: the world rejecting the naive patch with the exact regression error, then `solve_single_shot` vs `solve_in_world`; state the isolation property (same prompts; only the feedback loop differs) and one paragraph on E28/E29: atomic suite Δ≈0 at every model size, staged suite Δ=+0.13 (3B) and +0.33 (7B) — feedback pays only on multi-stage bugs and only for models strong enough to use it; pointer to `paper/` §"When does the feedback loop pay off?" and `datasets/openworld-repairbench-staged/`.
5. **Recipes make it reproducible** — anatomy of `recipes/owrb-atomic-v1.json` (generator+seed, frozen tasks.jsonl sha256, eval ladder/budget/temperature/seed); the five CLI verbs; the three reproducibility tiers (Tier 0 mock-in-pytest, Tier 1 byte-identical rebuild, Tier 2 statistically-compatible reruns at pinned digests); cards (`CARD.md`) as committed provenance.

Closing: "Where next" — author 5 instances and a builder + recipe of your own; the dataset-factory spec (`docs/superpowers/specs/2026-06-11-dataset-factory-design.md`) for what's coming (generators, gate v2).

- [ ] **Step 2: Verify guide/script consistency**

Every code excerpt in the .md must appear in (or be a faithful abridgement of) `benchmark_dataset.py`; the expected table row in the guide must match the script's actual output. Re-run `python tutorials/benchmark_dataset.py` and compare.

- [ ] **Step 3: Commit**

```bash
git add tutorials/benchmark_dataset.md
git commit -m "Add benchmark-dataset tutorial guide"
```

---

### Task 3: README row + final verification

**Files:**
- Modify: `tutorials/README.md`

- [ ] **Step 1: Add the row and command.** In the table, after the software-engineering row:

```markdown
| [Benchmarking: build a dataset](benchmark_dataset.md) | Evaluation / benchmarks | The instance schema and validation gate, the paired single-shot vs in-world ablation, recipes and reproducibility tiers |
```

In the commands block add: `python tutorials/benchmark_dataset.py`.

- [ ] **Step 2: Final verification**

```bash
python tutorials/benchmark_dataset.py        # exit 0, expected output
python -m pytest tests/ -q                   # nothing broken (no test changes expected)
git status --short                           # only the three intended files
```

- [ ] **Step 3: Commit**

```bash
git add tutorials/README.md
git commit -m "tutorials: index the benchmark-dataset tutorial"
```
