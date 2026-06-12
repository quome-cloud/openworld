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
from openworld.swebench import (
    SWEBenchInstance,
    build_swebench_world,
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

INSTANCE = SWEBenchInstance(
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
    "name": f"swebench:{INSTANCE.instance_id}",
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

world = build_swebench_world(INSTANCE)
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


def factory():
    """Fresh scripted model per condition: the naive patch, then the truth."""
    if live is not None:
        return live
    return MockLLM([f"```python\n{NAIVE_PATCH}```",
                    f"```python\n{INSTANCE.reference_source}```"])


single = solve_single_shot(INSTANCE, factory())
in_world = solve_in_world(INSTANCE, factory(), budget=4)
rows = [{"instance_id": INSTANCE.instance_id,
         "single_shot": single, "in_world": in_world}]
label = model_name if live is not None else "mock-naive-then-oracle"
print("\n" + markdown_table([{"model": label, "budget": 4,
                              "summary": summarize(rows, budget=4)}]))
if live is None:
    assert not single["solved"], "single-shot saw only the issue; the naive patch fails"
    assert in_world["solved"] and in_world["attempts"] == 2, (
        "in-world read the regression error and recovered")

# ---------------------------------------------------------------------------
# 5. Recipes: how a dataset of these becomes reproducible.
# ---------------------------------------------------------------------------

recipe = load_recipe("recipes/owsb-atomic-v1.json")
frozen = recipe["artifacts"]["tasks_jsonl_sha256"]
print(f"\n[recipe] {recipe['dataset']['name']} {recipe['dataset']['version']}: "
      f"{recipe['dataset']['path'].name} pinned at sha256 {frozen[:12]}…, "
      f"ladder {', '.join(recipe['eval']['models'])}, budget {recipe['eval']['budget']}")
print("[recipe] full flow: python -m openworld.bench recipes/owsb-atomic-v1.json all --mock")
