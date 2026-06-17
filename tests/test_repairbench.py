"""Tests for the OpenWorld-SWE-bench dataset and harness."""

import json

import pytest

from openworld import Action, MockLLM
from openworld.repairbench import (
    DEFAULT_DATASET_PATH,
    RepairBenchInstance,
    build_repairbench_world,
    initial_world_state,
    load_dataset,
    merged_errors,
    run_instance_tests,
    solve_in_world,
    solve_single_shot,
)

# A minimal class-based instance used by unit tests (not part of the dataset).
FIXTURE = RepairBenchInstance(
    instance_id="openworld-repairbench-000-fixture",
    module_name="counter",
    issue=(
        "Counter seems to double-count. After calling increment() once, "
        "value is 2 instead of 1."
    ),
    buggy_source=(
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.value = 0\n"
        "    def increment(self):\n"
        "        self.value += 2\n"
        "        return self.value\n"
    ),
    reference_source=(
        "class Counter:\n"
        "    def __init__(self):\n"
        "        self.value = 0\n"
        "    def increment(self):\n"
        "        self.value += 1\n"
        "        return self.value\n"
    ),
    test_preamble=(
        "def bump(n):\n"
        "    c = Counter()\n"
        "    for _ in range(n):\n"
        "        c.increment()\n"
        "    return c.value\n"
    ),
    fail_to_pass=[("bump(1)", "1"), ("bump(3)", "3")],
    pass_to_pass=[("Counter().value", "0")],
    world={
        "name": "repairbench:openworld-repairbench-000-fixture",
        "description": "Program repair world for the counter module.",
        "initial_state": {
            "instance": "openworld-repairbench-000-fixture",
            "source": "",  # filled below
            "fail_to_pass_passed": 0,
            "fail_to_pass_failed": 2,
            "pass_to_pass_passed": 1,
            "pass_to_pass_failed": 0,
            "last_errors": [],
            "attempts": 0,
            "solved": False,
        },
        "actions": ["submit_patch"],
        "rules": ["submit_patch runs both hidden suites bit-exactly."],
        "invariants": ["attempts never decreases"],
    },
)
FIXTURE.world["initial_state"]["source"] = FIXTURE.buggy_source


def test_run_instance_tests_supports_classes_and_preamble():
    buggy = run_instance_tests(FIXTURE.buggy_source, FIXTURE)
    assert buggy["fail_to_pass"]["failed"] == 2
    assert buggy["pass_to_pass"]["failed"] == 0
    assert buggy["solved"] is False

    fixed = run_instance_tests(FIXTURE.reference_source, FIXTURE)
    assert fixed["solved"] is True
    assert fixed["fail_to_pass"]["errors"] == []


def test_run_instance_tests_broken_source_fails_everything():
    result = run_instance_tests("class Counter(:\n  pass", FIXTURE)
    assert result["solved"] is False
    assert result["fail_to_pass"]["passed"] == 0
    assert "failed to execute" in result["fail_to_pass"]["errors"][0]


def test_initial_world_state_reflects_buggy_results():
    state = initial_world_state(FIXTURE)
    assert state["fail_to_pass_failed"] == 2
    assert state["pass_to_pass_failed"] == 0
    assert state["attempts"] == 0
    assert state["solved"] is False
    assert state["source"] == FIXTURE.buggy_source
    assert len(state["last_errors"]) == 2
    assert all(state["last_errors"])
    assert "bump" in state["last_errors"][0]


def test_load_dataset_round_trip(tmp_path):
    record = {
        "instance_id": FIXTURE.instance_id,
        "module_name": FIXTURE.module_name,
        "issue": FIXTURE.issue,
        "buggy_source": FIXTURE.buggy_source,
        "reference_source": FIXTURE.reference_source,
        "test_preamble": FIXTURE.test_preamble,
        "fail_to_pass": [list(t) for t in FIXTURE.fail_to_pass],
        "pass_to_pass": [list(t) for t in FIXTURE.pass_to_pass],
        "world": FIXTURE.world,
        "future_field": "ignored",
    }
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    loaded = load_dataset(path)
    assert len(loaded) == 1
    assert loaded[0].fail_to_pass == FIXTURE.fail_to_pass
    assert loaded[0].pass_to_pass == FIXTURE.pass_to_pass
    assert loaded[0].buggy_source == FIXTURE.buggy_source


def test_merged_errors_keeps_regression_visibility():
    result = {
        "fail_to_pass": {"errors": ["f1", "f2", "f3", "f4"]},
        "pass_to_pass": {"errors": ["r1"]},
    }
    assert merged_errors(result) == ["f1", "f2", "r1"]
    no_regressions = {
        "fail_to_pass": {"errors": ["f1", "f2", "f3", "f4"]},
        "pass_to_pass": {"errors": []},
    }
    assert merged_errors(no_regressions) == ["f1", "f2", "f3"]


def test_world_episode_garbage_then_reference():
    world = build_repairbench_world(FIXTURE)
    assert world.state["solved"] is False
    assert world.state["fail_to_pass_failed"] == 2

    world.step(Action("submit_patch", params={"source": "x = 1\n"}))
    assert world.state["attempts"] == 1
    assert world.state["solved"] is False
    assert world.state["last_errors"]

    world.step(Action("submit_patch", params={"source": FIXTURE.reference_source}))
    assert world.state["attempts"] == 2
    assert world.state["solved"] is True
    assert world.state["fail_to_pass_failed"] == 0
    assert world.state["pass_to_pass_failed"] == 0


def test_solved_world_ignores_further_steps():
    world = build_repairbench_world(FIXTURE)
    world.step(Action("submit_patch", params={"source": FIXTURE.reference_source}))
    assert world.state["solved"] is True
    world.step(Action("submit_patch", params={"source": "x = 1\n"}))
    assert world.state["attempts"] == 1  # unchanged
    assert world.state["solved"] is True


def _fenced(source):
    return f"```python\n{source}\n```"


def test_solve_single_shot_with_mock():
    llm = MockLLM([_fenced(FIXTURE.reference_source)])
    record = solve_single_shot(FIXTURE, llm)
    assert record == {
        "instance_id": FIXTURE.instance_id,
        "condition": "single_shot",
        "solved": True,
        "solved_first_attempt": True,
        "attempts": 1,
        "saw_regression": False,
    }
    # Single-shot prompt must contain the issue but no test feedback.
    prompt = llm.calls[0][-1]["content"]
    assert "double-count" in prompt
    assert "failing" not in prompt.lower()


def test_solve_in_world_recovers_on_second_attempt():
    llm = MockLLM(["no code at all", _fenced(FIXTURE.reference_source)])
    record = solve_in_world(FIXTURE, llm, budget=4)
    assert record["solved"] is True
    assert record["solved_first_attempt"] is False
    assert record["attempts"] == 2
    assert record["condition"] == "in_world"
    # The second prompt carries world feedback from the failed first attempt.
    second_prompt = llm.calls[1][-1]["content"]
    assert "failing" in second_prompt.lower()


def test_solve_in_world_exhausts_budget():
    llm = MockLLM(["still not code"])
    record = solve_in_world(FIXTURE, llm, budget=2)
    assert record["solved"] is False
    assert record["attempts"] == 2


@pytest.fixture(scope="module")
def dataset():
    assert DEFAULT_DATASET_PATH.exists(), "run datasets/openworld-repairbench/build_tasks.py"
    return load_dataset()


def test_dataset_has_20_unique_instances(dataset):
    ids = [i.instance_id for i in dataset]
    assert len(ids) == 20
    assert len(set(ids)) == 20


def test_dataset_schema(dataset):
    for inst in dataset:
        assert inst.instance_id.startswith("openworld-repairbench-")
        for fld in ("module_name", "issue", "buggy_source", "reference_source"):
            assert getattr(inst, fld).strip(), f"{inst.instance_id}: empty {fld}"
        assert len(inst.fail_to_pass) >= 2, inst.instance_id
        assert len(inst.pass_to_pass) >= 2, inst.instance_id
        for key in ("name", "description", "initial_state", "actions", "rules", "invariants"):
            assert key in inst.world, f"{inst.instance_id}: world missing {key}"
        assert inst.world["actions"] == ["submit_patch"]


def test_dataset_reference_passes_both_suites(dataset):
    for inst in dataset:
        result = run_instance_tests(inst.reference_source, inst)
        assert result["solved"], (
            f"{inst.instance_id}: reference fails: "
            f"{result['fail_to_pass']['errors'] + result['pass_to_pass']['errors']}"
        )


def test_dataset_buggy_fails_f2p_and_passes_p2p(dataset):
    for inst in dataset:
        result = run_instance_tests(inst.buggy_source, inst)
        assert result["fail_to_pass"]["passed"] == 0, (
            f"{inst.instance_id}: a fail_to_pass test passes on the buggy source"
        )
        assert result["pass_to_pass"]["failed"] == 0, (
            f"{inst.instance_id}: buggy source breaks a pass_to_pass test: "
            f"{result['pass_to_pass']['errors']}"
        )


def test_dataset_world_initial_state_matches_recomputation(dataset):
    for inst in dataset:
        recomputed = initial_world_state(inst)
        stored = inst.world["initial_state"]
        for key in (
            "fail_to_pass_passed", "fail_to_pass_failed",
            "pass_to_pass_passed", "pass_to_pass_failed",
            "attempts", "solved", "source",
        ):
            assert stored[key] == recomputed[key], f"{inst.instance_id}: {key} drifted"
