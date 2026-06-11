"""Tests for the OpenWorld-SWE-bench dataset and harness."""

import json

from openworld.swebench import (
    SWEBenchInstance,
    initial_world_state,
    load_dataset,
    merged_errors,
    run_instance_tests,
)

# (Tasks 2-4 extend this import block as they add symbols.)

# A minimal class-based instance used by unit tests (not part of the dataset).
FIXTURE = SWEBenchInstance(
    instance_id="openworld-swebench-000-fixture",
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
        "name": "swebench:openworld-swebench-000-fixture",
        "description": "Program repair world for the counter module.",
        "initial_state": {
            "instance": "openworld-swebench-000-fixture",
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
