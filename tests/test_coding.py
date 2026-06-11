"""Tests for the coding world."""

from openworld import Action
from openworld.coding import BENCHMARK, build_codefix_world, run_tests


def test_benchmark_buggy_sources_fail_and_references_pass():
    for task in BENCHMARK:
        buggy = run_tests(task.buggy_source, task.tests)
        assert buggy["failed"] > 0, f"{task.name}: buggy source unexpectedly passes"
        reference = run_tests(task.reference_source, task.tests)
        assert reference["failed"] == 0, (
            f"{task.name}: reference fails: {reference['errors']}"
        )


def test_run_tests_handles_broken_source():
    result = run_tests("def f(:\n  pass", [("f()", "None")])
    assert result["passed"] == 0
    assert result["failed"] == 1
    assert "failed to execute" in result["errors"][0]


def test_codefix_world_episode():
    task = BENCHMARK[0]  # sum_range_off_by_one
    world = build_codefix_world(task)
    assert world.state["solved"] is False
    assert world.state["tests_failed"] > 0

    # A wrong patch records feedback but does not solve.
    world.step(Action("submit_patch", params={"source": "def sum_to(n):\n    return 0\n"}))
    assert world.state["attempts"] == 1
    assert world.state["solved"] is False
    assert world.state["last_errors"]

    # The reference patch solves the task.
    world.step(Action("submit_patch", params={"source": task.reference_source}))
    assert world.state["solved"] is True
    assert world.state["tests_failed"] == 0

    # The world is inert once solved.
    world.step(Action("submit_patch", params={"source": "garbage"}))
    assert world.state["solved"] is True
    assert world.state["attempts"] == 2


def test_benchmark_has_twenty_distinct_tasks():
    assert len(BENCHMARK) == 20
    assert len({t.name for t in BENCHMARK}) == 20


def test_run_tests_times_out_on_infinite_loop():
    result = run_tests("def f():\n    while True:\n        pass\nf()\n",
                       [("1", "1")], timeout_seconds=0.5)
    assert result["failed"] == 1
    assert "timed out" in result["errors"][0]


def test_timeout_survives_exception_swallowing_code():
    # A patch that catches Exception inside its loop must not be able to
    # swallow the timeout alarm and spin forever.
    source = (
        "def f():\n"
        "    while True:\n"
        "        try:\n"
        "            x = 1\n"
        "        except Exception:\n"
        "            pass\n"
        "f()\n"
    )
    result = run_tests(source, [("1", "1")], timeout_seconds=0.5)
    assert result["failed"] == 1
    assert "timed out" in result["errors"][0]
