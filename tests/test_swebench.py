"""Tests for OpenWorld-SWE-bench (openworld.swebench). All offline, no Ollama.

Validates the dataset (schema, oracle correctness, bug reality), the world
wrapper, and the single-shot / in-world harness with a scripted MockLLM.
"""

import pytest

from openworld.llm import MockLLM
from openworld.state import Action
from openworld.swebench import (
    SWEBenchInstance,
    build_swebench_world,
    load_dataset,
    run_instance_tests,
    solve_in_world,
    solve_single_shot,
)

INSTANCES = load_dataset()


def test_dataset_loads_and_ids_unique():
    assert len(INSTANCES) >= 6
    ids = [i.instance_id for i in INSTANCES]
    assert len(ids) == len(set(ids))
    for inst in INSTANCES:
        assert inst.issue.strip()
        assert inst.buggy_source.strip()
        assert inst.reference_source.strip()
        assert inst.fail_to_pass and inst.pass_to_pass
        assert inst.world.get("actions") == ["submit_patch"]


@pytest.mark.parametrize("inst", INSTANCES, ids=[i.instance_id for i in INSTANCES])
def test_oracle_passes_both_suites(inst: SWEBenchInstance):
    result = run_instance_tests(inst.reference_source, inst)
    assert result["fail_to_pass"]["failed"] == 0, result["fail_to_pass"]["errors"]
    assert result["pass_to_pass"]["failed"] == 0, result["pass_to_pass"]["errors"]
    assert result["solved"] is True


@pytest.mark.parametrize("inst", INSTANCES, ids=[i.instance_id for i in INSTANCES])
def test_bug_is_real(inst: SWEBenchInstance):
    # buggy source must FAIL every fail_to_pass and PASS every pass_to_pass.
    result = run_instance_tests(inst.buggy_source, inst)
    assert result["fail_to_pass"]["passed"] == 0, (
        f"{inst.instance_id}: buggy source unexpectedly passed a fail_to_pass test"
    )
    assert result["pass_to_pass"]["failed"] == 0, (
        f"{inst.instance_id}: buggy source broke a pass_to_pass test: "
        f"{result['pass_to_pass']['errors']}"
    )
    assert result["solved"] is False


def test_world_flips_solved_on_reference():
    inst = INSTANCES[0]
    world = build_swebench_world(inst)
    assert world.state["solved"] is False
    # garbage patch: attempts increments, not solved
    world.step(Action("submit_patch", params={"source": "def nope():\n    return 1\n"}, agent="t"))
    assert world.state["attempts"] == 1
    assert world.state["solved"] is False
    # reference patch: solved
    world.step(Action("submit_patch", params={"source": inst.reference_source}, agent="t"))
    assert world.state["solved"] is True
    # solved world ignores further steps
    attempts_before = world.state["attempts"]
    world.step(Action("submit_patch", params={"source": "broken"}, agent="t"))
    assert world.state["attempts"] == attempts_before


def test_single_shot_and_in_world_records():
    inst = INSTANCES[1]
    fenced = f"```python\n{inst.reference_source}```"
    ss = solve_single_shot(inst, MockLLM(responses=[fenced]))
    assert ss["condition"] == "single_shot" and ss["solved"] is True
    assert ss["attempts"] == 1

    # in-world: fail on attempt 1 (garbage), recover on attempt 2 (reference)
    garbage = "```python\ndef wrong():\n    return 0\n```"
    iw = solve_in_world(inst, MockLLM(responses=[garbage, fenced]), budget=4)
    assert iw["condition"] == "in_world" and iw["solved"] is True
    assert iw["solved_first_attempt"] is False
    assert iw["attempts"] == 2


def test_unparseable_output_is_a_failed_attempt():
    inst = INSTANCES[0]
    ss = solve_single_shot(inst, MockLLM(responses=["no code here, sorry"]))
    # extract_code returns the raw text; it won't define the class -> unsolved
    assert ss["solved"] is False
