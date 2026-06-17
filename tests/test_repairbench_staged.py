"""Tests for OpenWorld-SWE-bench-STAGED (the two-stage repair dataset).

All offline, no Ollama. Validates the dataset contract (schema, oracle solves,
bug reality) exactly like the atomic set, PLUS the property that makes this
dataset worth shipping: each instance is genuinely *staged* — the stage-1
"obvious" patch (what a model writes from the issue text alone) repairs the
first failing test but NOT the second, while leaving the regression suite green.
That gap is what the in-world feedback loop is built to close.
"""

import copy
import importlib.util
from pathlib import Path

import pytest

from openworld.llm import MockLLM
from openworld.state import Action
from openworld.repairbench import (
    RepairBenchInstance,
    build_repairbench_world,
    load_dataset,
    run_instance_tests,
    solve_in_world,
    solve_single_shot,
)

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "datasets" / "openworld-repairbench-staged" / "tasks.jsonl"

INSTANCES = load_dataset(_DATA)

# Load the builder module to reach STAGE1_PATCHES (the stage-1-only fixes).
_spec = importlib.util.spec_from_file_location(
    "_staged_build",
    _HERE.parent / "datasets" / "openworld-repairbench-staged" / "build_tasks.py",
)
_bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bt)
STAGE1_PATCHES = _bt.STAGE1_PATCHES


def _slug(instance_id: str) -> str:
    # openworld-repairbench-staged-000-config-parser-staged -> config-parser-staged
    return instance_id.split("-", 4)[-1]


def test_dataset_loads_and_ids_unique():
    assert len(INSTANCES) >= 6
    ids = [i.instance_id for i in INSTANCES]
    assert len(ids) == len(set(ids))
    for inst in INSTANCES:
        assert inst.issue.strip()
        assert inst.buggy_source.strip()
        assert inst.reference_source.strip()
        # every staged instance has exactly the two stages encoded as f2p tests
        assert len(inst.fail_to_pass) >= 2
        assert inst.pass_to_pass
        assert inst.world.get("actions") == ["submit_patch"]


@pytest.mark.parametrize("inst", INSTANCES, ids=[i.instance_id for i in INSTANCES])
def test_oracle_passes_both_suites(inst: RepairBenchInstance):
    result = run_instance_tests(inst.reference_source, inst)
    assert result["fail_to_pass"]["failed"] == 0, result["fail_to_pass"]["errors"]
    assert result["pass_to_pass"]["failed"] == 0, result["pass_to_pass"]["errors"]
    assert result["solved"] is True


@pytest.mark.parametrize("inst", INSTANCES, ids=[i.instance_id for i in INSTANCES])
def test_bug_is_real(inst: RepairBenchInstance):
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


@pytest.mark.parametrize("inst", INSTANCES, ids=[i.instance_id for i in INSTANCES])
def test_staging_is_real(inst: RepairBenchInstance):
    """The stage-1 patch fixes test 1 but not test 2 — the loop has work to do.

    This is the property that distinguishes this dataset from the atomic set:
    the issue-only fix is a *partial* repair, so single-shot is expected to stall
    on stage 2 while in-world reads the stage-2 error and finishes.
    """
    s1 = STAGE1_PATCHES[_slug(inst.instance_id)]

    def first_suite_passes(src: str, pair) -> bool:
        probe = copy.copy(inst)
        probe.fail_to_pass = [pair]
        probe.pass_to_pass = []
        return run_instance_tests(src, probe)["fail_to_pass"]["failed"] == 0

    # stage-1 patch: passes the FIRST f2p test (the issue symptom)...
    assert first_suite_passes(s1, inst.fail_to_pass[0]), (
        f"{inst.instance_id}: stage-1 patch failed stage-1 test (issue not fixed)"
    )
    # ...but FAILS the SECOND f2p test (the latent, undescribed defect)...
    assert not first_suite_passes(s1, inst.fail_to_pass[1]), (
        f"{inst.instance_id}: stage-1 patch already passes stage 2 — not staged"
    )
    # ...and does NOT introduce a regression (pass_to_pass stays green).
    assert run_instance_tests(s1, inst)["pass_to_pass"]["failed"] == 0, (
        f"{inst.instance_id}: stage-1 patch broke a regression test"
    )


def test_world_flips_solved_on_reference():
    inst = INSTANCES[0]
    world = build_repairbench_world(inst)
    assert world.state["solved"] is False
    world.step(Action("submit_patch", params={"source": "def nope():\n    return 1\n"}, agent="t"))
    assert world.state["attempts"] == 1
    assert world.state["solved"] is False
    world.step(Action("submit_patch", params={"source": inst.reference_source}, agent="t"))
    assert world.state["solved"] is True
    attempts_before = world.state["attempts"]
    world.step(Action("submit_patch", params={"source": "broken"}, agent="t"))
    assert world.state["attempts"] == attempts_before


def test_in_world_recovers_in_two_steps():
    """Mirror the staged dynamic with a scripted model: stage-1 patch, then ref."""
    inst = INSTANCES[0]
    s1 = STAGE1_PATCHES[_slug(inst.instance_id)]
    fenced_s1 = f"```python\n{s1}```"
    fenced_ref = f"```python\n{inst.reference_source}```"

    # single-shot only ever sees the issue -> stage-1 patch -> NOT solved
    ss = solve_single_shot(inst, MockLLM(responses=[fenced_s1]))
    assert ss["solved"] is False

    # in-world: stage-1 patch fails, feedback drives the reference -> solved
    iw = solve_in_world(inst, MockLLM(responses=[fenced_s1, fenced_ref]), budget=4)
    assert iw["solved"] is True
    assert iw["solved_first_attempt"] is False
    assert iw["attempts"] == 2
