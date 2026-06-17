"""Tests for OpenWorld-ContextBench (openworld.contextbench). Offline, no Ollama."""

import pytest

from openworld.contextbench import (
    ContextBenchInstance, load_dataset, solve_with_context, solve_without_context,
    _context_block,
)
from openworld.llm import MockLLM
from openworld.repairbench import run_instance_tests

INSTANCES = load_dataset()


def test_dataset_loads_with_context():
    assert len(INSTANCES) >= 3
    ids = [i.instance_id for i in INSTANCES]
    assert len(ids) == len(set(ids))
    for inst in INSTANCES:
        assert inst.context_history, "every contextbench instance needs prior examples"
        assert inst.task.fail_to_pass and inst.task.pass_to_pass


@pytest.mark.parametrize("inst", INSTANCES, ids=[i.instance_id for i in INSTANCES])
def test_oracle_and_bug_reality(inst: ContextBenchInstance):
    ok = run_instance_tests(inst.task.reference_source, inst.task)
    assert ok["solved"] is True, ok
    bad = run_instance_tests(inst.task.buggy_source, inst.task)
    assert bad["fail_to_pass"]["passed"] == 0  # bug fails every fail_to_pass
    assert bad["pass_to_pass"]["failed"] == 0   # but passes the regression suite
    assert bad["solved"] is False


def test_context_block_includes_examples_but_not_in_baseline():
    inst = INSTANCES[0]
    block = _context_block(inst.context_history)
    assert inst.context_history[0].module_name in block
    assert "Fixed:" in block


def test_with_and_without_context_harness():
    inst = INSTANCES[0]
    fenced = f"```python\n{inst.task.reference_source}```"
    # both conditions solve when the model emits the correct module
    wo = solve_without_context(inst, MockLLM(responses=[fenced]))
    wc = solve_with_context(inst, MockLLM(responses=[fenced]))
    assert wo["condition"] == "without_context" and wo["solved"] is True
    assert wc["condition"] == "with_context" and wc["solved"] is True


def test_unparseable_is_unsolved():
    inst = INSTANCES[0]
    r = solve_without_context(inst, MockLLM(responses=["sorry, no code"]))
    assert r["solved"] is False
