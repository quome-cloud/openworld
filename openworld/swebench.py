"""OpenWorld-SWE-bench: SWE-bench-style program repair with explicit world specs.

Each instance is a small multi-function (or class-based) Python module, a
natural-language issue report, and two hidden suites: fail_to_pass exercises
the reported bug, pass_to_pass guards against regressions. A patch solves the
instance only when zero tests fail in BOTH suites.

Every instance also carries a world-model spec; submitting a patch is a world
transition whose dynamics ARE exact test execution, as in openworld.coding.
The harness runs the same model two ways on every instance:
  - single-shot: issue + buggy module, one completion, no feedback;
  - in-world: iterative submit_patch with exact failing-test feedback.
"""

from __future__ import annotations

import builtins as _py_builtins
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .coding import run_tests
from .llm import BaseLLM
from .parsing import extract_code
from .state import Action, WorldState
from .transition import Transition
from .world import World

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "datasets" / "openworld-swebench" / "tasks.jsonl"
)

# coding's restricted builtins cannot define classes; these instances can.
# Note: object/type expose the __subclasses__() chain, so this sandbox (like
# coding's) is a guard against accidental misuse, not a security boundary
# for adversarial code.
CLASS_BUILTINS = {
    name: getattr(_py_builtins, name)
    for name in (
        "__build_class__", "getattr", "setattr", "hasattr", "delattr", "super",
        "object", "type", "property", "staticmethod", "classmethod", "callable",
        "hash", "AttributeError", "RuntimeError", "NotImplementedError",
        "LookupError", "OverflowError",
    )
}


@dataclass
class SWEBenchInstance:
    """One SWE-bench-style instance plus its world-model spec."""

    instance_id: str
    module_name: str
    issue: str
    buggy_source: str
    reference_source: str
    test_preamble: str
    fail_to_pass: List[Tuple[str, str]]
    pass_to_pass: List[Tuple[str, str]]
    world: Dict[str, Any]


_INSTANCE_FIELDS = {f.name for f in fields(SWEBenchInstance)}


def load_dataset(path: Optional[Path] = None) -> List[SWEBenchInstance]:
    """Read instances from the JSONL dataset artifact."""
    text = Path(path or DEFAULT_DATASET_PATH).read_text(encoding="utf-8")
    instances: List[SWEBenchInstance] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["fail_to_pass"] = [tuple(t) for t in record["fail_to_pass"]]
        record["pass_to_pass"] = [tuple(t) for t in record["pass_to_pass"]]
        instances.append(SWEBenchInstance(**{k: v for k, v in record.items() if k in _INSTANCE_FIELDS}))
    return instances


def run_instance_tests(
    source: str, instance: SWEBenchInstance, timeout_seconds: float = 5.0
) -> Dict[str, Any]:
    """Run both hidden suites against `source`. Solved = zero failures in both."""
    program = source
    if instance.test_preamble:
        program = source + "\n\n" + instance.test_preamble
    fail_to_pass = run_tests(
        program, instance.fail_to_pass, timeout_seconds, extra_builtins=CLASS_BUILTINS
    )
    pass_to_pass = run_tests(
        program, instance.pass_to_pass, timeout_seconds, extra_builtins=CLASS_BUILTINS
    )
    return {
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "solved": fail_to_pass["failed"] == 0 and pass_to_pass["failed"] == 0,
    }


def merged_errors(result: Dict[str, Any], limit: int = 3) -> List[str]:
    """Up to `limit` failure strings, never letting fail_to_pass crowd out
    regression errors entirely."""
    f2p = result["fail_to_pass"]["errors"]
    p2p = result["pass_to_pass"]["errors"]
    if p2p:
        return (f2p[: limit - 1] + p2p)[:limit]
    return f2p[:limit]


def initial_world_state(instance: SWEBenchInstance) -> Dict[str, Any]:
    """The world's initial symbolic state: the buggy module's exact test results."""
    result = run_instance_tests(instance.buggy_source, instance)
    return {
        "instance": instance.instance_id,
        "source": instance.buggy_source,
        "fail_to_pass_passed": result["fail_to_pass"]["passed"],
        "fail_to_pass_failed": result["fail_to_pass"]["failed"],
        "pass_to_pass_passed": result["pass_to_pass"]["passed"],
        "pass_to_pass_failed": result["pass_to_pass"]["failed"],
        "last_errors": merged_errors(result),
        "attempts": 0,
        "solved": result["solved"],
    }


class SWEBenchTransition(Transition):
    """Exact dynamics: submitting a patch runs both hidden suites."""

    def __init__(self, instance: SWEBenchInstance):
        self.instance = instance

    def step(self, state: WorldState, action: Action) -> WorldState:
        s = state.copy()
        if s.get("solved"):
            return s
        if action.name == "submit_patch":
            source = str(action.params.get("source", "")) or s["source"]
            result = run_instance_tests(source, self.instance)
            s["source"] = source
            s["fail_to_pass_passed"] = result["fail_to_pass"]["passed"]
            s["fail_to_pass_failed"] = result["fail_to_pass"]["failed"]
            s["pass_to_pass_passed"] = result["pass_to_pass"]["passed"]
            s["pass_to_pass_failed"] = result["pass_to_pass"]["failed"]
            s["last_errors"] = merged_errors(result)
            s["attempts"] += 1
            s["solved"] = result["solved"]
        return s


def build_swebench_world(instance: SWEBenchInstance) -> World:
    """Instantiate the instance's world spec with exact test-running dynamics."""
    spec = instance.world
    return World(
        name=spec["name"],
        description=spec["description"],
        initial_state=spec["initial_state"],
        actions=spec["actions"],
        rules=spec.get("rules", []),
        transition=SWEBenchTransition(instance),
    )


SYSTEM_PROMPT = (
    "You are an expert Python maintainer. You receive a bug report and the "
    "full source of a module. Reply with ONLY a python code block containing "
    "the complete corrected module. Keep all public names and signatures. "
    "Use only pure python and math."
)


def _base_prompt(instance: SWEBenchInstance, source: str) -> str:
    return (
        f"Bug report for module `{instance.module_name}`:\n{instance.issue}\n\n"
        f"Current module source:\n```python\n{source}\n```\n"
    )


def _feedback_prompt(instance: SWEBenchInstance, state: WorldState) -> str:
    errors = "\n".join(f"- {e}" for e in state["last_errors"]) or "- (none reported)"
    return (
        _base_prompt(instance, state["source"])
        + f"\nBug-report tests passing: {state['fail_to_pass_passed']}, "
        f"failing: {state['fail_to_pass_failed']}\n"
        f"Regression tests passing: {state['pass_to_pass_passed']}, "
        f"failing: {state['pass_to_pass_failed']}\n"
        f"Failing test feedback:\n{errors}\n\n"
        "Provide the corrected module."
    )


def solve_single_shot(instance: SWEBenchInstance, llm: BaseLLM) -> Dict[str, Any]:
    """Condition A: one completion from issue + buggy module, no feedback."""
    prompt = _base_prompt(instance, instance.buggy_source) + "\nProvide the corrected module."
    patch = extract_code(llm.ask(prompt, system=SYSTEM_PROMPT))
    result = run_instance_tests(patch, instance)
    return {
        "instance_id": instance.instance_id,
        "condition": "single_shot",
        "solved": result["solved"],
        "solved_first_attempt": result["solved"],
        "attempts": 1,
        "saw_regression": result["pass_to_pass"]["failed"] > 0,
    }


def solve_in_world(
    instance: SWEBenchInstance, llm: BaseLLM, budget: int = 4
) -> Dict[str, Any]:
    """Condition B: iterative repair inside the world, exact feedback each step."""
    world = build_swebench_world(instance)
    attempts_used = 0
    first_attempt_solved = False
    saw_regression = False
    for attempt in range(budget):
        patch = extract_code(
            llm.ask(_feedback_prompt(instance, world.state), system=SYSTEM_PROMPT)
        )
        world.step(Action("submit_patch", params={"source": patch}))
        attempts_used = attempt + 1
        if world.state["pass_to_pass_failed"] > 0:
            saw_regression = True
        if world.state["solved"]:
            first_attempt_solved = attempt == 0
            break
    return {
        "instance_id": instance.instance_id,
        "condition": "in_world",
        "solved": bool(world.state["solved"]),
        "solved_first_attempt": first_attempt_solved,
        "attempts": attempts_used,
        "saw_regression": saw_regression,
    }
