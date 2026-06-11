"""OpenWorld-SWE-bench: program repair with an explicit world-model wrapper.

A SWE-bench-style program-repair dataset where every instance carries an explicit
world-model representation. Each instance is a buggy Python module plus a
natural-language issue and two hidden test suites (`fail_to_pass`, `pass_to_pass`).
The world's single action `submit_patch` runs the suites bit-exactly in a
restricted sandbox; dynamics are EXACT by construction (the simulator IS code
execution), so every reward is verifiable.

The point of the world-model framing is the ablation: the same model is compared
**single-shot** (one patch, no feedback) against itself operating **inside the
world** (iterative patches with exact failing-test feedback). Same prompts, same
model — the only difference is the feedback loop.

This module follows the `openworld.coding` pattern; tests reuse its fork+SIGKILL
runner philosophy, widened so generated *classes* (stateful instances) can be
defined in the sandbox.
"""

from __future__ import annotations

import builtins as _builtins
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .coding import _TEST_BUILTINS
from .llm import BaseLLM
from .parsing import extract_code
from .state import Action, WorldState
from .transition import Transition
from .world import World

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "datasets" / "openworld-swebench" / "tasks.jsonl"

# Instances may define stateful classes, so the sandbox needs the class-creation
# machinery and a few more names than the function-only coding benchmark.
_SWE_BUILTINS: Dict[str, Any] = {
    **_TEST_BUILTINS,
    **{
        name: getattr(_builtins, name)
        for name in (
            "__build_class__", "object", "property", "staticmethod", "classmethod",
            "super", "hasattr", "getattr", "setattr", "delattr", "callable", "format",
            "hash", "id", "type", "AttributeError", "RuntimeError", "NotImplementedError",
            "OverflowError", "ArithmeticError", "StopAsyncIteration",
        )
        if hasattr(_builtins, name)
    },
}

TestPair = Tuple[str, str]  # (call_expression, expected_repr)


@dataclass
class SWEBenchInstance:
    """One program-repair instance with an explicit world spec.

    Only `module_name`, `issue`, and `buggy_source` are shown to a model under
    test. `reference_source`, `test_preamble`, and the two suites are the
    held-out answer key.
    """

    instance_id: str
    module_name: str
    issue: str
    buggy_source: str
    reference_source: str = field(repr=False, default="")
    test_preamble: str = field(repr=False, default="")
    fail_to_pass: List[TestPair] = field(repr=False, default_factory=list)
    pass_to_pass: List[TestPair] = field(repr=False, default_factory=list)
    world: Dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SWEBenchInstance":
        return cls(
            instance_id=d["instance_id"],
            module_name=d["module_name"],
            issue=d["issue"],
            buggy_source=d["buggy_source"],
            reference_source=d.get("reference_source", ""),
            test_preamble=d.get("test_preamble", ""),
            fail_to_pass=[tuple(p) for p in d.get("fail_to_pass", [])],
            pass_to_pass=[tuple(p) for p in d.get("pass_to_pass", [])],
            world=d.get("world", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "module_name": self.module_name,
            "issue": self.issue,
            "buggy_source": self.buggy_source,
            "reference_source": self.reference_source,
            "test_preamble": self.test_preamble,
            "fail_to_pass": [list(p) for p in self.fail_to_pass],
            "pass_to_pass": [list(p) for p in self.pass_to_pass],
            "world": self.world,
        }


def load_dataset(path: Optional[Any] = None) -> List[SWEBenchInstance]:
    """Read the JSONL dataset (default `datasets/openworld-swebench/tasks.jsonl`)."""
    path = Path(path) if path is not None else DEFAULT_PATH
    instances: List[SWEBenchInstance] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            instances.append(SWEBenchInstance.from_dict(json.loads(line)))
    return instances


# ---------------------------------------------------------------------------
# Test execution (exact dynamics)
# ---------------------------------------------------------------------------

def _eval_suite(namespace: Dict[str, Any], tests: List[TestPair]) -> Dict[str, Any]:
    passed = 0
    errors: List[str] = []
    for expression, expected in tests:
        try:
            result = eval(expression, namespace)  # noqa: S307 - sandboxed namespace
            if repr(result) == expected:
                passed += 1
            else:
                errors.append(f"{expression} -> {result!r}, expected {expected}")
        except Exception as exc:
            errors.append(f"{expression} raised {exc!r}")
    return {"passed": passed, "failed": len(tests) - passed, "errors": errors}


def _run_inline(source: str, instance: "SWEBenchInstance") -> Dict[str, Any]:
    import math

    f2p, p2p = instance.fail_to_pass, instance.pass_to_pass
    namespace: Dict[str, Any] = {
        "__builtins__": dict(_SWE_BUILTINS), "__name__": "submission", "math": math,
    }
    try:
        exec(compile(source, "<submission>", "exec"), namespace)
        if instance.test_preamble:
            exec(compile(instance.test_preamble, "<preamble>", "exec"), namespace)
    except Exception as exc:
        msg = f"setup failed: {exc!r}"
        return {
            "fail_to_pass": {"passed": 0, "failed": len(f2p), "errors": [msg]},
            "pass_to_pass": {"passed": 0, "failed": len(p2p), "errors": [msg]},
            "solved": False,
        }
    f2p_result = _eval_suite(namespace, f2p)
    p2p_result = _eval_suite(namespace, p2p)
    return {
        "fail_to_pass": f2p_result,
        "pass_to_pass": p2p_result,
        "solved": f2p_result["failed"] == 0 and p2p_result["failed"] == 0,
    }


def _all_failed(instance: "SWEBenchInstance", message: str) -> Dict[str, Any]:
    return {
        "fail_to_pass": {"passed": 0, "failed": len(instance.fail_to_pass), "errors": [message]},
        "pass_to_pass": {"passed": 0, "failed": len(instance.pass_to_pass), "errors": [message]},
        "solved": False,
    }


def run_instance_tests(
    source: str, instance: "SWEBenchInstance", timeout_seconds: float = 5.0
) -> Dict[str, Any]:
    """Run both hidden suites against `source`. Returns per-suite results + solved.

    `solved` requires zero failures in BOTH suites (the fix repairs the bug
    without breaking regression tests). On POSIX the suites run in a forked child
    that the parent SIGKILLs at the deadline (a bare `except:` in a patched loop
    can swallow an in-process alarm — only a hard kill is loop-proof).
    """
    import os

    if timeout_seconds and hasattr(os, "fork"):
        return _run_forked(source, instance, timeout_seconds)
    return _run_inline(source, instance)


def _run_forked(
    source: str, instance: "SWEBenchInstance", timeout_seconds: float
) -> Dict[str, Any]:
    import os
    import signal
    import time

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.close(read_fd)
            payload = json.dumps(_run_inline(source, instance)).encode("utf-8")
            os.write(write_fd, payload)
            os.close(write_fd)
        finally:
            os._exit(0)

    os.close(write_fd)
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    while True:
        done_pid, _status = os.waitpid(pid, os.WNOHANG)
        if done_pid:
            break
        if time.monotonic() > deadline:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            timed_out = True
            break
        time.sleep(0.02)

    chunks = []
    while True:
        chunk = os.read(read_fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)

    if timed_out:
        return _all_failed(instance, "source timed out (killed; possible infinite loop)")
    data = b"".join(chunks)
    if not data:
        return _all_failed(instance, "test process died without reporting")
    return json.loads(data.decode("utf-8"))


# ---------------------------------------------------------------------------
# World wrapper
# ---------------------------------------------------------------------------

class SWEBenchTransition(Transition):
    """Exact dynamics: `submit_patch` runs the hidden suites and updates state."""

    def __init__(self, instance: SWEBenchInstance, timeout_seconds: float = 5.0):
        self.instance = instance
        self.timeout_seconds = timeout_seconds

    def step(self, state: WorldState, action: Action) -> WorldState:
        s = state.copy()
        if s.get("solved"):
            return s
        if action.name == "submit_patch":
            source = str(action.params.get("source", "")) or s["source"]
            result = run_instance_tests(source, self.instance, self.timeout_seconds)
            s["source"] = source
            s["fail_to_pass_passed"] = result["fail_to_pass"]["passed"]
            s["fail_to_pass_failed"] = result["fail_to_pass"]["failed"]
            s["pass_to_pass_passed"] = result["pass_to_pass"]["passed"]
            s["pass_to_pass_failed"] = result["pass_to_pass"]["failed"]
            s["last_errors"] = (result["fail_to_pass"]["errors"] + result["pass_to_pass"]["errors"])[:3]
            s["attempts"] += 1
            s["solved"] = result["solved"]
        return s


def build_swebench_world(instance: SWEBenchInstance, timeout_seconds: float = 5.0) -> World:
    """Instantiate a World wrapping one instance, seeded with the buggy source."""
    initial = run_instance_tests(instance.buggy_source, instance, timeout_seconds)
    spec = instance.world or {}
    return World(
        name=spec.get("name", f"swebench:{instance.instance_id}"),
        description=spec.get(
            "description",
            f"Program repair for module {instance.module_name!r}. Submit corrected "
            "source via submit_patch(params={'source': ...}).",
        ),
        initial_state={
            "instance_id": instance.instance_id,
            "source": instance.buggy_source,
            "fail_to_pass_passed": initial["fail_to_pass"]["passed"],
            "fail_to_pass_failed": initial["fail_to_pass"]["failed"],
            "pass_to_pass_passed": initial["pass_to_pass"]["passed"],
            "pass_to_pass_failed": initial["pass_to_pass"]["failed"],
            "last_errors": (initial["fail_to_pass"]["errors"] + initial["pass_to_pass"]["errors"])[:3],
            "attempts": 0,
            "solved": initial["solved"],
        },
        actions=["submit_patch"],
        rules=spec.get("rules", []),
        transition=SWEBenchTransition(instance, timeout_seconds),
    )


# ---------------------------------------------------------------------------
# Episode runners: single-shot vs in-world (the paired ablation)
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are an expert Python engineer fixing a bug. Reply with a single python "
    "code block containing the COMPLETE corrected module (all functions/classes), "
    "not a diff. Keep the public interface identical; fix only what the issue "
    "describes without breaking unrelated behavior."
)


def _base_prompt(instance: SWEBenchInstance, source: str) -> str:
    return (
        f"Module: {instance.module_name}\n\n"
        f"Issue report:\n{instance.issue}\n\n"
        f"Current source:\n```python\n{source}\n```\n"
    )


def solve_single_shot(instance: SWEBenchInstance, llm: BaseLLM) -> Dict[str, Any]:
    """One prompt, one completion, one hidden-suite run."""
    reply = llm.ask(_base_prompt(instance, instance.buggy_source), system=_SYSTEM)
    source = extract_code(reply)
    if not source.strip():
        return {"instance_id": instance.instance_id, "condition": "single_shot",
                "solved": False, "solved_first_attempt": False, "attempts": 1,
                "regression_failures_seen": 0}
    result = run_instance_tests(source, instance)
    return {
        "instance_id": instance.instance_id,
        "condition": "single_shot",
        "solved": result["solved"],
        "solved_first_attempt": result["solved"],
        "attempts": 1,
        "regression_failures_seen": result["pass_to_pass"]["failed"],
    }


def solve_in_world(instance: SWEBenchInstance, llm: BaseLLM, budget: int = 4) -> Dict[str, Any]:
    """Iterate inside the world: each prompt carries exact failing-test feedback."""
    world = build_swebench_world(instance)
    state = world.state
    solved_first = False
    regressions_seen = 0
    attempts = 0
    for i in range(budget):
        s = state
        feedback = ""
        if i > 0:
            feedback = (
                f"\nAfter your last patch: fail_to_pass {s['fail_to_pass_passed']} passed / "
                f"{s['fail_to_pass_failed']} failed; pass_to_pass {s['pass_to_pass_passed']} "
                f"passed / {s['pass_to_pass_failed']} failed.\n"
                f"Errors:\n" + "\n".join(f"- {e}" for e in s["last_errors"]) + "\n"
                "Fix the remaining failures without breaking the passing tests."
            )
        reply = llm.ask(_base_prompt(instance, s["source"]) + feedback, system=_SYSTEM)
        source = extract_code(reply)
        attempts += 1
        action = Action("submit_patch", params={"source": source or s["source"]}, agent="solver")
        state = world.step(action)
        regressions_seen = max(regressions_seen, state["pass_to_pass_failed"])
        if state["solved"]:
            solved_first = (i == 0)
            break
    return {
        "instance_id": instance.instance_id,
        "condition": "in_world",
        "solved": bool(state["solved"]),
        "solved_first_attempt": solved_first,
        "attempts": attempts,
        "regression_failures_seen": regressions_seen,
    }
