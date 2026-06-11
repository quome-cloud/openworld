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
