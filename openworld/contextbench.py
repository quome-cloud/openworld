"""OpenWorld-ContextBench: in-context learning for program repair.

The SWE-ContextBench analogue, built on the openworld-repairbench harness. Each
instance is a program-repair task (reusing `RepairBenchInstance` + the exact
fork+SIGKILL test runner) PLUS a `context_history`: a few *related* bugs that were
already fixed, on different modules but sharing the same underlying fix pattern
(e.g. "cap a value with min()", "reject an out-of-range update", "sort before
indexing").

The ablation is **with-context vs. without-context**: does feeding the model the
related solved examples help it transfer the fix pattern to a new module? This is
a different axis from openworld-repairbench's single-shot-vs-in-world loop — there the
feedback was *test results*; here it's *prior solved examples* (in-context
learning, not iterative feedback). Scoring is identical: solved = zero failures in
both hidden suites.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .llm import BaseLLM
from .parsing import extract_code
from .repairbench import (
    RepairBenchInstance, _SYSTEM, _base_prompt, _safe_ask, run_instance_tests,
)

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "datasets" / "openworld-contextbench" / "tasks.jsonl"


@dataclass
class ContextExample:
    """A related, already-solved bug shown as in-context guidance."""

    module_name: str
    issue: str
    buggy_source: str
    reference_source: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextExample":
        return cls(d["module_name"], d["issue"], d["buggy_source"], d["reference_source"])

    def to_dict(self) -> Dict[str, Any]:
        return {"module_name": self.module_name, "issue": self.issue,
                "buggy_source": self.buggy_source, "reference_source": self.reference_source}


@dataclass
class ContextBenchInstance:
    """A repair task plus a history of related solved bugs (the context)."""

    instance_id: str
    task: RepairBenchInstance
    context_history: List[ContextExample] = field(default_factory=list)
    pattern: str = ""  # the shared fix pattern (metadata/analysis only)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextBenchInstance":
        return cls(
            instance_id=d["instance_id"],
            task=RepairBenchInstance.from_dict(d["task"]),
            context_history=[ContextExample.from_dict(e) for e in d.get("context_history", [])],
            pattern=d.get("pattern", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "task": self.task.to_dict(),
            "context_history": [e.to_dict() for e in self.context_history],
            "pattern": self.pattern,
        }


def load_dataset(path: Optional[Any] = None) -> List[ContextBenchInstance]:
    path = Path(path) if path is not None else DEFAULT_PATH
    out: List[ContextBenchInstance] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(ContextBenchInstance.from_dict(json.loads(line)))
    return out


def _context_block(history: List[ContextExample]) -> str:
    if not history:
        return ""
    parts = ["Here are related bugs that were fixed earlier. The same kind of fix "
             "may apply to the new task."]
    for ex in history:
        parts.append(
            f"--- example: {ex.module_name} ---\n"
            f"Issue: {ex.issue}\n"
            f"Buggy:\n```python\n{ex.buggy_source}```\n"
            f"Fixed:\n```python\n{ex.reference_source}```"
        )
    return "\n\n".join(parts)


def _solve(task: RepairBenchInstance, llm: BaseLLM, context: str) -> Dict[str, Any]:
    prompt = (context + "\n\n" if context else "") + _base_prompt(task, task.buggy_source)
    reply = _safe_ask(llm, prompt, _SYSTEM)
    source = extract_code(reply)
    if not source.strip():
        return {"solved": False, "regression_failures_seen": 0}
    result = run_instance_tests(source, task)
    return {"solved": result["solved"],
            "regression_failures_seen": result["pass_to_pass"]["failed"]}


def solve_without_context(instance: ContextBenchInstance, llm: BaseLLM) -> Dict[str, Any]:
    """Baseline: solve the task with no prior examples."""
    r = _solve(instance.task, llm, context="")
    return {"instance_id": instance.instance_id, "condition": "without_context", **r}


def solve_with_context(instance: ContextBenchInstance, llm: BaseLLM) -> Dict[str, Any]:
    """Solve the task with the related solved examples prepended."""
    r = _solve(instance.task, llm, context=_context_block(instance.context_history))
    return {"instance_id": instance.instance_id, "condition": "with_context", **r}
