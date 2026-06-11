"""Best-effort restricted execution of generated transition code.

Generated dynamics run with a curated set of builtins and a whitelist of safe
modules (math, random with explicit seeding left to callers). This guards
against accidents in LLM-written code, not against adversarial code.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict

SAFE_BUILTINS: Dict[str, Any] = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
        "float", "frozenset", "int", "isinstance", "issubclass", "len", "list",
        "map", "max", "min", "pow", "range", "repr", "reversed", "round", "set",
        "sorted", "str", "sum", "tuple", "zip", "ValueError", "KeyError",
        "TypeError", "Exception", "print",
    )
}


class SandboxError(RuntimeError):
    pass


def load_transition_code(code: str, func_name: str = "transition") -> Callable:
    """Exec `code` in a restricted namespace and return the named function.

    The namespace exposes `math` and the `random` module (for stochastic
    worlds that thread a seed through the state, keeping rollouts
    replayable); imports and I/O remain unavailable.
    """
    import random

    namespace: Dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS, "math": math, "random": random,
    }
    try:
        exec(compile(code, "<transition>", "exec"), namespace)
    except Exception as exc:
        raise SandboxError(f"Code failed to execute: {exc!r}") from exc
    func = namespace.get(func_name)
    if not callable(func):
        raise SandboxError(f"Code does not define a callable {func_name}()")
    return func


def run_transition_code(
    code: str,
    state: Dict[str, Any],
    action: Dict[str, Any],
    func_name: str = "transition",
) -> Dict[str, Any]:
    """Execute generated transition code on (state, action) and return next state.

    State and action are deep-copied first: generated code often shallow-copies
    and mutates nested structures, which must never leak back to the caller.
    """
    func = load_transition_code(code, func_name)
    try:
        result = func(copy.deepcopy(dict(state)), copy.deepcopy(dict(action)))
    except Exception as exc:
        raise SandboxError(f"{func_name}(state, action) raised: {exc!r}") from exc
    if not isinstance(result, dict):
        raise SandboxError(
            f"{func_name}() must return a dict next-state, got {type(result).__name__}"
        )
    return result
