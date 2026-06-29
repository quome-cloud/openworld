"""Compile a model-authored stateful Engine into a fresh-instance factory, sandboxed.

Engines run inside openworld's SAFE_BUILTINS (no __import__, no open/exec) PLUS a numpy handle
`np`, because pixel/grid math legitimately needs arrays. The gate environment is therefore a
SUBSET of what a World transition would later get (numpy + safe builtins), so a compiling engine
never fails for a missing name at search time. A source that fails to define `Engine` or raises at
class-definition time returns None; runtime faults inside reset/step surface to the caller."""
import builtins as _builtins
import numpy as np
from openworld.sandbox import SAFE_BUILTINS

# Class definitions executed via exec() need __build_class__ from builtins.
# We add it explicitly without broadening to dangerous builtins (__import__, open, eval, etc.).
_EXEC_BUILTINS = {**SAFE_BUILTINS, "__build_class__": _builtins.__build_class__, "__name__": "engine"}


def compile_engine(src):
    """Return a zero-arg factory producing fresh Engine instances, or None on compile failure."""
    ns = {"np": np, "__builtins__": _EXEC_BUILTINS}
    try:
        exec(src, ns)
    except Exception:
        return None
    cls = ns.get("Engine")
    if not isinstance(cls, type):
        return None

    def factory():
        return cls()

    # Smoke: construct once so an Engine whose __init__ explodes is rejected early.
    try:
        factory()
    except Exception:
        return None
    return factory
