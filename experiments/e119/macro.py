"""E119 macro slot: object/action-referential op grammar + compiler, SLM proposer (best-of-N +
behavioral clustering + abstention), subgoal-proxy ranker, and a seeded random-macro baseline.
The env decides correctness; macros only ORDER/extend search and are replay-verified before banking."""
import json, re
import numpy as np

_MAX_REPEAT = 4


def _parse_op(op):
    """'a3 x2' -> ('a', 3, 2); 'click #1 x2' -> ('click', 1, 2); times defaults to 1."""
    m = re.match(r"\s*a(\d+)(?:\s*x\s*(\d+))?\s*$", op)
    if m:
        return ("a", int(m.group(1)), min(int(m.group(2) or 1), _MAX_REPEAT))
    m = re.match(r"\s*click\s*#(\d+)(?:\s*x\s*(\d+))?\s*$", op)
    if m:
        return ("click", int(m.group(1)), min(int(m.group(2) or 1), _MAX_REPEAT))
    return None


def compile_macro(ops, obj_json, avail):
    """Compile object/action-referential ops to primitive action tuples. Unresolvable ops dropped."""
    objs = {o["id"]: o for o in obj_json.get("objects", [])}
    out = []
    for op in ops:
        parsed = _parse_op(op) if isinstance(op, str) else None
        if parsed is None:
            continue
        kind, idx, times = parsed
        if kind == "a":
            if idx in avail and idx != 6:
                out += [(idx,)] * times
        else:  # click
            if 6 in avail and idx in objs:
                cy, cx = objs[idx]["centroid"]
                out += [(6, int(round(cx)), int(round(cy)))] * times
    return out
