"""E119 macro slot: object/action-referential op grammar + compiler, SLM proposer (best-of-N +
behavioral clustering + abstention), subgoal-proxy ranker, and a seeded random-macro baseline.
The env decides correctness; macros only ORDER/extend search and are replay-verified before banking."""
import json, re
import numpy as np
from collections import defaultdict
from e119 import planner, solve, slm

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


_PROMPT = (
    "Blind search STALLED on an interactive puzzle. Propose ONE short action PROCEDURE (a macro) "
    "to make progress. Relational scene:\n{oj}\nWhat each action did from the current state:\n{diffs}\n"
    "Goal to pursue: {subgoal}\n"
    'Reply ONLY a JSON list of {k} ops max. Ops: "aN" (do action N), "aN xK" (repeat K), '
    '"click #I" (click object I). Example: ["a7","a7","a1"].'
)


def _endpoint(game, prefix, macro_actions, key_fn):
    """Replay prefix+macro from reset on a FRESH _PrefixGame view; return (masked key, level delta)."""
    pg = solve._PrefixGame(game, prefix)
    base = pg.levels
    frame, levels, _ = planner._frame_after(pg, list(macro_actions))
    return key_fn(frame), levels - base


def rank_macros(macros, game, prefix, subgoal, key_fn, seen):
    """Order macros: subgoal-satisfying endpoints first, then novel (unseen) endpoints."""
    pred = slm.compile_predicate(subgoal) if subgoal else (lambda f: False)
    scored = []
    for i, m in enumerate(macros):
        pg = solve._PrefixGame(game, prefix)
        frame, _, _ = planner._frame_after(pg, list(m))
        sat = 1 if pred(frame) else 0
        novel = 1 if key_fn(frame) not in seen else 0
        scored.append((-sat, -novel, i, m))         # stable within tier via original index i
    scored.sort(key=lambda t: t[:3])
    return [t[3] for t in scored]


def propose_macros(llm, game, prefix, obj_json, diffs, subgoal, avail, key_fn,
                   k_max=8, n=6, tau=0.5):
    """Sample n op-lists from LLM, compile, cluster by behavioral effect (endpoint key + level delta).
    Return cluster representatives sorted by cluster mass, or [] (abstain) if top cluster doesn't clear tau."""
    prompt = _PROMPT.format(oj=json.dumps(obj_json)[:1200], diffs=json.dumps(diffs)[:800],
                            subgoal=json.dumps(subgoal), k=k_max)
    clusters = defaultdict(list)      # behavioral signature -> [compiled macro, ...]
    drawn = 0
    for _ in range(n):
        try:
            ops = json.loads(re.search(r"\[.*\]", llm.ask(prompt), re.S).group(0))
            m = compile_macro(ops, obj_json, avail)[:k_max]
        except Exception:
            continue
        drawn += 1
        if not m:                     # empty/ungradeable macro discarded, not fatal
            continue
        try:
            sig = _endpoint(game, prefix, m, key_fn)
        except Exception:
            continue
        clusters[sig].append(m)
    if not clusters:
        return []
    ranked = sorted(clusters.values(), key=len, reverse=True)
    top = len(ranked[0])
    if drawn == 0 or top / drawn < tau:    # no consensus -> abstain
        return []
    return [reps[0] for reps in ranked if len(reps) >= 1]
