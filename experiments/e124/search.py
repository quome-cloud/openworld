"""Subgoal-hill-climbing search with macros as multi-step options. Rungs:
  blind        : single-step BFS over candidates (the control floor)
  blind_macros : BFS over candidates + codex macros (macros applied atomically)
  subgoals     : reach each codex subgoal predicate IN ORDER (depth-collapsing), candidates only
  full         : subgoals + macros as options
A level is solved only when the env raises `levels` (the caller re-verifies by replay).

Predicates are compiled IN-PROCESS for the search hot loop (a subprocess per search node, as sandbox_exec
does, would be far too slow); each predicate was already subprocess-validated once at compile time
(codex_goalc). A predicate that errors on a frame is treated as not-satisfied -- robustness, not security
(codex is not adversarial)."""
from collections import deque
import numpy as np


def _apply(game, seq):
    game.reset(); base = game.levels
    for a in seq:
        game.step(*a)
        if game.levels > base:
            return True
        if game.done:
            break
    return game.levels > base


def _candidate_steps(frame, candidates_fn, macros, use_macros):
    # Each element from candidates_fn is an action-arg list like [1] or [6,x,y]; wrap as a single-action
    # step-sequence so cand = prefix + st stays a list of action-arg-lists (game.step(*a) for a in seq).
    steps = [[s] if isinstance(s, list) else [[s]] for s in candidates_fn(frame)]
    if use_macros:
        steps = list(macros) + steps      # try whole macros first
    return steps


def run(game, goal, budget, rung, candidates_fn, mask):
    if rung in ("subgoals", "full"):
        return _subgoal_search(game, goal, budget, rung == "full", candidates_fn, mask)
    use_macros = (rung == "blind_macros")
    game.reset()
    steps = _candidate_steps(game.frame, candidates_fn, getattr(goal, "macros", []), use_macros)
    frontier = deque([[]]); seen = set(); n = 0
    while frontier and n < budget:
        prefix = frontier.popleft()
        for st in steps:
            cand = prefix + st                       # a macro st extends the prefix by several actions
            key = tuple(map(tuple, cand))
            if key in seen:
                continue
            seen.add(key); n += 1
            if _apply(game, cand):
                return cand
            if len(cand) < 8:
                frontier.append(cand)
            if n >= budget:
                break
    return None


# ---- subgoal hill-climbing (Task 6b): the depth-collapse mechanism ----

def _compile_pred(src):
    """Compile a codex `def predicate(frame)->bool` source into an in-process callable (or None)."""
    ns = {"np": np, "__builtins__": __builtins__}
    try:
        exec(src, ns)
        fn = ns.get("predicate")
        return fn if callable(fn) else None
    except Exception:
        return None


def _masked(frame, mask):
    fr = np.asarray(frame)
    return np.where(mask, 0, fr) if mask is not None else fr


def _run_seq(game, seq, mask):
    """Replay seq from reset; return (raised_levels, masked_final_frame). masked frame is None if solved."""
    game.reset(); base = game.levels
    for a in seq:
        game.step(*a)
        if game.levels > base:
            return True, None
        if game.done:
            break
    return (game.levels > base), _masked(game.frame, mask)


def _pred_ok(pred_fn, mf):
    if pred_fn is None or mf is None:
        return False
    try:
        return bool(pred_fn(mf))
    except Exception:
        return False


def _reach(game, prefix, pred_fn, budget, steps, mask, maxdepth=12):
    """BFS from `prefix` for a sequence that raises levels ('solved') or satisfies the subgoal ('subgoal').
    Returns (status, seq, used) with status in {'solved','subgoal',None}."""
    frontier = deque([list(prefix)]); seen = {tuple(map(tuple, prefix))}; n = 0
    while frontier and n < budget:
        cur = frontier.popleft()
        for st in steps:
            cand = cur + st
            key = tuple(map(tuple, cand))
            if key in seen:
                continue
            seen.add(key); n += 1
            raised, mf = _run_seq(game, cand, mask)
            if raised:
                return "solved", cand, n
            if _pred_ok(pred_fn, mf):
                return "subgoal", cand, n
            if len(cand) < maxdepth:
                frontier.append(cand)
            if n >= budget:
                break
    return None, None, n


def _subgoal_search(game, goal, budget, use_macros, candidates_fn, mask):
    """Pursue codex's subgoals IN ORDER: reach subgoal_0, commit that prefix, reach subgoal_1 from there, ...
    Each step is a shallow BFS, so a deep procedure collapses into k shallow searches. A level-up at any
    point (env-verified) is a solve."""
    subs = getattr(goal, "subgoals", None) or []
    if not subs:
        return None
    game.reset()
    steps = _candidate_steps(game.frame, candidates_fn, getattr(goal, "macros", []), use_macros)
    committed = []; spent = 0
    for name, src in subs:
        pred_fn = _compile_pred(src)
        status, seq, used = _reach(game, committed, pred_fn, budget - spent, steps, mask)
        spent += used
        if status == "solved":
            return seq
        if status is None:
            return None                       # could not reach this subgoal within the remaining budget
        committed = seq                       # subgoal achieved -> continue from here
        if spent >= budget:
            return None
    # final leg: from the last waypoint, search for the level-up itself (no predicate -> only a solve ends it)
    status, seq, _ = _reach(game, committed, None, budget - spent, steps, mask)
    return seq if status == "solved" else None
