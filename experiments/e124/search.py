"""Subgoal-hill-climbing search with macros as multi-step options. Rungs:
  blind        : single-step BFS over pixel candidates (the control floor)
  blind_macros : BFS over pixel candidates + codex macros (macros applied atomically)
  subgoals     : best-first toward each subgoal predicate in order, pixel candidates
  full         : subgoals + macros (+ optional score_fn)
A level is solved only when the env raises `levels` (the caller re-verifies by replay)."""
from collections import deque
from e124 import sandbox_exec

def _apply(game, seq):
    game.reset()
    base = game.levels
    for a in seq:
        game.step(*a)
        if game.levels > base:
            return True
        if game.done:
            break
    return game.levels > base

def _candidate_steps(frame, candidates_fn, macros, use_macros):
    # Each element from candidates_fn is an action-arg list like [1] or [6,x,y].
    # Wrap each as a single-action step-sequence so cand = prefix + st stays as a
    # list of action-arg-lists (the format _apply expects: game.step(*a) for a in seq).
    # Note: the brief's stated correction kept s as-is when isinstance(s, list), but
    # that produces a flat [int] cand from prefix+[1] which breaks game.step(*int).
    steps = [[s] if isinstance(s, list) else [[s]] for s in candidates_fn(frame)]
    if use_macros:
        steps = list(macros) + steps      # try whole macros first
    return steps

def run(game, goal, budget, rung, candidates_fn, mask):
    use_macros = rung in ("blind_macros", "full")
    game.reset()
    frame0 = game.frame
    steps = _candidate_steps(frame0, candidates_fn, getattr(goal, "macros", []), use_macros)
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
