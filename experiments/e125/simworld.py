"""Wrap a synthesized predict() as a SimGame so planning happens IN the code model (free, deep), not the
real env. plan() does best-first ENERGY DESCENT in the code model: it expands the frontier node with the
lowest goal_score (codex's hypothesised win energy), so deep wins a blind BFS can't reach become tractable.
Each frontier node STORES its own frame, so predict() runs once per expansion -- never re-replaying the
prefix (the M2 perf bottleneck). With no goal_fn it degrades to plain breadth-first."""
import heapq
import numpy as np


class SimGame:
    """A game-shaped wrapper over predict(frame,action)->(next_frame,level_up). reset()/step(*a) only touch
    the code model, never the real env."""
    def __init__(self, predict_fn, initial_frame):
        self.predict_fn = predict_fn
        self._init = np.asarray(initial_frame).copy()
        self.reset()
    def reset(self):
        self.frame = self._init.copy(); self.levels = 0; self.done = False; return self
    def step(self, a, x=None, y=None):
        action = [a] if x is None else [a, x, y]
        try:
            nf, lu = self.predict_fn(self.frame, action)
        except Exception:
            self.done = True; return
        self.frame = np.asarray(nf)
        if lu:
            self.levels += 1; self.done = True


def _energy(goal_fn, frame):
    """goal_score as energy (lower = closer to goal); a broken/None heuristic is treated as flat (BFS)."""
    if goal_fn is None:
        return 0.0
    try:
        v = float(goal_fn(np.asarray(frame)))
    except Exception:
        return 1e18
    return v if np.isfinite(v) else 1e18


def plan(predict_fn, initial_frame, candidates_fn, budget, max_depth=40, goal_fn=None):
    """Best-first energy descent in the SimGame for an action sequence whose predicted level_up fires. Frontier
    is ordered by goal_fn energy (then shallower-first), each node carries its own frame (one predict() per
    expansion, no prefix replay), budget bounds predict() calls. Returns the winning action list or None."""
    init = np.asarray(initial_frame)
    seen = {init.tobytes()}
    counter = 0
    # heap items: (energy, depth, tiebreak, frame, actions). tiebreak keeps the heap total-ordered (never
    # compares ndarrays) and makes ties FIFO so goal_fn=None is plain breadth-first.
    heap = [(_energy(goal_fn, init), 0, counter, init, [])]
    n = 0
    while heap and n < budget:
        _, depth, _, frame, actions = heapq.heappop(heap)
        if depth >= max_depth:
            continue
        for st in (s if isinstance(s, list) else [s] for s in candidates_fn(frame)):
            try:
                nf, lu = predict_fn(frame, st)
            except Exception:
                continue
            n += 1
            nf = np.asarray(nf)
            if lu:
                return actions + [st]
            key = nf.tobytes()
            if key not in seen:
                seen.add(key); counter += 1
                heapq.heappush(heap, (_energy(goal_fn, nf), depth + 1, counter, nf, actions + [st]))
            if n >= budget:
                break
    return None
