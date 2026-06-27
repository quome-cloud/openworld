"""Wrap a synthesized predict() as a SimGame so planning happens IN the code model (free, deep), not the
real env. plan() searches the SimGame for a trajectory whose predicted level_up fires."""
from collections import deque
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


def plan(predict_fn, initial_frame, candidates_fn, budget, max_depth=40):
    """BFS in the SimGame for an action sequence whose predicted level_up fires. Returns the sequence or None."""
    steps = [s if isinstance(s, list) else [s] for s in candidates_fn(initial_frame)]
    frontier = deque([[]]); seen = set(); n = 0
    while frontier and n < budget:
        prefix = frontier.popleft()
        for st in steps:
            cand = prefix + [st]
            key = tuple(map(tuple, cand))
            if key in seen:
                continue
            seen.add(key); n += 1
            g = SimGame(predict_fn, initial_frame)
            for a in cand:
                g.step(*a)
                if g.done:
                    break
            if g.levels > 0:
                return cand
            if len(cand) < max_depth and not g.done:
                frontier.append(cand)
            if n >= budget:
                break
    return None
