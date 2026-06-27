"""Change-seeking exploration: collect exact (frame,action,next_frame,level_up) transitions, preferring
actions that change the board (signal). Env = ground truth."""
import numpy as np


def collect(game_factory, candidates_fn, budget):
    g = game_factory(); g.reset()
    trans = []; seen = set()
    for _ in range(budget):
        cands = [s if isinstance(s, list) else [s] for s in candidates_fn(g.frame)]
        if not cands:
            break
        a = cands[len(trans) % len(cands)]               # round-robin (deterministic, covers the action set)
        pf = np.asarray(g.frame).copy(); lv = g.levels
        g.step(*a)
        nf = np.asarray(g.frame).copy()
        key = (pf.tobytes(), tuple(a))
        if key not in seen:
            seen.add(key)
            trans.append({"frame": pf, "action": list(a), "next_frame": nf, "level_up": g.levels > lv})
        if g.done:
            g = game_factory(); g.reset()
    return trans
