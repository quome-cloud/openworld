"""Exploration collects exact (frame,action,next_frame,level_up) transitions from the REAL env (ground truth).
Two policies: change-seeking round-robin (collect), and GOAL-DIRECTED energy descent (goal_directed_collect)
that walks the real env toward the synthesized goal_score hypothesis to reach and GROUND a real level-up -- the
online g.levels oracle the restartable sweep agent uses, the legitimate fix for the M2 win-coverage gap."""
import numpy as np
from e125 import simworld


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


def goal_directed_collect(game_factory, candidates_fn, predict_fn, goal_fn, budget):
    """Act in the REAL env to DESCEND the goal_score energy toward the hypothesised win, grounding a real
    level-up. Each step: preview every candidate's next frame via predict_fn, prefer the lowest-energy action
    that leads somewhere unseen (escape local minima / no-ops), execute it for real, record the transition. The
    ENV decides the win: when g.levels bumps, that transition is recorded level_up=True and we stop -- the win
    is now grounded in DATA, so re-synthesis turns the hypothesis into a verified win condition."""
    g = game_factory(); g.reset()
    trans, seen = [], set()
    for _ in range(budget):
        frame = np.asarray(g.frame)
        cands = [s if isinstance(s, list) else [s] for s in candidates_fn(frame)]
        if not cands:
            break
        scored = []
        for a in cands:
            try:
                nf, _ = predict_fn(frame, a)
            except Exception:
                nf = frame
            scored.append((simworld._energy(goal_fn, nf), a))
        scored.sort(key=lambda ea: ea[0])
        chosen = next((a for _, a in scored if (frame.tobytes(), tuple(a)) not in seen), scored[0][1])
        pf = frame.copy(); lv = g.levels
        g.step(*chosen)
        nf = np.asarray(g.frame).copy()
        seen.add((pf.tobytes(), tuple(chosen)))
        trans.append({"frame": pf, "action": list(chosen), "next_frame": nf, "level_up": g.levels > lv})
        if g.levels > lv:
            break                                     # grounded a real win -> stop so synth can learn it
        if g.done:
            g = game_factory(); g.reset()
    return trans


def collect_obj(game_factory, candidates_fn, budget, perceive, prefix=None):
    """Object-state exploration: replay prefix, then round-robin candidates, perceiving each frame to an object
    state; dedup by (state_key, action). Returns object transitions {state,action,next_state,level_up}."""
    from e125 import objstate
    prefix = list(prefix or [])

    def _fresh():
        g = game_factory(); g.reset()
        for a in prefix:
            g.step(*a)
        return g

    g = _fresh()
    trans, seen = [], set()
    for _ in range(budget):
        state = perceive(g.frame)
        cands = [s if isinstance(s, list) else [s] for s in candidates_fn(state)]
        if not cands:
            break
        a = cands[len(trans) % len(cands)]
        lv = g.levels
        g.step(*a)
        nstate = perceive(g.frame)
        key = (objstate.state_key(state), tuple(a))
        if key not in seen:
            seen.add(key)
            trans.append({"state": state, "action": list(a), "next_state": nstate, "level_up": g.levels > lv})
        if g.done:
            g = _fresh()
    return trans
