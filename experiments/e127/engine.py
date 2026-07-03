"""Rollout, scoring, identity-masking and disagreement for stateful reconstructed engines.

Correctness scoring compares FULL frames (the engine must predict every cell, including the
status bar, from its own latent state). The identity mask is computed separately and used ONLY
for state-novelty hashing in probes — never to relax a correctness comparison."""
import numpy as np


class EngineError(Exception):
    pass


def rollout(factory, actions):
    """Return [reset_frame, frame_after_action0, ...]; raises EngineError on any runtime fault."""
    try:
        e = factory()
        frames = [np.asarray(e.reset())]
        for a in actions:
            frames.append(np.asarray(e.step(a)))
        return frames
    except Exception as ex:
        raise EngineError(str(ex))


def play(game, actions):
    """Drive a GameLike with `actions` (list of (kind,x,y)); return an Episode (elem 0 = reset)."""
    f0 = np.asarray(game.reset())
    ep = [{"action": None, "frame": f0, "levels": int(game.levels)}]
    for a in actions:
        kind, x, y = a
        f = np.asarray(game.step(kind, x, y))
        ep.append({"action": a, "frame": f, "levels": int(game.levels)})
    return ep


def score_rollout(factory, episode):
    """Score a factory's rollout against an observed Episode. FULL-frame correctness."""
    actions = [s["action"] for s in episode[1:]]
    try:
        e = factory()
        pred = [np.asarray(e.reset())]
        levels_pred = [int(e.state.get("levels", 0))]
        for a in actions:
            pred.append(np.asarray(e.step(a)))
            levels_pred.append(int(e.state.get("levels", 0)))
    except Exception:
        n = len(actions)
        return {"transitions": n, "exact": 0, "cell_acc": 0.0,
                "levelup_match": 0, "levelup_total": 0, "errored": True}
    exact = cell_sum = cell_tot = lv_match = lv_tot = 0
    for i in range(1, len(episode)):
        real = episode[i]["frame"]; pf = pred[i]
        if pf.shape != real.shape:
            cell_tot += real.size
            continue
        eq = (pf == real)
        if eq.all():
            exact += 1
        cell_sum += int(eq.sum()); cell_tot += real.size
        real_up = episode[i]["levels"] - episode[i - 1]["levels"]
        pred_up = levels_pred[i] - levels_pred[i - 1]
        if real_up > 0:
            lv_tot += 1
            if pred_up == real_up:
                lv_match += 1
    return {"transitions": len(episode) - 1, "exact": exact,
            "cell_acc": (cell_sum / cell_tot) if cell_tot else 0.0,
            "levelup_match": lv_match, "levelup_total": lv_tot, "errored": False}


def identity_mask(episodes, thr=0.95):
    """Bool mask (H,W): True where a cell changes between consecutive frames on >= thr of steps,
    aggregated across episodes. For state-IDENTITY hashing only."""
    H = W = None
    changed = total = None
    for ep in episodes:
        for i in range(1, len(ep)):
            a, b = ep[i - 1]["frame"], ep[i]["frame"]
            if changed is None:
                H, W = a.shape
                changed = np.zeros((H, W), dtype=float)
                total = 0
            changed += (a != b).astype(float)
            total += 1
    if total == 0:
        return np.zeros((1, 1), dtype=bool)
    return (changed / total) >= thr


def first_disagreement(factoryA, factoryB, actions):
    """First action index (0-based) whose resulting frame first differs between A and B, else None.
    Index 0 is also returned on rollout error. Reset-frame differences return 0."""
    try:
        fa = rollout(factoryA, actions); fb = rollout(factoryB, actions)
    except EngineError:
        return 0
    n = min(len(fa), len(fb))
    for i in range(n):
        if fa[i].shape != fb[i].shape or not np.array_equal(fa[i], fb[i]):
            return max(0, i - 1)
    return None


def looks_like_lookup_table(src, max_int_literals=30000):
    """Static degeneracy backstop: an engine that memorizes observed frames as a giant literal table.
    Flags only sources with an ENORMOUS count of integer literals -- a frame->frame dump of the corpus
    is ~(n_transitions x 4096) literals (hundreds of thousands), whereas a LEGITIMATE engine that
    embeds even a full literal 64x64 board is only ~4096 (and loop/region-based engines are ~100-300).
    The threshold sits far above any honest board-embedding so real-game engines are never false-
    rejected (the 120 default false-rejected real engines -- see dc22). The PRIMARY anti-memorization
    defense is the DISJOINT held-out set in certify: a corpus-memorizer scores ~0 on transitions it
    never saw and cannot certify regardless. This static check is only a cheap backstop for absurd dumps."""
    import re
    ints = re.findall(r"(?<![\w.])\d+", src)
    return len(ints) > max_int_literals
