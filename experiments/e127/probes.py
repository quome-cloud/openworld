"""Counterexample search against the REAL env (the oracle). Combines: (a) replaying observed
prefixes + novelty-guided extensions and diffing engine vs real, and (b) universal property
falsifiers. Real-env steps are charged to a shared budget {'limit','used'}; a depth-d replay costs d
steps (no env cloning), so we replay each candidate once from a fresh real_factory().

ALL-MODALITY candidate generation: the action pool is derived from the real env's actual action
space via perception.candidate_actions(reset_frame, avail) -- directional moves for a directional
game, pixel-inferred clicks (6,x,y) for a click game. Identity-masked click targets (e.g. the
status-bar cell that surfaces as a spurious size-1 component) are dropped to avoid wasted no-op
probes. This replaces the brief's hard-coded directional [1,2,3,4,5,7], which fails every click game.

Extensions are NOVELTY-GUIDED on the engine's own (masked) state space rather than purely random:
greedily choosing pool actions that drive the engine into states it has not yet rendered. This costs
no real budget (engine-only) yet systematically covers the board, so a wrong engine's divergence from
reality is reached far more reliably than a clamped random walk (which seldom hits a sparse target).
"""
import numpy as np
from experiments.e127 import engine as _engine
from experiments.e127 import perception as _perception


def _real_frames(real_factory, actions, budget):
    """Replay `actions` on a fresh real env; return (frames incl. reset, levels), charging
    len(actions) steps. Stops early (returns the partial list) if the budget is exhausted."""
    g = real_factory()
    frames = [np.asarray(g.reset())]
    levels = [int(g.levels)]
    for a in actions:
        if budget["used"] >= budget["limit"]:
            break
        kind, x, y = a
        frames.append(np.asarray(g.step(kind, x, y)))
        levels.append(int(g.levels))
        budget["used"] += 1
    return frames, levels


def _candidate_pool(real_factory, mask):
    """Derive the candidate action POOL from the real env's actual action space (all-modality).
    Excludes clicks onto identity-masked cells; falls back to a single no-op if empty."""
    g = real_factory()
    f0 = np.asarray(g.reset())
    avail = list(getattr(g, "avail", [1, 2, 3, 4, 5, 7]))
    pool = [tuple(a) for a in _perception.candidate_actions(f0, avail)]
    if mask is not None and getattr(mask, "shape", None) == f0.shape:
        pool = [a for a in pool if not (a[0] == 6 and bool(mask[a[2], a[1]]))]
    if not pool:
        pool = [(7, None, None)]
    return pool


def _masked_key(frame, mask):
    frame = np.asarray(frame)
    if mask is not None and getattr(mask, "shape", None) == frame.shape:
        f = frame.copy(); f[mask] = 0
        return f.tobytes()
    return frame.tobytes()


def _novelty_walk(factory, prefix, pool, mask, rng, n_extend):
    """Extend `prefix` by greedily picking pool actions whose resulting engine frame (status-bar
    masked) is novel; tie-break / fall back to a random pool action. Engine-only (no real budget)."""
    seq = list(prefix)
    try:
        frames = _engine.rollout(factory, seq)
    except _engine.EngineError:
        return seq
    seen = {_masked_key(f, mask) for f in frames}
    for _ in range(n_extend):
        order = [int(i) for i in rng.permutation(len(pool))]
        chosen = pool[order[0]]
        chosen_key = None
        for idx in order:
            a = pool[idx]
            try:
                fr = _engine.rollout(factory, seq + [a])
            except _engine.EngineError:
                continue
            key = _masked_key(fr[-1], mask)
            chosen, chosen_key = a, key
            if key not in seen:
                break          # found a novel state -> take it
        if chosen_key is not None:
            seen.add(chosen_key)
        seq.append(chosen)
    return seq


def _candidate_action_seqs(factory, observed, pool, mask, seed=0, n_extend=16, max_seqs=24):
    """Per episode: the bare observed prefixes plus novelty-guided extensions of them."""
    rng = np.random.default_rng(seed)
    seqs = []
    for ep in observed:
        base = [s["action"] for s in ep[1:]]
        for cut in (len(base), max(1, len(base) // 2)):
            seqs.append(base[:cut])                                            # bare prefix
            seqs.append(_novelty_walk(factory, base[:cut], pool, mask, rng, n_extend))
            if len(seqs) >= max_seqs:
                return seqs
    return seqs


def find_counterexamples(factory, real_factory, observed, mask, action_api, budget):
    """Return counterexamples where the engine's full frame != the real frame (earliest per seq)."""
    pool = _candidate_pool(real_factory, mask)
    cexs = []
    for actions in _candidate_action_seqs(factory, observed, pool, mask):
        if budget["used"] >= budget["limit"]:
            break
        real_frames, _ = _real_frames(real_factory, actions, budget)
        try:
            eng_frames = _engine.rollout(factory, actions[:len(real_frames) - 1])
        except _engine.EngineError:
            cexs.append({"actions": actions[:1], "index": 0, "real_frame": real_frames[0],
                         "engine_frame": np.full_like(real_frames[0], -1), "kind": "engine_error"})
            continue
        for i in range(1, min(len(real_frames), len(eng_frames))):
            if eng_frames[i].shape != real_frames[i].shape or not np.array_equal(eng_frames[i], real_frames[i]):
                cexs.append({"actions": actions[:i], "index": i, "real_frame": real_frames[i],
                             "engine_frame": eng_frames[i], "kind": "diff"})
                break
    return cexs


def property_violations(factory, real_factory, action_api, budget):
    """Falsify universal properties of the engine using the real env as reference. The probe is
    drawn (all-modality) from the real env's action space so click games are exercised too."""
    pool = _candidate_pool(real_factory, None)
    rng = np.random.default_rng(0)
    probe = [pool[int(rng.integers(len(pool)))] for _ in range(8)]
    viols = []
    # color_range: engine cells must stay within the real env's color alphabet (0..15)
    try:
        eng_frames = _engine.rollout(factory, probe)
        for i, f in enumerate(eng_frames):
            if f.min() < 0 or f.max() > 15:
                viols.append({"kind": "color_range", "index": i, "actions": probe[:i],
                              "engine_frame": f, "real_frame": None})
                break
    except _engine.EngineError:
        viols.append({"kind": "engine_error", "index": 0, "actions": probe[:1],
                      "engine_frame": None, "real_frame": None})
    # determinism: the engine must be deterministic (same actions -> same frames)
    try:
        f1 = _engine.rollout(factory, probe); f2 = _engine.rollout(factory, probe)
        if not all(np.array_equal(a, b) for a, b in zip(f1, f2)):
            viols.append({"kind": "determinism", "index": 0, "actions": probe,
                          "engine_frame": None, "real_frame": None})
    except _engine.EngineError:
        pass
    # levelup_delta: where the REAL env levels-up, the board changes substantially
    real_frames, real_levels = _real_frames(real_factory, probe, budget)
    for i in range(1, len(real_frames)):
        if real_levels[i] > real_levels[i - 1]:
            changed = (real_frames[i] != real_frames[i - 1]).mean()
            if changed < 0.02:
                viols.append({"kind": "levelup_delta", "index": i, "actions": probe[:i],
                              "engine_frame": None, "real_frame": real_frames[i]})
    return viols
