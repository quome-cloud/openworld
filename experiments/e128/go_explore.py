"""Go-Explore for source-free ARC-AGI-3 final-level solving (Ecoffet et al. 2019, arXiv:1901.10995).

ARC-3 final levels are deterministic, sparse-reward, procedural-goal problems -- the Montezuma's
Revenge of grid games -- and Go-Explore is the method built for exactly this. We implement Phase 1
(explore-until-solved); Phase 2 (robustify via imitation) is UNNECESSARY because ARC-3 is
replay-deterministic, so a stored trajectory replayed from reset() is already robust.

Phase 1, faithful to the paper:
  - CELL: an abstract state. We hash the masked frame (mask cells that change every step -- status
    bars / animations -- so they don't explode the archive). The paper stresses the cell
    representation is everything; masked-frame is our v1 (object-state is a refinement).
  - ARCHIVE: cell -> the BEST trajectory reaching it (deepest level, then shortest action path).
  - SELECT: pick a cell to return to, weighted by a count-based novelty heuristic (rarely-chosen
    cells score higher), times a LEVEL bonus so the search is driven toward the frontier (our
    focused-final-level twist).
  - RETURN: REPLAY the cell's stored trajectory from reset() (the paper's determinism trick -- no
    relearning).
  - EXPLORE: take random candidate actions from the cell for a short horizon; archive every new/
    better cell; bank the moment levels reaches `win`.

Source-free by construction: only frames/levels are read through the GameLike client; never code.
"""
import numpy as np
from experiments.e125 import objstate   # OpenWorld object-centric perceptor (the cell representation)

_DIR_DEFAULT = [1, 2, 3, 4, 5, 7]


def identity_mask(frames, thr=0.95):
    """Bool mask (H,W): True where a cell changes between consecutive frames on >= thr of steps.
    Those are status/animation 'noise' cells -- zeroed before hashing so they don't explode cells."""
    changed = total = None
    for i in range(1, len(frames)):
        a, b = np.asarray(frames[i - 1]), np.asarray(frames[i])
        if a.shape != b.shape:
            continue
        if changed is None:
            changed = np.zeros(a.shape, dtype=float)
            total = 0
        changed += (a != b).astype(float)
        total += 1
    if not total:
        return None
    return (changed / total) >= thr


def _denoise(frame, mask):
    """Zero ONLY per-step-noise cells (status timers/animations) to the background color, so they don't
    spawn phantom objects -- meaningful counters (which change SELECTIVELY, not every step) survive."""
    f = np.asarray(frame).astype(int)
    if mask is not None and getattr(mask, "shape", None) == f.shape:
        bg = int(np.bincount(f.ravel()).argmax())
        f = np.where(mask, bg, f)
    return f


def cell_key(frame, mask):
    """Masked-frame byte hash (the 'masked' cell rep / fallback)."""
    return _denoise(frame, mask).astype(np.int16).tobytes()


def object_cell(frame, mask, levels=0, ignore_colors=()):
    """OpenWorld object-state cell rep: de-noise, then the object-centric perceptor (connected
    components -> {bg, objects[color,size,y,x]}) canonicalized, COMBINED with the level (so progress is
    never collapsed). Captures object configuration, click-target sprites, and meaningful counters
    (rendered as objects/lit cells) while abstracting away pixel/animation noise. Clicks (mouse) are
    covered by candidate_actions; the level signal (which detects the win) is in the key."""
    f = _denoise(frame, mask)
    s = objstate.object_state(f.tolist(), ignore_colors)
    return (int(levels), objstate.state_key(s))


def _cellfn(rep):
    if rep == "masked":
        return lambda frame, mask, levels: (int(levels), cell_key(frame, mask))
    return lambda frame, mask, levels: object_cell(frame, mask, levels)


def _probe_mask(game_factory, steps=40, seed=0):
    """Short random rollout to learn which cells are per-step noise (for the cell mask)."""
    rng = np.random.default_rng(seed)
    g = game_factory()
    frames = [np.asarray(g.reset())]
    avail = list(getattr(g, "avail", _DIR_DEFAULT))
    for _ in range(steps):
        a = int(rng.choice(avail))
        frames.append(np.asarray(g.step(a, 0, 0) if a == 6 else g.step(a)))
    return identity_mask(frames)


def _select(archive, rng):
    """Paper's count-based novelty selection x a level bonus (drive toward the frontier).
    weight = (1 + levels) / sqrt(times_chosen + 1)."""
    cells = list(archive.values())
    w = np.array([(1.0 + c["levels"]) / np.sqrt(c["chosen"] + 1.0) for c in cells])
    w = w / w.sum()
    return cells[int(rng.choice(len(cells), p=w))]


def _safe_step(g, a):
    """Step the env; return True if the resulting state is explorable (valid frame AND not done),
    False if TERMINAL (done, or the env returned an empty frame = game over). Terminal states are
    dead ends -- never explored from, never archived."""
    try:
        g.step(a[0], a[1], a[2]) if a[0] == 6 else g.step(a[0])
        return not bool(getattr(g, "done", False))
    except Exception:
        return False


def _result(win, lv, actions, archive, real):
    return {"win": bool(win), "best_levels": int(lv), "best_actions": [list(x) for x in actions],
            "archive": len(archive), "real_steps": int(real)}


def _micro_executor(g, spec):
    """Default executor: a spec IS a single micro-action (kind,x,y). Returns the micro-actions it took
    (one) and whether the resulting state is explorable. Macro-Go-Explore passes an executor that
    realizes an object-macro as a SHORT sub-sequence of micro-actions (see e128.macros)."""
    a = tuple(spec)
    return [a], _safe_step(g, a)


def go_explore(game_factory, candidate_actions, budget, seed_actions=None,
               explore_horizon=15, seed=0, win=None, mask=None, cell_rep="object", executor=None):
    """Returns {win, best_levels, best_actions, archive, real_steps}. `candidate_actions(frame,avail)`
    returns a list of action SPECS; `executor(g, spec) -> (micro_actions_taken, ok)` applies one (default:
    a spec is a single micro-action (kind,x,y) -- directional or mouse click). Pass a MACRO executor +
    macro candidate_actions for object-level macro-Go-Explore. `seed_actions` = a banked frontier
    trajectory (micro) to seed from. `cell_rep`: 'object' (OpenWorld object-state, default) or 'masked'.
    The cell key always includes `levels`. Cell trajectories are stored as MICRO actions for exact replay."""
    rng = np.random.default_rng(seed)
    executor = executor or _micro_executor
    if mask is None:
        mask = _probe_mask(game_factory, seed=seed)
    cf = _cellfn(cell_rep)
    g = game_factory()
    f0 = np.asarray(g.reset())
    win = int(win if win is not None else getattr(g, "win", 1))
    archive = {}

    def consider(actions, frame, levels, done):
        if done:                     # terminal / dead-end states are NOT explorable -> never archived
            return
        k = cf(frame, mask, levels)
        cur = archive.get(k)
        if cur is None or levels > cur["levels"] or (levels == cur["levels"] and len(actions) < len(cur["actions"])):
            archive[k] = {"actions": [tuple(a) for a in actions], "levels": int(levels), "chosen": 0}

    real = 0
    consider([], f0, int(g.levels), bool(getattr(g, "done", False)))
    best = (int(g.levels), [])

    # SEED from the banked frontier, stopping at the deepest PRE-terminal state (banked frontiers can
    # overshoot the deepest level and flail into a done state -- we keep the clean level-N-1 cells).
    if seed_actions:
        gs = game_factory(); gs.reset(); path = []
        for a in seed_actions:
            a = tuple(a); ok = _safe_step(gs, a); real += 1; path.append(a)
            lv = int(gs.levels)
            if lv > best[0]:
                best = (lv, list(path))
            if lv >= win:
                return _result(True, lv, path, archive, real)
            if not ok:               # reached terminal -> stop seeding (don't archive the dead end)
                break
            consider(path, gs.frame, lv, False)

    while real < budget and archive:
        cell = _select(archive, rng); cell["chosen"] += 1
        # RETURN: replay the cell's trajectory from reset()
        g.reset(); terminal = False
        for a in cell["actions"]:
            if not _safe_step(g, a):
                terminal = True; break
        real += len(cell["actions"])
        if terminal:
            continue
        # EXPLORE: random candidate actions (directional + mouse clicks at inferred targets)
        path = list(cell["actions"])
        for _ in range(explore_horizon):
            if real >= budget:
                break
            acts = candidate_actions(g.frame, list(getattr(g, "avail", _DIR_DEFAULT)))
            if not acts:
                break
            spec = acts[int(rng.integers(0, len(acts)))]
            micro, ok = executor(g, spec)         # one macro = a short micro-action sub-sequence
            real += len(micro); path = path + [tuple(a) for a in micro]
            lv = int(g.levels)
            if lv > best[0]:
                best = (lv, list(path))
            if lv >= win:
                return _result(True, lv, path, archive, real)
            if not ok:               # terminal -> stop this rollout (don't archive the dead end)
                break
            consider(path, g.frame, lv, False)
    return _result(best[0] >= win, best[0], best[1], archive, real)
