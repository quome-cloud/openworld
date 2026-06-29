"""Object-level MACRO-actions for macro-Go-Explore.

Micro Go-Explore fails on ARC-3's procedural walls because the win procedure is hundreds of exact
micro-moves -- astronomically unlikely under random rollouts. As OBJECT-MACROS -- "click object X",
"move the avatar to object X" (which naturally pushes objects it moves into) -- the same procedure is
a handful of steps, so the search becomes tractable. Macros are realized as short micro-action
sub-sequences on the deterministic real env, built on the OpenWorld object perceptor (e125.objstate).
Direction semantics are LEARNED (not assumed) by probing the avatar, so it works regardless of a
game's action mapping."""
import numpy as np
from experiments.e125 import objstate

_DIRS = (1, 2, 3, 4)


def _step(g, a):
    try:
        g.step(a[0], a[1], a[2]) if a[0] == 6 else g.step(a[0])
        return not bool(getattr(g, "done", False))
    except Exception:
        return False


def _objs(frame):
    return objstate.object_state(np.asarray(frame).astype(int).tolist())["objects"]


def _avatar_pos(frame, color):
    cs = [(o["y"], o["x"]) for o in _objs(frame) if o["color"] == color]
    return cs[0] if cs else None


def find_avatar(game_factory, seed_actions, avail):
    """Identify the controllable avatar's COLOR and its learned direction->displacement map by probing
    each directional action from the frontier. Returns (avatar_color | None, {action:(dy,dx)}). None
    for click-only games (no avatar)."""
    dirs = [a for a in _DIRS if a in avail]
    if not dirs:
        return None, {}

    def frontier():
        g = game_factory(); g.reset()
        for a in (seed_actions or []):
            if not _step(g, tuple(a)):
                break
        return g

    def singletons(frame):
        # only UNIQUE-colored objects: the avatar is a distinct sprite, so this excludes groups of
        # same-colored objects (gems, walls) AND the color-cycling status bar (its color never matches).
        from collections import Counter
        objs = _objs(frame); cnt = Counter(o["color"] for o in objs)
        return {o["color"]: (o["y"], o["x"]) for o in objs if cnt[o["color"]] == 1}

    base = singletons(frontier().frame)
    moves = {}                              # color -> {action: (dy,dx)}
    for a in dirs:
        g = frontier()
        if not _step(g, (a, None, None)):
            continue
        for c, (y, x) in singletons(g.frame).items():
            if c in base and (y, x) != base[c]:
                moves.setdefault(c, {})[a] = (y - base[c][0], x - base[c][1])
    if not moves:
        return None, {}
    avatar = max(moves, key=lambda c: len(moves[c]))    # the most-mobile object = the avatar
    return avatar, moves[avatar]


def object_macros(frame, avail, avatar, dir_map):
    """Macro specs from the current objects: click each object (if clicks available); reach each
    non-avatar object (if an avatar + learned directions are known); raw 5/7 acts for completeness."""
    macros = []
    can_click = 6 in avail
    for o in _objs(frame):
        if can_click:
            macros.append(("click", int(o["y"]), int(o["x"])))
        if avatar is not None and o["color"] != avatar and dir_map:
            macros.append(("reach", int(o["y"]), int(o["x"])))
    for a in avail:
        if a in (5, 7):
            macros.append(("act", int(a), 0))
    return macros or [("act", 7, 0)]


def make_executor(avatar, dir_map, reach_micro=30):
    """executor(g, spec) -> (micro_actions_taken, ok). 'reach' greedily steps the avatar toward the
    target using the LEARNED dir_map (picks the action that most reduces Manhattan distance), which
    naturally pushes objects the avatar moves into. Caps at reach_micro micro-steps."""
    def executor(g, spec):
        kind = spec[0]
        if kind == "click":
            a = (6, int(spec[2]), int(spec[1]))         # x=col, y=row
            return [a], _step(g, a)
        if kind == "act":
            a = (int(spec[1]), None, None)
            return [a], _step(g, a)
        if kind == "reach":
            ty, tx = int(spec[1]), int(spec[2]); taken = []
            for _ in range(reach_micro):
                av = _avatar_pos(g.frame, avatar)
                if av is None:
                    return taken, True
                ay, ax = av
                if (ay, ax) == (ty, tx):
                    break
                best_a, best_d = None, abs(ay - ty) + abs(ax - tx)
                for a, (dy, dx) in dir_map.items():
                    nd = abs(ay + dy - ty) + abs(ax + dx - tx)
                    if nd < best_d:
                        best_d, best_a = nd, a
                if best_a is None:                       # no action reduces distance -> stuck
                    break
                a = (best_a, None, None); taken.append(a)
                if not _step(g, a):
                    return taken, False
            return taken, True
        return [], True
    return executor


def macro_solve(game_factory, budget, seed_actions=None, win=None, seed=0, explore_horizon=12):
    """Macro-Go-Explore: detect the avatar, expose object-macros, and run Go-Explore in MACRO space.
    Same archive/return/explore loop as micro Go-Explore, but the action vocabulary is object-directed
    macros -> a tractable search over the procedure. Returns the go_explore result + the avatar."""
    from experiments.e128.go_explore import go_explore
    g = game_factory(); g.reset()
    avail = list(getattr(g, "avail", [1, 2, 3, 4, 5, 7]))
    avatar, dir_map = find_avatar(game_factory, seed_actions, avail)
    cand = lambda frame, av: object_macros(frame, av, avatar, dir_map)
    ex = make_executor(avatar, dir_map)
    res = go_explore(game_factory, cand, budget, seed_actions=seed_actions, win=win, seed=seed,
                     explore_horizon=explore_horizon, executor=ex)
    res["avatar"] = avatar
    return res
