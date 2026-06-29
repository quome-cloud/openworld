# experiments/e130/navigation.py
"""Avatar detection + learned-direction navigation, so a 'reach' waypoint MOVES the avatar (the RL
review's fix for the realize() reach->click stub). Reuses e128.macros.find_avatar to learn the
direction->displacement map by probing; plan_reach greedily reduces Manhattan distance."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from experiments.e128 import macros as _m


def detect(game_factory, seed_actions, avail):
    return _m.find_avatar(game_factory, seed_actions, list(avail))


def avatar_pos(stereotype, avatar_color):
    for o in stereotype.objects:
        if o["color"] == avatar_color:
            return (o["y"], o["x"])
    return None


def plan_reach(stereotype, avatar_color, dir_map, target_yx, max_steps=30):
    if not dir_map or avatar_color is None:
        return []
    pos = avatar_pos(stereotype, avatar_color)
    if pos is None:
        return []
    ty, tx = target_yx
    y, x = pos
    plan = []
    for _ in range(max_steps):
        if (y, x) == (ty, tx):
            break
        best_a, best_d = None, abs(y - ty) + abs(x - tx)
        for a, (dy, dx) in dir_map.items():
            nd = abs(y + dy - ty) + abs(x + dx - tx)
            if nd < best_d:
                best_d, best_a = nd, a
        if best_a is None:
            break
        dy, dx = dir_map[best_a]; y, x = y + dy, x + dx
        plan.append((best_a, None, None))
    return plan
