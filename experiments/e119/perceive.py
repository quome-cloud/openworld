"""Deterministic perception: masking, object-JSON, diffs, click candidates, probe."""
import numpy as np
import arc3_graph  # sibling module on sys.path when run from experiments/


def status_mask(frames, thresh=0.95):
    """Cells that change on >thresh of step-to-step transitions -> mask (zero before hashing)."""
    arr = np.stack([np.asarray(f).reshape(64, 64) for f in frames])
    if len(arr) < 2:
        return np.zeros((64, 64), bool)
    changes = (arr[1:] != arr[:-1]).mean(axis=0)   # fraction of steps each cell changed
    return changes > thresh


def state_key(frame, mask):
    g = np.asarray(frame).reshape(64, 64).copy()
    g[mask] = 0
    return g.astype(np.int16).tobytes()


def object_json(frame, bg=None):
    g = np.asarray(frame).reshape(64, 64)
    objs, bg = arc3_graph.objects(g, bg=bg)
    out = []
    for i, o in enumerate(objs):
        out.append({"id": i, "color": o["color"], "size": o["size"],
                    "centroid": o["centroid"], "bbox": o["bbox"]})
    relations = []
    if out:
        ref = out[0]["centroid"]                       # largest object is the anchor
        for o in out[1:]:
            dy = round(o["centroid"][0] - ref[0], 1)
            dx = round(o["centroid"][1] - ref[1], 1)
            relations.append(f"#{o['id']}(c{o['color']}) at dy={dy},dx={dx} of #0")
    return {"bg": bg, "objects": out, "relations": relations}


def contrastive_diff(before, after, bg=None):
    ba, aa = object_json(before, bg)["objects"], object_json(after, bg)["objects"]
    by_color_b, by_color_a = {}, {}
    for o in ba: by_color_b.setdefault(o["color"], []).append(o)
    for o in aa: by_color_a.setdefault(o["color"], []).append(o)
    moved, appeared, vanished = [], [], []
    for color, alist in by_color_a.items():
        blist = by_color_b.get(color, [])
        if blist and alist:
            b0, a0 = blist[0], alist[0]
            if b0["centroid"] != a0["centroid"]:
                moved.append({"color": color, "from": b0["centroid"], "to": a0["centroid"]})
        elif alist and not blist:
            appeared.append({"color": color, "at": alist[0]["centroid"]})
    for color in by_color_b:
        if color not in by_color_a:
            vanished.append({"color": color})
    return {"moved": moved, "appeared": appeared, "vanished": vanished, "recolored": []}


def click_candidates(frame, bg=None, max_size=40):
    g = np.asarray(frame).reshape(64, 64)
    objs, bg = arc3_graph.objects(g, bg=bg)
    color_counts = {}
    for o in objs:
        color_counts[o["color"]] = color_counts.get(o["color"], 0) + o["size"]
    cands = []
    for o in objs:
        small = o["size"] <= max_size
        rare = color_counts[o["color"]] <= max_size
        if small or rare:
            cy, cx = o["centroid"]
            cands.append((int(round(cx)), int(round(cy))))   # (x=col, y=row)
    # dedup, stable order
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def probe(game):
    """Single-step transitions from reset(): each directional avail action + each click candidate.
    Returns list of {action, before, after, dlevels}. Replays from reset() per probe (env is replay-only)."""
    game.reset()
    base_frame = np.asarray(game.frame).reshape(64, 64).copy()
    base_levels = game.levels
    avail = [a for a in getattr(game, "avail", [1, 2, 3, 4, 5, 7]) if a != 6]
    transitions = []
    for a in avail:
        game.reset()
        game.step(a)
        transitions.append({"action": (a,), "before": base_frame,
                            "after": np.asarray(game.frame).reshape(64, 64).copy(),
                            "dlevels": game.levels - base_levels})
    if 6 in getattr(game, "avail", []):
        for (x, y) in click_candidates(base_frame):
            game.reset()
            game.step(6, x, y)
            transitions.append({"action": (6, x, y), "before": base_frame,
                                "after": np.asarray(game.frame).reshape(64, 64).copy(),
                                "dlevels": game.levels - base_levels})
    return transitions
