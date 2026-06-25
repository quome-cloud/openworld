"""Object-graph abstraction for ARC-AGI-3 grids: turn a 64x64 pixel frame into a relational graph
of objects (connected color components) + their attributes, so a synthesizer can reason about
dynamics and GOALS relationally ("move block onto target") instead of pixel-by-pixel.

Used to upgrade E86 synthesis (relational rules) and E89 goal inference (relational goals):
  graph_repr(frame)        -> compact text of objects (color/size/centroid/bbox) for prompting
  graph_diff(frame, next)  -> relational transition (which objects MOVED / appeared / vanished / resized)

Zero-dependency beyond numpy (matches the experiments stack).
"""
import numpy as np


def bg_of(g):
    v, c = np.unique(g, return_counts=True)
    return int(v[np.argmax(c)])


def objects(frame, bg=None, connectivity=4):
    """Connected color components (per-color flood fill). Returns list of object dicts + bg color."""
    g = np.asarray(frame)
    H, W = g.shape
    if bg is None:
        bg = bg_of(g)
    seen = np.zeros((H, W), bool)
    nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if connectivity == 8:
        nbrs += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    objs = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [y for y, x in cells]
            xs = [x for y, x in cells]
            objs.append({"color": color, "size": len(cells),
                         "centroid": (round(sum(ys) / len(ys), 1), round(sum(xs) / len(xs), 1)),
                         "bbox": (min(ys), min(xs), max(ys), max(xs)),
                         "shape": tuple(sorted((y - min(ys), x - min(xs)) for y, x in cells))})
    objs.sort(key=lambda o: (-o["size"], o["centroid"]))
    return objs, bg


def graph_repr(frame, max_objs=24):
    """Compact text of the object graph (for prompting a synthesizer)."""
    objs, bg = objects(frame)
    lines = [f"background=color{bg}, {len(objs)} objects:"]
    for i, o in enumerate(objs[:max_objs]):
        r0, c0, r1, c1 = o["bbox"]
        lines.append(f"  obj{i}: color={o['color']} size={o['size']} "
                     f"centroid={o['centroid']} bbox=({r0},{c0})-({r1},{c1})")
    if len(objs) > max_objs:
        lines.append(f"  ...(+{len(objs) - max_objs} more)")
    return "\n".join(lines)


def _match(a_objs, b_objs):
    """Greedy match objects across two frames by (same shape & color) then nearest centroid."""
    used = set()
    pairs = []
    for i, a in enumerate(a_objs):
        best, bj = 1e9, None
        for j, b in enumerate(b_objs):
            if j in used or b["color"] != a["color"]:
                continue
            shape_pen = 0 if b["shape"] == a["shape"] else 50
            d = abs(a["centroid"][0] - b["centroid"][0]) + abs(a["centroid"][1] - b["centroid"][1]) + shape_pen
            if d < best:
                best, bj = d, j
        if bj is not None and best < 1e9:
            used.add(bj)
            pairs.append((i, bj, best))
    matched_b = {j for _, j, _ in pairs}
    return pairs, matched_b


def graph_diff(frame, nxt):
    """Relational transition: objects that MOVED (centroid delta), APPEARED, VANISHED, RESIZED."""
    a_objs, _ = objects(frame)
    b_objs, _ = objects(nxt)
    pairs, matched_b = _match(a_objs, b_objs)
    moved, resized = [], []
    matched_a = set()
    for i, j, _ in pairs:
        matched_a.add(i)
        a, b = a_objs[i], b_objs[j]
        dr = round(b["centroid"][0] - a["centroid"][0], 1)
        dc = round(b["centroid"][1] - a["centroid"][1], 1)
        if dr or dc:
            moved.append({"color": a["color"], "size": a["size"], "delta": (dr, dc)})
        if b["size"] != a["size"]:
            resized.append({"color": a["color"], "from": a["size"], "to": b["size"]})
    vanished = [{"color": o["color"], "size": o["size"], "centroid": o["centroid"]}
                for i, o in enumerate(a_objs) if i not in matched_a]
    appeared = [{"color": o["color"], "size": o["size"], "centroid": o["centroid"]}
                for j, o in enumerate(b_objs) if j not in matched_b]
    return {"moved": moved, "appeared": appeared, "vanished": vanished, "resized": resized}


if __name__ == "__main__":
    import sys
    # quick self-test on a synthetic frame: a 2x2 block that moves right by 3
    a = np.zeros((64, 64), int)
    a[10:12, 10:12] = 9
    b = a.copy(); b[10:12, 10:12] = 0; b[10:12, 13:15] = 9
    print(graph_repr(a))
    print("diff:", graph_diff(a, b))
    sys.exit(0)
