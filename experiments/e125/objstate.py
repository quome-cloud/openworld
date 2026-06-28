"""Object-centric perception: a 64x64 (or NxN) frame -> {"bg", "objects":[{color,size,y,x}]} via 4-connectivity
connected components, canonically sorted. state_key projects the DECISION-RELEVANT fields (positions) the
verifier gate compares -- abstracting away pixels/animation. PERCEIVE_SRC is the SAME logic as a self-contained
`perceive(data)` for an OpenWorld CodePerceptor (stdlib only, no imports -- runs in the sandbox)."""

_BODY = '''
    g = [list(map(int, row)) for row in (data[0] if (len(data) == 1 and isinstance(data[0], (list, tuple))
                                                     and isinstance(data[0][0], (list, tuple))) else data)]
    h = len(g); w = len(g[0])
    cnt = {}
    for row in g:
        for c in row:
            cnt[c] = cnt.get(c, 0) + 1
    bg = max(cnt, key=cnt.get)
    seen = [[False] * w for _ in range(h)]
    ents = []
    for i in range(h):
        for j in range(w):
            if seen[i][j] or g[i][j] == bg:
                continue
            color = g[i][j]; stack = [(i, j)]; seen[i][j] = True; cells = []
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and g[ny][nx] == color:
                        seen[ny][nx] = True; stack.append((ny, nx))
            if color in ignore_colors:
                continue
            ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
            ents.append({"color": int(color), "size": len(cells),
                         "y": int(round(sum(ys) / len(ys))), "x": int(round(sum(xs) / len(xs)))})
    ents.sort(key=lambda e: (e["color"], e["y"], e["x"], e["size"]))
    return {"bg": int(bg), "objects": ents}
'''


exec("def _extract(data, ignore_colors):" + _BODY, globals())   # compiled ONCE at import (not per call)


def object_state(frame, ignore_colors=()):
    return _extract(frame, set(ignore_colors))


def state_key(s, fields=("color", "y", "x")):
    return (int(s.get("bg", -1)),
            tuple(tuple(o[f] for f in fields) for o in s.get("objects", [])))


# Self-contained perceive(data) for a CodePerceptor (ignore_colors fixed to none at the boundary).
PERCEIVE_SRC = "def perceive(data):\n    ignore_colors = set()" + _BODY
