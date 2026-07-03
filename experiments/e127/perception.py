# experiments/e127/perception.py
"""All-modality perception for reconstruction. Frame perception = the (H,W) grid. Interaction =
directional (1-5,7) + click/mouse (ACTION6 at x,y). Click targets are inferred ONLY from pixels
(small 4-connected components + rare-color cells) -- honest, source-free. board_match_error /
render_diff measure the simulated board against the perceived real board (perception vs reality)."""
import numpy as np


def _background(frame):
    vals, cnts = np.unique(frame, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _components(mask):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not seen[y, x]:
                stack = [(y, x)]; seen[y, x] = True; comp = []
                while stack:
                    cy, cx = stack.pop(); comp.append((cy, cx))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; stack.append((ny, nx))
                comps.append(comp)
    return comps


def infer_click_targets(frame, max_size=40, rare_frac=0.02):
    frame = np.asarray(frame); bg = _background(frame)
    targets = set()
    for comp in _components(frame != bg):
        if len(comp) <= max_size:
            targets.update(comp)
    vals, cnts = np.unique(frame, return_counts=True); tot = frame.size
    rare = {int(v) for v, c in zip(vals, cnts) if int(v) != bg and c / tot <= rare_frac}
    if rare:
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                if int(frame[y, x]) in rare:
                    targets.add((y, x))
    return sorted(targets)


def board_match_error(pred, real):
    pred = np.asarray(pred); real = np.asarray(real)
    if pred.shape != real.shape:
        return {"cells_total": int(real.size), "cells_wrong": int(real.size),
                "exact": False, "error_map": np.ones(real.shape, dtype=bool)}
    diff = pred != real
    return {"cells_total": int(real.size), "cells_wrong": int(diff.sum()),
            "exact": bool(not diff.any()), "error_map": diff}


def render_diff(pred, real):
    e = board_match_error(pred, real); em = e["error_map"]
    lines = [f"board-match: {e['cells_total'] - e['cells_wrong']}/{e['cells_total']} cells, exact={e['exact']}"]
    for y in range(em.shape[0]):
        lines.append("".join("X" if em[y, x] else "." for x in range(em.shape[1])))
    return "\n".join(lines)


def candidate_actions(frame, avail):
    acts = []
    for k in avail:
        if int(k) == 6:
            for (y, x) in infer_click_targets(frame):
                acts.append((6, int(x), int(y)))
        else:
            acts.append((int(k), None, None))
    return acts
