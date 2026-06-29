import numpy as np
from e119 import proxy_probe


def _frame(cells):
    """64x64 grid; cells = {(r,c): color}. Background 0."""
    g = np.zeros((64, 64), int)
    for (r, c), v in cells.items():
        g[r, c] = v
    return g


def test_enumerate_covers_present_colors_and_kinds():
    frames = [_frame({(0, 0): 4, (1, 1): 4, (2, 2): 7})]
    preds = proxy_probe.enumerate_predicates(frames)
    kinds = {p["type"] for p in preds}
    assert kinds == {"reach", "count", "align"}
    reach_colors = {p["color"] for p in preds if p["type"] == "reach"}
    assert reach_colors == {0, 4, 7}                      # every observed color
    assert any(p["type"] == "align" and p["a"] == 4 and p["b"] == 7 for p in preds)


def test_scan_satisfiable_filters_to_true_on_some_frame():
    frames = [_frame({(0, 0): 4}), _frame({(0, 0): 4, (0, 1): 4})]  # color 4 count is 1 then 2
    preds = proxy_probe.enumerate_predicates(frames)
    sat = proxy_probe.scan_satisfiable(preds, frames)
    assert {"type": "reach", "color": 4} in sat
    assert {"type": "count", "color": 4, "op": "==", "k": 2} in sat   # true on frame 2
    # a count that is never observed is not satisfiable
    assert {"type": "count", "color": 4, "op": "==", "k": 5} not in sat
