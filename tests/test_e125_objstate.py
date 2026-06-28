import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e125 import objstate

def _grid():
    g = [[0]*8 for _ in range(8)]      # bg=0
    g[1][1] = 3                         # a size-1 object color 3 at (1,1)
    g[5][5] = 3; g[5][6] = 3            # a size-2 object color 3 at (5, 5.5->6)
    g[2][6] = 7                         # a size-1 object color 7 at (2,6)
    return g

def test_object_state_extracts_entities_and_bg():
    s = objstate.object_state(_grid())
    assert s["bg"] == 0
    cols = sorted((o["color"], o["size"]) for o in s["objects"])
    assert cols == [(3, 1), (3, 2), (7, 1)]

def test_object_state_is_canonically_sorted():
    s = objstate.object_state(_grid())
    keys = [(o["color"], o["y"], o["x"]) for o in s["objects"]]
    assert keys == sorted(keys)

def test_state_key_projects_decision_relevant_fields():
    s = objstate.object_state(_grid())
    k = objstate.state_key(s)
    assert k == (0, ((3, 1, 1), (3, 5, 6), (7, 2, 6)))   # (bg, ((color,y,x)...))

def test_ignore_colors_drops_entities():
    s = objstate.object_state(_grid(), ignore_colors=(7,))
    assert all(o["color"] != 7 for o in s["objects"])

def test_perceive_src_matches_object_state():
    ns = {}
    exec(objstate.PERCEIVE_SRC, ns)
    assert ns["perceive"](_grid()) == objstate.object_state(_grid())

def test_perceive_src_sandbox_safe_no_hasattr():
    """PERCEIVE_SRC must work in a sandbox that excludes hasattr (like OpenWorld SAFE_BUILTINS).
    Verifies both a plain 8x8 grid and a (1,8,8)-wrapped grid."""
    import builtins
    safe = {k: getattr(builtins, k) for k in (
        "len", "range", "int", "float", "list", "tuple", "dict", "set",
        "round", "sum", "max", "min", "sorted", "enumerate", "isinstance",
        "map", "abs"
    )}
    ns = {"__builtins__": safe}
    exec(objstate.PERCEIVE_SRC, ns)

    grid = _grid()
    # plain (H, W) grid
    assert ns["perceive"](grid) == objstate.object_state(grid)

    # (1, H, W) wrapped grid — same data with an extra outer dimension
    wrapped = [grid]
    assert ns["perceive"](wrapped) == objstate.object_state(grid)


# --- I2: order-insensitive state_key (TDD: this FAILS before the fix) ---

def test_state_key_order_insensitive():
    """state_key must be identical when same-color objects appear in reversed list order."""
    s_a = {"bg": 0, "objects": [
        {"color": 3, "y": 1, "x": 1},
        {"color": 3, "y": 5, "x": 6},
    ]}
    s_b = {"bg": 0, "objects": [
        {"color": 3, "y": 5, "x": 6},
        {"color": 3, "y": 1, "x": 1},
    ]}
    assert objstate.state_key(s_a) == objstate.state_key(s_b)
