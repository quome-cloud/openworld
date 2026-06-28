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
