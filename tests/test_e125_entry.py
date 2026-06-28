import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
import e125_executable_world as entry


class FlipGame:
    """(0,0) flips EVERY step (freq~1.0); (1,1) flips every OTHER step (freq~0.5) -- lets a test prove the
    mask threshold is honored (0.95 masks only the every-step cell; a lower thr masks both)."""
    def __init__(self): self.reset()
    def reset(self):
        self.t = 0; self.done = False; self.frame = np.zeros((64, 64), dtype=int); return self.frame
    def step(self, a, x=None, y=None):
        self.t += 1
        f = np.zeros((64, 64), dtype=int)
        f[0, 0] = self.t % 2                 # changes every step
        f[1, 1] = (self.t // 2) % 2          # changes every other step
        self.frame = f


def test_probe_mask_default_thr_masks_only_every_step_cell():
    m = entry._probe_mask(FlipGame(), [1], steps=20, thr=0.95)
    assert m[0, 0] and not m[1, 1]

def test_probe_mask_lower_thr_masks_animation_cells():
    m = entry._probe_mask(FlipGame(), [1], steps=20, thr=0.3)
    assert m[0, 0] and m[1, 1]               # the ~0.5-freq animation cell now masked too

def test_probe_mask_leaves_game_reset():
    g = FlipGame(); entry._probe_mask(g, [1], steps=20, thr=0.95)
    assert g.t == 0                          # probe replays from reset and leaves the game clean

def test_candidates_directional():
    cands = entry._candidates_fn([1, 2, 3, 4])
    assert cands(np.zeros((64, 64), dtype=int)) == [[1], [2], [3], [4]]

def test_candidates_mixed_includes_directional_and_clicks():
    """A game with both movement (1-4) and clicks (6) must expose BOTH -- not clicks only."""
    fr = np.zeros((64, 64), dtype=int); fr[10, 10] = 5     # one rare cell -> a click target
    out = entry._candidates_fn([1, 2, 3, 4, 6])(fr)
    assert [1] in out and [2] in out and [3] in out and [4] in out
    assert any(c[0] == 6 for c in out)


# --- _obj_candidates_fn tests (Task 3) ---

def test_obj_candidates_directional():
    cands = entry._obj_candidates_fn([1, 2, 3, 4])
    st = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 2, "x": 5}]}
    assert cands(st) == [[1], [2], [3], [4]]


def test_obj_candidates_includes_clicks_from_objects():
    cands = entry._obj_candidates_fn([1, 2, 3, 4, 6])
    st = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 2, "x": 5}]}
    out = cands(st)
    assert [1] in out and [6, 5, 2] in out          # click target at (x=5,y=2)
