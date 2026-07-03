import numpy as np
from e119 import perceive


def test_status_mask_flags_only_always_changing_cells():
    # cell (0,0) flips every step; everything else constant.
    frames = []
    for t in range(10):
        f = np.zeros((64, 64), int)
        f[0, 0] = t % 2          # changes every step
        f[5, 5] = 7              # constant
        frames.append(f)
    mask = perceive.status_mask(frames, thresh=0.95)
    assert mask.shape == (64, 64)
    assert mask[0, 0] == True
    assert mask[5, 5] == False


def test_state_key_ignores_masked_cells():
    mask = np.zeros((64, 64), bool)
    mask[0, 0] = True
    a = np.zeros((64, 64), int); a[0, 0] = 1
    b = np.zeros((64, 64), int); b[0, 0] = 9   # differs only in masked cell
    assert perceive.state_key(a, mask) == perceive.state_key(b, mask)


def test_object_json_is_relational_to_largest_object():
    f = np.zeros((64, 64), int)
    f[10:14, 10:14] = 3        # big object (agent proxy) size 16
    f[2, 40] = 5               # tiny object size 1
    oj = perceive.object_json(f)
    assert oj["bg"] == 0
    ids = {o["color"]: o for o in oj["objects"]}
    assert 3 in ids and 5 in ids
    # relations are expressed relative to the largest object (id 0 by sort order)
    assert any("of #0" in r for r in oj["relations"])


def test_contrastive_diff_detects_a_move():
    a = np.zeros((64, 64), int); a[10, 10] = 4
    b = np.zeros((64, 64), int); b[10, 11] = 4    # same color moved +1 col
    d = perceive.contrastive_diff(a, b)
    assert d["moved"], "expected one moved object"
    assert d["moved"][0]["color"] == 4


def test_click_candidates_are_small_components_as_xy():
    f = np.zeros((64, 64), int)
    f[0:30, 0:30] = 2          # big region -> NOT a candidate
    f[2, 40] = 5               # tiny sprite -> candidate at (x=40, y=2)
    cands = perceive.click_candidates(f, max_size=40)
    assert (40, 2) in cands
    assert (0, 0) not in cands
