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
