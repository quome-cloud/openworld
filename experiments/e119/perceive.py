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
