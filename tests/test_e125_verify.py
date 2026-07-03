import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import verify

def _t(f, a, nf, lu): return {"frame": f, "action": a, "next_frame": nf, "level_up": lu}

# a toy world: action [1] sets cell (0,0)=action count (frame[0,0]+1); level_up when it hits 3
F0 = np.zeros((64,64), dtype=int)
F1 = F0.copy(); F1[0,0] = 1
F2 = F1.copy(); F2[0,0] = 2
GOOD = "def predict(frame, action):\n    nf = frame.copy(); nf[0,0] = frame[0,0] + 1\n    return nf, bool(nf[0,0] == 3)"
BAD  = "def predict(frame, action):\n    return frame.copy(), False"   # never changes -> mispredicts

TRANS = [_t(F0,[1],F1,False), _t(F1,[1],F2,False)]

def test_compile_predict_returns_callable():
    fn = verify.compile_predict(GOOD); assert callable(fn)
    nf, lu = fn(F0, [1]); assert nf[0,0] == 1 and lu is False

def test_compile_predict_bad_src_returns_none():
    assert verify.compile_predict("def predict(:\n bad") is None

def test_check_accepts_exact_model():
    ok, ce = verify.check(verify.compile_predict(GOOD), TRANS, mask=None)
    assert ok is True and ce is None

def test_check_rejects_with_counterexample():
    ok, ce = verify.check(verify.compile_predict(BAD), TRANS, mask=None)
    assert ok is False and ce is not None and ce["action"] == [1]

def test_check_masks_status_bar():
    # a status cell at (63,63) flips every step; with it masked, an otherwise-correct model passes
    a = F0.copy(); a[63,63] = 5; b = F1.copy(); b[63,63] = 9
    mask = np.zeros((64,64), dtype=bool); mask[63,63] = True
    ok, _ = verify.check(verify.compile_predict(GOOD), [_t(a,[1],b,False)], mask=mask)
    assert ok is True


# --- goal_score (energy) compilation: a SYMBOLIC heuristic, not gated against data ---
GOAL = "def goal_score(frame):\n    return float(3 - frame[0,0])"   # energy: 0 at the goal (frame[0,0]==3)

def test_compile_goal_returns_callable():
    g = verify.compile_goal(GOAL); assert callable(g)
    assert g(F0) == 3.0 and g(F2) == 1.0

def test_compile_goal_bad_src_returns_none():
    assert verify.compile_goal("def goal_score(:\n bad") is None

def test_compile_goal_missing_fn_returns_none():
    assert verify.compile_goal("x = 1") is None
