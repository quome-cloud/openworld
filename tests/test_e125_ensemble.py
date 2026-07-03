import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import synth, verify

S0 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 1}]}
MOVE = "def predict(state, action):\n    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n    [o.__setitem__('x',o['x']+1) for o in ns['objects']]\n    return ns, False"
STAY = "def predict(state, action):\n    return {'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}, False"
BOOM = "def predict(state, action):\n    raise ValueError('x')"

def test_top_k_returns_highest_scoring_distinct():
    db = synth._Database(rng=np.random.RandomState(0))
    db.register("a", None, 1, (True, False), [], None)
    db.register("b", None, 3, (True, True), [], None)
    db.register("a", None, 1, (True, False), [], None)   # dup src
    top = db.top_k(2)
    assert [p["score"] for p in top] == [3, 1] and len({p["src"] for p in top}) == 2

def test_disagreement_zero_when_all_agree():
    fns = [verify.compile_obj_predict(MOVE), verify.compile_obj_predict(MOVE)]
    assert synth.ensemble_disagreement(fns, S0, [4]) == 0.0

def test_disagreement_positive_when_split():
    fns = [verify.compile_obj_predict(MOVE), verify.compile_obj_predict(STAY), verify.compile_obj_predict(BOOM)]
    d = synth.ensemble_disagreement(fns, S0, [4])
    assert d > 0.0

def test_disagreement_single_fn_is_zero():
    assert synth.ensemble_disagreement([verify.compile_obj_predict(MOVE)], S0, [4]) == 0.0
