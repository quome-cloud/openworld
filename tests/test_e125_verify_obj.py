import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from e125 import verify

def _t(s, a, ns, lu): return {"state": s, "action": a, "next_state": ns, "level_up": lu}

S0 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 1}]}
S1 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 2}]}   # moved x:1->2
S2 = {"bg": 0, "objects": [{"color": 3, "size": 1, "y": 1, "x": 3}]}
TR = [_t(S0, [4], S1, False), _t(S1, [4], S2, False)]

# correct: action [4] increments object x by 1 (decision-relevant move); win never
GOODP = ("def predict(state, action):\n"
         "    ns = {'bg': state['bg'], 'objects': [dict(o) for o in state['objects']]}\n"
         "    if action == [4]:\n"
         "        for o in ns['objects']: o['x'] += 1\n"
         "    return ns, False")
# cosmetically different but decision-equivalent: also bumps 'size' (NOT a decision field) -> still passes
COSMETIC = GOODP.replace("o['x'] += 1", "o['x'] += 1; o['size'] += 9")
# wrong on a decision field (x): mispredicts -> fails
BADP = ("def predict(state, action):\n    return {'bg': state['bg'], 'objects': [dict(o) for o in state['objects']]}, False")

def test_check_obj_accepts_decision_correct():
    ok, ce = verify.check_obj(verify.compile_predict(GOODP), TR)
    assert ok is True and ce is None

def test_check_obj_ignores_non_decision_fields():
    ok, ce = verify.check_obj(verify.compile_predict(COSMETIC), TR)
    assert ok is True                      # size differs but is not a decision field

def test_check_obj_rejects_decision_wrong():
    ok, ce = verify.check_obj(verify.compile_predict(BADP), TR)
    assert ok is False and ce is not None and ce["action"] == [4]

def test_score_obj_counts_matches():
    n, fails = verify.score_obj(verify.compile_predict(GOODP), TR)
    assert n == 2 and fails == []
    n, fails = verify.score_obj(verify.compile_predict(BADP), TR)
    assert n == 0 and len(fails) == 2
