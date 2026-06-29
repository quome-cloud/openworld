# tests/test_e125_fallback.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import synth

F0 = np.zeros((64,64), dtype=int); F1 = F0.copy(); F1[0,0]=1; F2=F1.copy(); F2[0,0]=2
def _t(f,a,nf,lu): return {"frame":f,"action":a,"next_frame":nf,"level_up":lu}
TRANS = [_t(F0,[1],F1,False), _t(F1,[1],F2,False)]
GOOD = "def predict(frame, action):\n    nf=frame.copy(); nf[0,0]=frame[0,0]+1\n    return nf, bool(nf[0,0]==3)"
GOAL = "def goal_score(frame):\n    return float(3 - frame[0,0])"
STUCK = "def predict(frame, action):\n    return frame.copy(), False"   # scores 0, never improves

def _runner(src, tag, log=None):
    def run(prompt, schema, model, game, **kw):
        if log is not None: log.append(tag)
        return {"final": {"predict_src": src, "goal_score_src": GOAL, "rationale": ""},
                "events": [], "tainted": False, "raw": "", "model_version": tag}
    return run

def test_no_fallback_when_not_provided():
    log = []
    synth.synthesize(TRANS, "a", "g", mask=None, n_retries=3, _runner=_runner(STUCK, "primary", log))
    assert set(log) == {"primary"}              # only the primary runner is ever called

def test_switches_to_fallback_after_stall():
    log = []
    primary = _runner(STUCK, "primary", log)
    fallback = _runner(GOOD, "claude", log)
    src, fn, goal = synth.synthesize(TRANS, "a", "g", mask=None, n_retries=6, _runner=primary,
                                     fallback_runner=fallback, stall_window=2)
    assert "claude" in log                       # stalled on primary -> Claude was called
    assert fn is not None                         # Claude's GOOD model passed the gate
    assert log[:2] == ["primary", "primary"]      # first stall_window attempts use the primary

def test_fallback_resets_counter_on_improvement():
    # primary improves on attempt 1 (GOOD), so the stall counter resets and Claude is never needed
    log = []
    src, fn, goal = synth.synthesize(TRANS, "a", "g", mask=None, n_retries=4, _runner=_runner(GOOD, "primary", log),
                                     fallback_runner=_runner(GOOD, "claude", log), stall_window=2)
    assert "claude" not in log and fn is not None

def test_synthesize_obj_also_supports_fallback():
    S0={"bg":0,"objects":[{"color":3,"size":1,"y":1,"x":1}]}; S1={"bg":0,"objects":[{"color":3,"size":1,"y":1,"x":2}]}
    S2={"bg":0,"objects":[{"color":3,"size":1,"y":1,"x":3}]}
    TRO=[{"state":S0,"action":[4],"next_state":S1,"level_up":False},
         {"state":S1,"action":[4],"next_state":S2,"level_up":False}]
    GOODO=("def predict(state, action):\n    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n"
           "    if action==[4]:\n        [o.__setitem__('x',o['x']+1) for o in ns['objects']]\n    return ns, False")
    STUCKO="def predict(state, action):\n    return state, False"
    log=[]
    r = synth.synthesize_obj(TRO, "a", "g", n_retries=6, _runner=_runner(STUCKO,"primary",log),
                             fallback_runner=_runner(GOODO,"claude",log), stall_window=2)
    assert "claude" in log and r[1] is not None    # (src, predict_fn, goal_fn, ensemble)
