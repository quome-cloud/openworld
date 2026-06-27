import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import execute, verify

class RealGame:
    """action [1] increments (0,0) until 3 then a level-up; BUT at (0,0)==2 the real board ALSO sets (1,1)=7
    (a dynamic the model below doesn't know) -> a sim-vs-real mismatch the executor must catch."""
    def __init__(self): self.reset()
    def reset(self): self.c=0; self.levels=0; self.done=False; self.frame=np.zeros((64,64),dtype=int)
    def step(self,a,x=None,y=None):
        if a==1: self.c+=1
        self.frame=np.zeros((64,64),dtype=int); self.frame[0,0]=self.c
        if self.c==2: self.frame[1,1]=7
        if self.c==3: self.levels=1; self.done=True

# model: increments (0,0), level_up at 3, but NEVER sets (1,1) -> mismatch at step where c becomes 2
PRED="def predict(frame, action):\n    nf=frame.copy()\n    if action==[1]: nf[0,0]=frame[0,0]+1\n    return nf, bool(nf[0,0]==3)"
fn=verify.compile_predict(PRED)

def test_execute_halts_on_mismatch_and_records_transition():
    r = execute.execute_plan(RealGame(), [[1],[1],[1]], fn, mask=None)
    assert r["solved"] is False
    assert r["halt_step"] == 2                       # mismatch when c becomes 2 (real sets (1,1)=7)
    assert len(r["new_transitions"]) == 1
    assert r["new_transitions"][0]["next_frame"][1,1] == 7

def test_execute_solves_when_model_matches():
    # mask out (1,1) so the unmodeled cell is ignored -> model matches -> plan solves
    mask=np.zeros((64,64),dtype=bool); mask[1,1]=True
    r = execute.execute_plan(RealGame(), [[1],[1],[1]], fn, mask=mask)
    assert r["solved"] is True and r["halt_step"] is None
