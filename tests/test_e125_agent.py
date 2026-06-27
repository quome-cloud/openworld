import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import agent, verify

class Deep:
    """level-up only after [1]x6 (depth 6); frame[0,0]=count of 1s. Real env BFS to depth 6 over {1,2} is
    expensive, but planning in the synthesized model is instant."""
    def __init__(self): self.reset()
    def reset(self): self.c=0; self.levels=0; self.done=False; self.frame=np.zeros((64,64),dtype=int)
    def step(self,a,x=None,y=None):
        if a==1: self.c+=1
        else: self.c=0
        self.frame=np.zeros((64,64),dtype=int); self.frame[0,0]=self.c
        if self.c==6: self.levels=1; self.done=True
        if self.c>8: self.done=True

TRUE="def predict(frame, action):\n    nf=frame.copy()\n    nf[0,0]=frame[0,0]+1 if action==[1] else 0\n    return nf, bool((frame[0,0]+1 if action==[1] else 0)==6)"

def test_solve_level_via_plan_in_sim():
    synth_fn = lambda transitions, action_api, game, mask, **kw: (TRUE, verify.compile_predict(TRUE))
    r = agent.solve_level(Deep, lambda fr:[[1],[2]], "actions=[1,2]", "deep", mask=None,
                          synth_fn=synth_fn, budget_explore=20, budget_plan=5000, rounds=3)
    assert r["solved"] is True
    assert r["actions"] == [[1],[1],[1],[1],[1],[1]]
