import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import explorer, verify

# counter c=frame[0,0]; [1]=+1,[2]=-1,[3]/[4]=noise; REAL level-up at c==10. goal_score=10-c (energy descends).
class Deep10:
    def __init__(self): self.reset()
    def reset(self):
        self.c=0; self.n5=0; self.n6=0; self.levels=0; self.done=False; self.frame=np.zeros((64,64),dtype=int)
    def step(self,a,x=None,y=None):
        if a==1: self.c=min(10,self.c+1)
        elif a==2: self.c=max(0,self.c-1)
        elif a==3: self.n5=(self.n5+1)%9
        elif a==4: self.n6=(self.n6+1)%9
        self.frame=np.zeros((64,64),dtype=int); self.frame[0,0]=self.c; self.frame[5,5]=self.n5; self.frame[6,6]=self.n6
        if self.c==10: self.levels=1; self.done=True

ENERGY = ("def predict(frame, action):\n"
          "    nf=frame.copy(); c=int(frame[0,0])\n"
          "    if action==[1]: c=min(10,c+1)\n"
          "    elif action==[2]: c=max(0,c-1)\n"
          "    elif action==[3]: nf[5,5]=(int(frame[5,5])+1)%9; return nf,False\n"
          "    elif action==[4]: nf[6,6]=(int(frame[6,6])+1)%9; return nf,False\n"
          "    nf[0,0]=c\n    return nf, bool(c==10)")
GOALE = "def goal_score(frame):\n    return float(10 - int(frame[0,0]))"


def test_goal_directed_collect_grounds_a_real_levelup():
    """Descending the goal_score energy in the REAL env reaches and GROUNDS a real level-up (the g.levels
    oracle), which blind round-robin exploration does not."""
    fn = verify.compile_predict(ENERGY); goal = verify.compile_goal(GOALE)
    trans = explorer.goal_directed_collect(Deep10, lambda fr: [[1],[2],[3],[4]], fn, goal, budget=40)
    assert any(t["level_up"] for t in trans)                # reached & grounded a real win
    assert trans[-1]["level_up"] is True                    # stops on the grounding level-up

def test_round_robin_collect_does_not_reach_levelup():
    """Contrast: the existing change-seeking round-robin collect never grounds the win (the M2 bottleneck)."""
    trans = explorer.collect(Deep10, lambda fr: [[1],[2],[3],[4]], budget=40)
    assert not any(t["level_up"] for t in trans)

def test_goal_directed_collect_no_goal_still_explores():
    """With no goal hypothesis (goal_fn=None) it degrades to plain exploration (records transitions, no crash)."""
    fn = verify.compile_predict(ENERGY)
    trans = explorer.goal_directed_collect(Deep10, lambda fr: [[1],[2],[3],[4]], fn, None, budget=8)
    assert len(trans) > 0
