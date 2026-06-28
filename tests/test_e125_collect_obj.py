import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import explorer, objstate

class G:
    def __init__(self): self.reset()
    def reset(self): self.x=1; self.levels=0; self.done=False; self._d()
    def _d(self): self.frame=np.zeros((8,8),dtype=int); self.frame[1,self.x]=3
    def step(self,a,x=None,y=None):
        if a==4: self.x=min(7,self.x+1)
        self._d()
perc = lambda f: objstate.object_state(f)

def test_collect_obj_records_object_transitions():
    tr = explorer.collect_obj(G, lambda s:[[4]], budget=3, perceive=perc)
    assert len(tr) == 3
    assert tr[0]["state"]["objects"][0]["x"] == 1 and tr[0]["next_state"]["objects"][0]["x"] == 2
    assert all(set(t) == {"state","action","next_state","level_up"} for t in tr)

def test_collect_obj_replays_prefix():
    tr = explorer.collect_obj(G, lambda s:[[4]], budget=1, perceive=perc, prefix=[[4],[4]])
    assert tr[0]["state"]["objects"][0]["x"] == 3   # started after 2x [4]
