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

def test_collect_obj_stops_on_level_up():
    """Game that levels up after 1 step of [4]. collect_obj must break immediately after the
    level-up transition so next-level dynamics don't pollute this level's training set.

    Without the break, after g.done→_fresh(), the new (state,[2]) pair is NOT in seen and
    WOULD be appended (len==2). With the break, we stop at len==1."""
    class LevelUpGame:
        def __init__(self): self.reset()
        def reset(self): self.x=1; self.levels=0; self.done=False; self._d()
        def _d(self): self.frame=np.zeros((8,8),dtype=int); self.frame[1,self.x]=3
        def step(self, a, x=None, y=None):
            if a==4: self.x=min(7,self.x+1)
            self._d()
            if self.x >= 2: self.levels=1; self.done=True   # level-up on first step of [4]
    # Two candidates: after level-up+reset the [2] action key is unseen and would be recorded
    # without a break, inflating len(tr) to 2.
    tr = explorer.collect_obj(LevelUpGame, lambda s:[[4],[2]], budget=10, perceive=perc)
    assert len(tr) == 1
    assert tr[-1]["level_up"] is True
