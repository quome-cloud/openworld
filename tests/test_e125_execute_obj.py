# tests/test_e125_execute_obj.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import execute, verify, objstate

# real game: action [4] moves a color-3 size-1 object right along row 1; win at x==4 (col 4).
class RealObjGame:
    def __init__(self): self.reset()
    def reset(self): self.x=1; self.levels=0; self.done=False; self._draw()
    def _draw(self):
        self.frame=np.zeros((8,8),dtype=int); self.frame[1,self.x]=3
    def step(self,a,x=None,y=None):
        if a==4: self.x=min(7,self.x+1)
        self._draw()
        if self.x==4: self.levels=1; self.done=True

PRED = ("def predict(state, action):\n"
        "    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n"
        "    if action==[4]:\n        o=ns['objects'][0]; o['x']=o['x']+1\n"
        "    return ns, bool(ns['objects'][0]['x']==4)")
fn = verify.compile_predict(PRED)
perc = lambda f: objstate.object_state(f)

def test_execute_obj_solves_on_real_levelup():
    r = execute.execute_obj(RealObjGame(), [[4],[4],[4]], fn, perc)
    assert r["solved"] is True and r["halt_step"] is None and r["verified_prefix"] == [[4],[4],[4]]

def test_execute_obj_halts_on_surprise():
    # model says [4] moves +1 but real game also has a wall: redefine a game where [4] does NOTHING -> surprise
    class Stuck(RealObjGame):
        def step(self,a,x=None,y=None): self._draw()   # never moves
    r = execute.execute_obj(Stuck(), [[4]], fn, perc)
    assert r["solved"] is False and r["halt_step"] == 1 and len(r["new_transitions"]) == 1
    assert r["new_transitions"][0]["level_up"] is False

def test_execute_obj_halts_on_refuted_win():
    # Real game: step redraws only -- never moves, never levels up.
    # Predict returns the SAME state (keys agree) but level_up=True -> refuted-win branch.
    class StationaryGame(RealObjGame):
        def step(self, a, x=None, y=None): self._draw()   # never moves, never levels up
    PRED_WIN = ("def predict(state, action):\n"
                "    return {'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}, True")
    fn_win = verify.compile_predict(PRED_WIN)
    r = execute.execute_obj(StationaryGame(), [[4]], fn_win, perc)
    assert r["solved"] is False
    assert r["halt_step"] == 1
    assert len(r["new_transitions"]) == 1
    assert r["new_transitions"][0]["level_up"] is False
