import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e125 import agent, verify, objstate

# 2-level game: level 0 win at x==3 (move [4]); level 1 win at x==6 (keep moving [4]). object color 3 row 1.
class TwoLevel:
    def __init__(self): self.reset()
    def reset(self): self.x=1; self.levels=0; self.done=False; self._d()
    def _d(self): self.frame=np.zeros((8,8),dtype=int); self.frame[1,self.x]=3
    def step(self,a,x=None,y=None):
        if a==4: self.x=min(7,self.x+1)
        self._d()
        if self.levels==0 and self.x==3: self.levels=1
        elif self.levels==1 and self.x==6: self.levels=2; self.done=True

PRED=("def predict(state, action):\n    ns={'bg':state['bg'],'objects':[dict(o) for o in state['objects']]}\n"
      "    if action==[4]:\n        ns['objects'][0]['x']=ns['objects'][0]['x']+1\n"
      "    return ns, False")   # dynamics correct; win discovered by the env oracle, not the model
GOAL="def goal_score(state):\n    return float(9 - state['objects'][0]['x'])"
perc=lambda f: objstate.object_state(f)

def _synth(transitions, action_api, game, seed_src=None, **kw):
    fn=verify.compile_obj_predict(PRED); goal=verify.compile_goal(GOAL)
    return PRED, fn, goal, [fn]

def _macro(prompt, schema, model, game, **kw):
    return {"final":{"macro":[[4],[4],[4]],"rationale":"right","goal_note":"x up"},
            "events":[],"tainted":False,"raw":"","model_version":""}

def test_solve_game_solves_two_levels_with_transfer():
    r = agent.solve_game(TwoLevel, lambda s:[[4],[2]], "actions=[2,4]", "g", _synth, perceive=perc,
                         macro_runner=_macro, budget_explore=6, budget_plan=50, rounds_per_level=3, max_levels=2)
    assert r["levels_solved"] == 2
    assert r["real_actions"] > 0 and len(r["levels"]) >= 2
