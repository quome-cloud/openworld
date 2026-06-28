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
    synth_fn = lambda transitions, action_api, game, mask, **kw: (TRUE, verify.compile_predict(TRUE), None)
    r = agent.solve_level(Deep, lambda fr:[[1],[2]], "actions=[1,2]", "deep", mask=None,
                          synth_fn=synth_fn, budget_explore=20, budget_plan=5000, rounds=3)
    assert r["solved"] is True
    assert r["actions"] == [[1],[1],[1],[1],[1],[1]]


class Deep10:
    """win at counter==10 (frame[0,0]); [1]=+1, [2]=-1, [3]/[4]=noise cells that blow up a blind frontier.
    A blind plan budget can't reach depth 10 with branching 4 -- but goal_score energy descent walks to it."""
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

def test_solve_level_uses_goal_score_for_energy_descent():
    """The agent threads goal_score into plan-in-sim so a deep win (blind budget can't reach) is solved."""
    synth_fn = lambda t, api, game, mask, **kw: (ENERGY, verify.compile_predict(ENERGY), verify.compile_goal(GOALE))
    r = agent.solve_level(Deep10, lambda fr:[[1],[2],[3],[4]], "actions=[1,2,3,4]", "deep10", mask=None,
                          synth_fn=synth_fn, budget_explore=12, budget_plan=80, rounds=2)
    assert r["solved"] is True
    assert r["actions"] == [[1]]*10


# win hypothesis WRONG (level_up never fires) until a real level-up is GROUNDED in the transitions, then CORRECT
WRONG = ENERGY.replace("return nf, bool(c==10)", "return nf, False")

def test_solve_level_grounds_win_via_goal_directed_exploration():
    """When plan-in-sim finds no win (wrong offline hypothesis), the agent does GOAL-DIRECTED real-env
    exploration to ground a real level-up (the online oracle), re-synthesizes, and then solves -- replicating
    the restartable sweep agent's win-grounding while keeping the verified world model."""
    def synth_fn(transitions, api, game, mask, **kw):
        grounded = any(t["level_up"] for t in transitions)
        src = ENERGY if grounded else WRONG
        return src, verify.compile_predict(src), verify.compile_goal(GOALE)
    r = agent.solve_level(Deep10, lambda fr:[[1],[2],[3],[4]], "actions=[1,2,3,4]", "deep10", mask=None,
                          synth_fn=synth_fn, budget_explore=12, budget_plan=400, rounds=4)
    assert r["solved"] is True
    assert r["actions"] == [[1]]*10


def test_solve_level_seeds_next_round_with_prior_program():
    """Within a level, each re-synth is SEEDED with the prior round's program (carry-forward, not cold restart)."""
    seeds = []
    def synth_fn(transitions, api, game, mask, **kw):
        seeds.append(kw.get("seed_src"))
        grounded = any(t["level_up"] for t in transitions)
        src = ENERGY if grounded else WRONG
        return src, verify.compile_predict(src), verify.compile_goal(GOALE)
    agent.solve_level(Deep10, lambda fr:[[1],[2],[3],[4]], "a", "deep10", mask=None, synth_fn=synth_fn,
                      budget_explore=12, budget_plan=400, rounds=4)
    assert seeds[0] is None                      # first round: cold (no prior program)
    assert any(s for s in seeds[1:])             # later rounds seeded with the carried-forward program
