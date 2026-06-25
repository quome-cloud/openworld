"""Clean ARC-AGI-3 harness for an agent: drive a game, render frames, replay-verify a solution.
Run with the arc-agi venv python. Deterministic env -> a saved action list replays to the same result."""
import json, numpy as np, arc_agi
from arcengine import GameAction
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]
def _g(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)
class Game:
    def __init__(self, gid):
        self.arc=arc_agi.Arcade(); self.gid=gid; self.env=self.arc.make(gid); self.reset()
    def reset(self):
        o=self.env.reset(); self.avail=list(o.available_actions); self.levels=o.levels_completed; self.win=o.win_levels; self.frame=_g(o); self.done=False; return self.frame
    def step(self, a):       # a in 1..7
        o=self.env.step(ACTS[a-1])
        if o is None or getattr(o,"frame",None) is None: self.done=True; return self.frame, self.levels, "BADSTEP"
        self.frame=_g(o); self.levels=o.levels_completed; self.done=str(o.state)!="GameState.NOT_FINISHED"
        return self.frame, self.levels, str(o.state)
def replay(gid, seq):        # verify a candidate solution against a fresh game
    g=Game(gid); base=g.levels; reached=base
    for a in seq:
        _,lv,st=g.step(a); reached=max(reached,lv)
        if st=="BADSTEP": break
    return {"start_levels":base,"reached_levels":reached,"win_levels":g.win,"completed_a_level":reached>base}
if __name__=="__main__":
    import sys; print(json.dumps(replay(sys.argv[1], json.loads(sys.argv[2]))))
