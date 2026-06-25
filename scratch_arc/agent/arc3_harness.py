"""ARC-AGI-3 harness for a coding agent. Deterministic env -> replay verifies.
Usage:
    from arc3_harness import Game
    g = Game("ka59")
    g.reset()                      # -> frame (64x64 numpy int array)
    g.step(1)                      # directional action 1..5 or 7
    g.step(6, x, y)                # ACTION6 is a CLICK at column x, row y (0..63)
    g.frame, g.levels, g.win, g.avail, g.done
Notes (hard-won):
  * available_actions tells the modality: [1..5,7]=directional, [6]=click-only, mixed=both.
  * Clicks only register on certain cells (sprites). Try clicking distinct/non-background cells.
  * A status counter cell changes every step; ignore it when comparing frames.
  * Deterministic: replaying actions from reset() reproduces frames -> verify by replay.
"""
import json, sys, numpy as np, arc_agi
from arcengine import GameAction
_A={1:GameAction.ACTION1,2:GameAction.ACTION2,3:GameAction.ACTION3,4:GameAction.ACTION4,5:GameAction.ACTION5,7:GameAction.ACTION7}
def _g(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)
class Game:
    def __init__(self, gid):
        self.arc=arc_agi.Arcade(); self.gid=gid; self.env=self.arc.make(gid); self.reset()
    def reset(self):
        o=self.env.reset(); self.avail=list(o.available_actions); self.levels=o.levels_completed
        self.win=o.win_levels; self.frame=_g(o); self.done=False; return self.frame
    def step(self, action, x=None, y=None):
        o=self.env.step(GameAction.ACTION6,{"x":int(x),"y":int(y)}) if action==6 else self.env.step(_A[action])
        if o is None or getattr(o,"frame",None) is None: self.done=True; return self.frame
        self.frame=_g(o); self.levels=o.levels_completed; self.done=str(o.state)!="GameState.NOT_FINISHED"
        return self.frame
def replay(gid, steps):
    """steps: list of [action] or [6,x,y]. Returns levels completed delta."""
    g=Game(gid); base=g.levels; mx=base
    for s in steps:
        g.step(*s) if isinstance(s,(list,tuple)) else g.step(s); mx=max(mx,g.levels)
        if g.done: break
    return mx-base
if __name__=="__main__":
    print(json.dumps({"levels_delta": replay(sys.argv[1], json.loads(sys.argv[2]))}))
