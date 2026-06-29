"""E108 -- salience-guided frontier NAVIGATION (paper #2's actual method, efficient form).

E107 did frontier-biased *rollouts*. Paper #2's key levers are (a) **shortest-path-to-frontier**
navigation -- deliberately go to the nearest UNTESTED (state,action) pair, not just stumble onto it --
and (b) **visual-salience** prioritization -- expand states reached by high-change transitions first.
Here each graph node stores its shortest action-path from reset (our determinism makes replay exact);
the frontier is a salience-ordered priority queue; we pop the most-salient frontier node, replay to it
(bounded), take an untested action, and enqueue the result with salience = cells changed. Chains levels.

  python3 e108_salience_explore.py --budget 4000
"""
import argparse, json, logging, contextlib, io, heapq
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]

def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)
def sig(f): return hash(np.asarray(f).tobytes())
def all_games():
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        envs=arc_agi.Arcade().available_environments
    return sorted({(e if isinstance(e,str) else getattr(e,"game_id",str(e))).split("-")[0] for e in envs})

def explore(game, budget, max_depth=90, max_states=8000):
    arc=arc_agi.Arcade()
    def fresh():
        e=arc.make(game); ob=e.reset(); return e,ob
    e,o=fresh(); avail0=list(o.available_actions); win=o.win_levels; base0=o.levels_completed
    s0=sig(grid(o))
    node={s0:{"path":[],"unt":list(avail0),"av":list(avail0)}}
    # priority queue: (-salience, depth, sig) -- most-salient, then shallowest first
    pq=[(0,0,s0)]; best=base0; solution=None; steps=0; expansions=0
    def replay(path):
        ee,ob=fresh()
        for a in path:
            ob=ee.step(ACTS[a-1])
            if ob is None or getattr(ob,"frame",None) is None: return None,None
        return ee,ob
    while pq and steps<budget and len(node)<max_states:
        _,depth,s=heapq.heappop(pq)
        nd=node.get(s)
        if not nd or not nd["unt"]: continue
        ee,ob=replay(nd["path"])
        if ob is None: continue
        before=grid(ob); base=ob.levels_completed
        # take ALL untested actions from this state (each is one frontier (state,action) pair)
        while nd["unt"] and steps<budget:
            a=nd["unt"].pop(0)
            ee2,ob2=replay(nd["path"]) if False else (ee,ob)   # reuse env, but actions mutate it
            nob=ee.step(ACTS[a-1]); steps+=1; expansions+=1
            if nob is None or getattr(nob,"frame",None) is None:
                ee,ob=replay(nd["path"]);                          # restore to state s
                if ob is None: break
                continue
            nf=grid(nob); lv=nob.levels_completed; ns=sig(nf)
            chg=int((before!=nf).sum())
            if lv>best: best=lv; solution=nd["path"]+[a]
            if ns not in node and str(nob.state)=="GameState.NOT_FINISHED":
                av=list(nob.available_actions)
                node[ns]={"path":nd["path"]+[a],"unt":list(av),"av":av}
                heapq.heappush(pq,(-chg, depth+1, ns))            # SALIENCE = cells changed
            # restore env to state s for the next untested action
            ee,ob=replay(nd["path"])
            if ob is None: break
    return {"game":game,"best_levels":int(best),"win_levels":int(win),"states":len(node),"steps":steps,
            "solution":solution,"verified":solution is not None}

def verify(game, seq):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); base=o.levels_completed
    for a in seq:
        o=env.step(ACTS[a-1])
        if o is None or getattr(o,"frame",None) is None: return False
        if o.levels_completed>base: return True
    return False

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=int,default=4000); ap.add_argument("--games",default=""); ap.add_argument("--out",default="results/e108_salience_explore.json"); a=ap.parse_args()
    games=a.games.split(",") if a.games else all_games()
    print(f"[e108] salience-guided frontier navigation on {len(games)} games (budget {a.budget})",flush=True); res={}
    for g in games:
        try:
            r=explore(g,a.budget)
            if r["solution"]: r["verified"]=verify(g,r["solution"])
        except Exception as e: r={"game":g,"verified":False,"error":str(e)[:90]}
        res[g]=r
        print(f"  {g}: best {r.get('best_levels',0)}/{r.get('win_levels','?')} states={r.get('states','-')} {'SOLVED' if r.get('verified') else ''}",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    Path(a.out).write_text(json.dumps({"solved":solved,"n_solved":len(solved),"results":res},indent=2))
    print(f"[e108] SALIENCE-EXPLORE SOLVED {len(solved)}/{len(games)}: {sorted(solved)}",flush=True)

if __name__=="__main__": main()
