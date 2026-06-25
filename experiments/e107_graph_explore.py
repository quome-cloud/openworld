"""E107 -- deterministic STATE-GRAPH FRONTIER EXPLORATION (paper #2's method, in OpenWorld terms).

Bypasses goal inference (our wall) the way the graph-exploration SOTA does: segment each frame into
an object-graph signature (our perceptor) = a state node; build the deterministic state-transition
graph; drive to the FRONTIER -- the nearest UNTESTED (state, action) pair -- by replaying the shortest
path to it (our determinism makes this exact); record any level-completion and replay-verify it.
Salience: expand shallow, high-change states first. Continues across levels (chains a full game).
Combines our learnings (object-graph perception, determinism, replay-verify) with the SOTA technique.

  python3 e107_graph_explore.py --budget 6000
"""
import argparse, json, logging, contextlib, io
from collections import deque
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
import arc3_graph as G
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]

def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)
def sig(f):                                   # object-graph signature = state identity (segmentation)
    objs,_=G.objects(f)
    return tuple(sorted((o["color"],o["size"],int(o["centroid"][0])//3,int(o["centroid"][1])//3) for o in objs))
def all_games():
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        envs=arc_agi.Arcade().available_environments
    return sorted({(e if isinstance(e,str) else getattr(e,"game_id",str(e))).split("-")[0] for e in envs})

def explore(game, budget, max_depth=140):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); avail=list(o.available_actions); win=o.win_levels
    s0=sig(grid(o))
    paths={s0:[]}; untested={s0:list(avail)}; salience={s0:0}
    frontier=deque([s0]); best=o.levels_completed; solution=None; steps=0
    def replay(path):
        e=arc.make(game); ob=e.reset()
        for a in path:
            ob=e.step(ACTS[a-1])
            if ob is None or getattr(ob,"frame",None) is None: return None,e
        return ob,e
    while frontier and steps<budget:
        # salience-guided pick: shallowest, then highest recent change
        frontier=deque(sorted(frontier, key=lambda s:(len(paths[s]), -salience.get(s,0))))
        s=frontier[0]
        if not untested.get(s): frontier.popleft(); continue
        if len(paths[s])>=max_depth: untested[s]=[]; frontier.popleft(); continue
        a=untested[s].pop(0)
        ob,e=replay(paths[s])
        if ob is None: continue
        before=grid(ob); base=ob.levels_completed
        nob=e.step(ACTS[a-1]); steps+=1
        if nob is None or getattr(nob,"frame",None) is None: continue
        nf=grid(nob); ns=sig(nf); lvl=nob.levels_completed
        chg=int((before!=nf).sum())
        if lvl>best:                                   # LEVEL COMPLETED via this (state,action)
            best=lvl; solution=paths[s]+[a]
        if ns not in paths and str(nob.state)=="GameState.NOT_FINISHED" and lvl==base:
            paths[ns]=paths[s]+[a]; untested[ns]=list(nob.available_actions); salience[ns]=chg; frontier.append(ns)
        elif lvl>base and ns not in paths:             # advanced a level -> keep exploring the new level
            paths[ns]=paths[s]+[a]; untested[ns]=list(nob.available_actions); salience[ns]=chg; frontier.append(ns)
    return {"game":game,"best_levels":int(best),"win_levels":int(win),"states":len(paths),"steps":steps,
            "solution":solution,"verified":solution is not None}

def verify(game, seq):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); base=o.levels_completed
    for a in seq:
        o=env.step(ACTS[a-1])
        if o is None or getattr(o,"frame",None) is None: return False
        if o.levels_completed>base: return True
    return False

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=int,default=6000); ap.add_argument("--games",default=""); ap.add_argument("--out",default="results/e107_graph_explore.json"); a=ap.parse_args()
    games=a.games.split(",") if a.games else all_games()
    print(f"[e107] state-graph frontier exploration on {len(games)} games (budget {a.budget})",flush=True); res={}
    for g in games:
        try:
            r=explore(g,a.budget)
            if r["solution"]: r["verified"]=verify(g,r["solution"])
        except Exception as e: r={"game":g,"verified":False,"error":str(e)[:90]}
        res[g]=r
        print(f"  {g}: best {r.get('best_levels',0)}/{r.get('win_levels','?')} states={r.get('states','-')} {'SOLVED' if r.get('verified') else ''}",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    Path(a.out).write_text(json.dumps({"solved":solved,"n_solved":len(solved),"results":res},indent=2))
    print(f"[e107] GRAPH-EXPLORE SOLVED {len(solved)}/{len(games)}: {sorted(solved)}",flush=True)

if __name__=="__main__": main()
