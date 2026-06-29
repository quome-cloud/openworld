"""E117 -- random-play baseline under the SAME offline/unbounded-reset protocol as our solvers, to
contextualize the >=1-level claim (reviewer control): how many games does undirected play solve?
Random selects a uniform action each step (directional games: from available; click games: a uniform
(x,y) click; mixed: both), resets on episode end, counts >=1-level within budget."""
import argparse, json, logging, contextlib, io, random
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
SIMPLE={1:GameAction.ACTION1,2:GameAction.ACTION2,3:GameAction.ACTION3,4:GameAction.ACTION4,5:GameAction.ACTION5,7:GameAction.ACTION7}
def all_games():
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        envs=arc_agi.Arcade().available_environments
    return sorted({(e if isinstance(e,str) else getattr(e,"game_id",str(e))).split("-")[0] for e in envs})
def run(game, budget, seed=0):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); avail=list(o.available_actions); win=o.win_levels; base=o.levels_completed
    rng=random.Random(seed); best=base; steps=0
    dirs=[a for a in avail if a in SIMPLE]; click=6 in avail
    while steps<budget:
        o=env.reset(); lvl=o.levels_completed; d=0
        while d<300 and steps<budget:
            if click and (not dirs or rng.random()<0.5):
                o=env.step(GameAction.ACTION6,{"x":rng.randint(0,63),"y":rng.randint(0,63)})
            else:
                o=env.step(SIMPLE[rng.choice(dirs)])
            steps+=1; d+=1
            if o is None or getattr(o,"frame",None) is None: break
            best=max(best,o.levels_completed)
            if str(o.state)!="GameState.NOT_FINISHED": break
        if best>=win: break
    return {"game":game,"random_levels":int(best-base),"win":int(win),"click":click}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=int,default=100000); ap.add_argument("--out",default="results/e117_random_baseline.json"); a=ap.parse_args()
    games=all_games(); print(f"[e117] random baseline on {len(games)} games (budget {a.budget})",flush=True); res={}
    for g in games:
        try: r=run(g,a.budget)
        except Exception as e: r={"game":g,"random_levels":0,"error":str(e)[:60]}
        res[g]=r; print(f"  {g}: random reaches {r.get('random_levels',0)} levels",flush=True)
    solved=[g for g,r in res.items() if r.get("random_levels",0)>=1]
    Path(a.out).write_text(json.dumps({"random_geq1_level":sorted(solved),"n":len(solved),
        "budget_steps":a.budget,"max_depth_per_episode":300,"metric":"reaches >=1 (first) level",
        "results":res},indent=2))
    print(f"[e117] RANDOM solves >=1 level on {len(solved)}/{len(games)}: {sorted(solved)}",flush=True)
if __name__=="__main__": main()
