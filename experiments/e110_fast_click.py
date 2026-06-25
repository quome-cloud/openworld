"""E110 -- fast click-capable frontier exploration (proves the click action space unlocks games).
Minimal per-step cost: single env (reset per rollout), masked-frame hash for state identity, non-bg
cells as click targets, frontier bias (prefer untested (state,target)). The OpenWorld ConsensusTransition
voting (E109) layers on top of whatever this cracks; here we isolate the SOTA fix (clicks+masking)."""
import argparse, json, logging, contextlib, io, random
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
SIMPLE={1:GameAction.ACTION1,2:GameAction.ACTION2,3:GameAction.ACTION3,4:GameAction.ACTION4,5:GameAction.ACTION5,7:GameAction.ACTION7}
def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)
def step_action(env,a):
    if isinstance(a,tuple): ga=GameAction.ACTION6; ga.set_data({"x":int(a[1]),"y":int(a[2])}); return env.step(ga)
    return env.step(SIMPLE[a])
def all_games():
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        envs=arc_agi.Arcade().available_environments
    return sorted({(e if isinstance(e,str) else getattr(e,"game_id",str(e))).split("-")[0] for e in envs})
def detect_mask(env,avail,seed,steps=60):
    o=env.reset(); g=grid(o); chg=np.zeros((64,64)); n=0; rng=random.Random(seed); click=6 in avail
    for _ in range(steps):
        a=(6,rng.randint(0,63),rng.randint(0,63)) if click else rng.choice([x for x in avail if x in SIMPLE] or [1])
        no=step_action(env,a)
        if no is None or getattr(no,"frame",None) is None: o=env.reset(); g=grid(o); continue
        ng=grid(no); chg+=(g!=ng); n+=1; g=ng
        if str(no.state)!="GameState.NOT_FINISHED": o=env.reset(); g=grid(o)
    return (chg/max(n,1))>0.85
def explore(game, budget, max_steps=120, seed=0):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); avail=list(o.available_actions); win=o.win_levels; base0=o.levels_completed
    mask=detect_mask(env,avail,seed); click=6 in avail
    g0=grid(o); vals,cnts=np.unique(g0,return_counts=True); bg=int(vals[np.argmax(cnts)])
    tested={}; seen=set(); best=base0; sol=None; steps=0; rng=random.Random(seed)
    while steps<budget:
        o=env.reset(); g=grid(o); seq=[]; d=0
        while d<max_steps and steps<budget:
            gm=g.copy(); gm[mask]=0; h=hash(gm.tobytes()); seen.add(h)
            if click:
                ys,xs=np.where((g!=bg)&(~mask)); tgts=list(zip(xs.tolist(),ys.tolist()))
                if not tgts: tgts=[(x,y) for x in range(4,64,10) for y in range(4,64,10)]
                tgts=[(6,x,y) for (x,y) in tgts[:100]]
            else: tgts=[a for a in avail if a in SIMPLE]
            ts=tested.setdefault(h,set()); unt=[t for t in tgts if (t if isinstance(t,int) else t) not in ts]
            t=rng.choice(unt) if unt else rng.choice(tgts); ts.add(t)
            no=step_action(env,t); steps+=1; d+=1
            if no is None or getattr(no,"frame",None) is None: break
            seq.append(t); g=grid(no)
            if no.levels_completed>best: best=no.levels_completed; sol=list(seq)
            if str(no.state)!="GameState.NOT_FINISHED": break
    return {"game":game,"best_levels":int(best),"win_levels":int(win),"click":click,"states":len(seen),"steps":steps,"solution":sol,"verified":sol is not None}
def verify(game,seq):
    env=arc_agi.Arcade().make(game); o=env.reset(); base=o.levels_completed
    for a in seq:
        o=step_action(env,a)
        if o is None or getattr(o,"frame",None) is None: return False
        if o.levels_completed>base: return True
    return False
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=int,default=150000); ap.add_argument("--games",default=""); ap.add_argument("--out",default="results/e110_fast_click.json"); a=ap.parse_args()
    games=a.games.split(",") if a.games else all_games()
    print(f"[e110] fast click+mask frontier exploration on {len(games)} games",flush=True); res={}
    for g in games:
        try:
            r=explore(g,a.budget)
            if r["solution"]: r["verified"]=verify(g,r["solution"])
        except Exception as e: r={"game":g,"verified":False,"error":str(e)[:90]}
        res[g]=r
        print(f"  {g}: best {r.get('best_levels',0)}/{r.get('win_levels','?')} {'CLICK' if r.get('click') else 'dir'} states={r.get('states','-')} {'SOLVED' if r.get('verified') else ''}",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    Path(a.out).write_text(json.dumps({"solved":solved,"n_solved":len(solved),"results":res},indent=2))
    print(f"[e110] FAST-CLICK SOLVED {len(solved)}/{len(games)}: {sorted(solved)}",flush=True)
if __name__=="__main__": main()
