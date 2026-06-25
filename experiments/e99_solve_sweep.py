"""E99 -- apply the solving toolkit to ALL ARC-AGI-3 games to find more solves.

Key insight from sp80: the win fires on an INTERACT action (5/6/7), which the uniform E93 sweep
under-weighted. So we run an interact-biased, longer reward search per game; on the first reward we
capture the full action prefix since the last reset and VERIFY it by deterministic replay (a solve
counts only if it reproduces). Then a greedy chain extends to further levels. CPU-only, no LLM/GPU.

  python3 e99_solve_sweep.py --budget 12000 --bias 0.45
"""
import argparse, json, logging, contextlib, io, random
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]

def grid(o):
    a=np.asarray(o.frame); return (a[-1] if a.ndim==3 else a).reshape(64,64)

def all_games():
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        envs=arc_agi.Arcade().available_environments
    return sorted({(e if isinstance(e,str) else getattr(e,"game_id",str(e))).split("-")[0] for e in envs})

def replay_verify(game, seq, want_level):
    arc=arc_agi.Arcade(); env=arc.make(game); o=env.reset(); base=o.levels_completed
    for a in seq:
        o=env.step(ACTS[a-1])
        if o is None or getattr(o,"frame",None) is None: return False
        if o.levels_completed>base: return True
    return False

def search(game, budget, bias, seed):
    arc=arc_agi.Arcade(); env=arc.make(game); obs=env.reset()
    avail=list(obs.available_actions); inter=[a for a in avail if a>=5]; move=[a for a in avail if a<5] or avail
    lvl=obs.levels_completed; rng=random.Random(seed); recent=[]
    for step in range(budget):
        a=rng.choice(inter) if (inter and rng.random()<bias) else rng.choice(move)
        no=env.step(ACTS[a-1])
        if no is None or getattr(no,"frame",None) is None:
            obs=env.reset(); avail=list(obs.available_actions); lvl=obs.levels_completed; recent=[]; continue
        recent.append(a)
        if no.levels_completed>lvl:                       # first reward
            seq=list(recent); ok=replay_verify(game, seq, no.levels_completed)
            return {"reward_step":step,"level":int(no.levels_completed),"seq_len":len(seq),
                    "verified":ok,"sequence":seq if ok else None}
        lvl=no.levels_completed
        if str(no.state)!="GameState.NOT_FINISHED":
            obs=env.reset(); avail=list(obs.available_actions); lvl=obs.levels_completed; recent=[]
    return {"level":0,"verified":False}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--budget",type=int,default=12000); ap.add_argument("--bias",type=float,default=0.45)
    ap.add_argument("--seed",type=int,default=0); ap.add_argument("--seeds",type=int,default=1); ap.add_argument("--out",default="results/e99_solve_sweep.json"); a=ap.parse_args()
    games=all_games(); print(f"[e99] sweeping {len(games)} games (budget {a.budget}, interact-bias {a.bias})",flush=True)
    res={}
    for g in games:
        r={"level":0,"verified":False}
        for sd in range(a.seeds):
            try: rr=search(g, a.budget, a.bias, a.seed+sd)
            except Exception as e: rr={"error":str(e)[:80],"verified":False}
            if rr.get("verified") or rr.get("level",0)>r.get("level",0): r=rr
            if r.get("verified"): break
        res[g]=r
        tag="SOLVED" if r.get("verified") else ("reward(unverified)" if r.get("level",0)>0 else "no reward")
        print(f"  {g}: {tag} level={r.get('level',0)} seq_len={r.get('seq_len','-')}",flush=True)
    solved=[g for g,r in res.items() if r.get("verified")]
    out={"games":len(games),"solved":solved,"n_solved":len(solved),"results":res}
    Path(a.out).write_text(json.dumps(out,indent=2))
    print(f"[e99] SOLVED {len(solved)}/{len(games)}: {solved}",flush=True); print("wrote",a.out)

if __name__=="__main__": main()
