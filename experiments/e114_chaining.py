"""E114 -- CHAINING to deeper levels on the games we solve. Sequential per-level solving: solve L1, then
explore FROM the L1-complete state for L2, ... (focuses world-time compute per level vs re-solving from
reset). Reports total LEVELS solved (paper #2 reports levels: 30/52). Reuses the E113 multi-perception
primitives + the discovered World map."""
import argparse, json
from pathlib import Path
import numpy as np
import arc_agi
import e113_multiperception as E

def chain(game, budget=70000, max_depth=250, seed=0):
    import random as _r
    arc=arc_agi.Arcade(); ENV=arc.make(game); o=ENV.reset(); avail=list(o.available_actions); win=o.win_levels; base=o.levels_completed
    mask,bg=E.detect_mask(ENV,avail); keep=(~mask).reshape(-1)
    def sig(g): return hash(g.reshape(-1)[keep].tobytes())
    def reach(path):
        ob=ENV.reset()
        for a in path:
            ob=E.step(ENV,a)
            if ob is None or getattr(ob,"frame",None) is None: return None
        return ob
    modes=(["click"] if avail==[6] else (["dir"] if 6 not in avail else ["dir","click"]))
    full=[]; level=base
    for L in range(win):
        ob=reach(full)
        if ob is None: break
        cur=ob.levels_completed; found=None
        for mode in modes:
            rng=_r.Random(seed); tested={}; steps=0; acache={}
            while steps<budget and found is None:
                if reach(full) is None: break
                g=E.g_of(ENV._last_response) if False else E.g_of(reach(full)); seq=[]; d=0; prev=None
                # re-reach already done above sets ENV at full's end; play forward
                while d<max_depth and steps<budget:
                    s=sig(g); acts=acache.get(s)
                    if acts is None: acts=E.action_model(g,bg,avail,mode,prev); acache[s]=acts
                    if not acts: break
                    ts=tested.setdefault(s,set()); unt=[x for x in acts if x not in ts]
                    a=rng.choice(unt) if unt else rng.choice(acts); ts.add(a)
                    no=E.step(ENV,a); steps+=1; d+=1
                    if no is None or getattr(no,"frame",None) is None: break
                    seq.append(a); ng=E.g_of(no)
                    if no.levels_completed>cur: found=seq; break
                    if str(no.state)!="GameState.NOT_FINISHED": break
                    prev=g; g=ng
                if found is None:                      # restart rollout from full
                    if reach(full) is None: break
            if found: break
        if found is None: break
        full+=found; level+=1
    total=E.verify(game,full)
    return {"game":game,"levels":int(total),"win":int(win),"reached":int(level-base),"sol_len":len(full),"solution":full}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--games",default="ar25,cd82,cn04,lf52,ls20,m0r0,s5i5,sk48,sp80,vc33")
    ap.add_argument("--budget",type=int,default=70000); ap.add_argument("--out",default="results/e114_chaining.json"); a=ap.parse_args()
    games=a.games.split(","); print(f"[e114] chaining on {len(games)} solved games",flush=True); res={}
    for g in games:
        try: r=chain(g,a.budget)
        except Exception as e:
            import traceback; traceback.print_exc(); r={"game":g,"levels":0,"error":str(e)[:80]}
        res[g]=r; print(f"  {g}: {r.get('levels',0)}/{r.get('win','?')} levels (len {r.get('sol_len','-')})",flush=True)
    tot=sum(r.get("levels",0) for r in res.values())
    Path(a.out).write_text(json.dumps({"total_levels":tot,"results":res},indent=2))
    print(f"[e114] CHAINED total levels: {tot} across {len(games)} games",flush=True)
if __name__=="__main__": main()
