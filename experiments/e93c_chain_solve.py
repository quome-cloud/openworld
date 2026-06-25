"""E93c -- greedy reward-seeking CHAIN solver (directed, beats random).

Random reaches only sp80 level 1. This locks each level's solution and searches forward for the
next, never losing progress (deterministic env: replay the locked prefix, then extend). Chains
level completions -> can fully solve a game. python3 e93c_chain_solve.py --game sp80
"""
import argparse, json, random
import arc_agi
from arcengine import GameAction
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]

def replay(env_make, prefix):
    env=env_make(); obs=env.reset()
    for a in prefix:
        obs=env.step(ACTS[a-1])
        if obs is None or getattr(obs,"frame",None) is None: return None,None
    return env,obs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--game",default="sp80")
    ap.add_argument("--prefix",default=""); ap.add_argument("--max_ext",type=int,default=40)
    ap.add_argument("--attempts",type=int,default=400); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out",default=""); a=ap.parse_args()
    arc=arc_agi.Arcade(); mk=lambda: arc.make(a.game)
    obs0=mk().reset(); win=obs0.win_levels; avail=list(obs0.available_actions)
    solution=[int(x) for x in a.prefix.split(",") if x.strip()] if a.prefix else []
    env,obs=replay(mk,solution) if solution else (mk(),mk().reset())
    best=obs.levels_completed if obs else 0
    rng=random.Random(a.seed)
    print(f"[e93c/{a.game}] start best={best}/{win}, solution_len={len(solution)}",flush=True)
    while best<win:
        found=False
        for at in range(a.attempts):
            env,obs=replay(mk,solution)
            if env is None: break
            ext=[]
            for _ in range(a.max_ext):
                act=rng.choice(avail); obs=env.step(ACTS[act-1]); ext.append(act)
                if obs is None or getattr(obs,"frame",None) is None: break
                if obs.levels_completed>best:
                    solution+=ext; best=obs.levels_completed; found=True
                    print(f"[e93c/{a.game}] LEVEL {best}/{win} reached (sol_len={len(solution)}, attempt {at})",flush=True)
                    break
                if str(obs.state)!="GameState.NOT_FINISHED": break
            if found: break
        if not found:
            print(f"[e93c/{a.game}] could not extend past level {best} in {a.attempts} attempts",flush=True); break
    res={"game":a.game,"best_levels":int(best),"win_levels":int(win),"solved_full":best>=win,
         "solution_len":len(solution),"solution":solution}
    print(f"[e93c/{a.game}] FINAL best {best}/{win} | full_solve={best>=win}",flush=True)
    import pathlib; outp=a.out or f"experiments/results/e93c_chain_{a.game}.json"; pathlib.Path(outp).write_text(json.dumps(res,indent=2)); print("wrote",outp)

if __name__=="__main__": main()
