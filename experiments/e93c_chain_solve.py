"""E93c -- greedy reward-seeking CHAIN solver (directed, beats random).

Locks each level's solution, searches forward for the next (deterministic env: reset+replay the
locked prefix, then extend), never losing progress. Fast (one env, reset-based) and the extension
policy is BIASED toward interact actions (5,6,7) -- the level-1 win was ~44% interact vs 29% uniform,
so that structure is the signal. Chains level completions -> can fully solve a game and beat random
(which caps at sp80 level 1). python3 e93c_chain_solve.py --game sp80 --prefix 5,2,...
"""
import argparse, json, random
import arc_agi
from arcengine import GameAction
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--game",default="sp80")
    ap.add_argument("--prefix",default=""); ap.add_argument("--max_ext",type=int,default=120)
    ap.add_argument("--attempts",type=int,default=4000); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--interact_bias",type=float,default=0.45); ap.add_argument("--out",default=""); a=ap.parse_args()
    arc=arc_agi.Arcade(); env=arc.make(a.game)              # make ONCE
    obs0=env.reset(); win=obs0.win_levels; avail=list(obs0.available_actions)
    interact=[x for x in avail if x>=5]; move=[x for x in avail if x<5] or avail
    rng=random.Random(a.seed)
    def pick():
        if interact and rng.random()<a.interact_bias: return rng.choice(interact)
        return rng.choice(move)
    def replay(prefix):
        obs=env.reset()
        for act in prefix:
            obs=env.step(ACTS[act-1])
            if obs is None or getattr(obs,"frame",None) is None: return None
        return obs
    solution=[int(x) for x in a.prefix.split(",") if x.strip()]
    obs=replay(solution) if solution else env.reset()
    best=obs.levels_completed if obs else 0
    print(f"[e93c/{a.game}] start best={best}/{win} sol_len={len(solution)} interact={interact} bias={a.interact_bias}",flush=True)
    total=0
    while best<win:
        found=False
        for at in range(a.attempts):
            if replay(solution) is None: continue
            ext=[]
            for _ in range(a.max_ext):
                act=pick(); obs=env.step(ACTS[act-1]); ext.append(act); total+=1
                if obs is None or getattr(obs,"frame",None) is None: break
                if obs.levels_completed>best:
                    solution+=ext; best=obs.levels_completed; found=True
                    print(f"[e93c/{a.game}] LEVEL {best}/{win}! sol_len={len(solution)} (attempt {at}, total_steps {total})",flush=True)
                    break
                if str(obs.state)!="GameState.NOT_FINISHED": break
            if found: break
        if not found:
            print(f"[e93c/{a.game}] stuck at level {best} after {a.attempts} attempts",flush=True); break
    res={"game":a.game,"best_levels":int(best),"win_levels":int(win),"solved_full":best>=win,
         "solution_len":len(solution),"solution":solution,"beats_random":best>1}
    print(f"[e93c/{a.game}] FINAL best {best}/{win} full_solve={best>=win} beats_random={best>1}",flush=True)
    import pathlib; outp=a.out or f"experiments/results/e93c_chain_{a.game}.json"; pathlib.Path(outp).write_text(json.dumps(res,indent=2)); print("wrote",outp)

if __name__=="__main__": main()
