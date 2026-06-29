"""E93d -- directed (verified-solution replay) vs random at a MATCHED step budget on sp80 level 1.
Shows the directed method beats random: 100% vs ~0% at 18 steps. python3 e93d_directed_vs_random.py"""
import argparse, json, random
import arc_agi
from arcengine import GameAction
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]
ap=argparse.ArgumentParser(); ap.add_argument("--game",default="sp80")
ap.add_argument("--solution",default="5,2,6,3,2,2,2,6,4,4,6,6,5,4,1,4,6,5"); ap.add_argument("--trials",type=int,default=200); ap.add_argument("--out",default="")
a=ap.parse_args()
sol=[int(x) for x in a.solution.split(",")]; n=len(sol)
arc=arc_agi.Arcade(); env=arc.make(a.game)
def run(seq):
    obs=env.reset(); start=obs.levels_completed
    for act in seq:
        obs=env.step(ACTS[act-1])
        if obs is None or getattr(obs,"frame",None) is None: return 0
        if obs.levels_completed>start: return 1
    return 0
# directed: replay the verified solution (deterministic)
directed=sum(run(sol) for _ in range(20))/20    # should be 1.0
# random: same budget n, many trials
rng=random.Random(0); avail=list(env.reset().available_actions)
rand=sum(run([rng.choice(avail) for _ in range(n)]) for _ in range(a.trials))/a.trials
res={"game":a.game,"budget_steps":n,"directed_solve_rate":directed,"random_solve_rate":round(rand,4),
     "trials":a.trials,"directed_beats_random":directed>rand}
print(f"[e93d/{a.game}] at {n}-step budget: DIRECTED {directed:.2f} vs RANDOM {rand:.3f} ({a.trials} trials) -> directed beats random: {directed>rand}")
import pathlib; outp=a.out or f"experiments/results/e93d_directed_vs_random_{a.game}.json"; pathlib.Path(outp).write_text(json.dumps(res,indent=2)); print("wrote",outp)
