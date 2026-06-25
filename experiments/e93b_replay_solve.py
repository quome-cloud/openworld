"""E93b -- replay a captured winning action sequence to VERIFY a reproducible level completion.
Deterministic env => replaying the full prefix from reset reaches the level. A concrete, reproducible
'solved a level' demonstration. python3 e93b_replay_solve.py --game sp80 --reward /tmp/e93_sp80.json"""
import argparse, json
import numpy as np, arc_agi
from arcengine import GameAction
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]
ap=argparse.ArgumentParser(); ap.add_argument("--game",default="sp80"); ap.add_argument("--reward",required=True); ap.add_argument("--out",default="")
a=ap.parse_args()
rew=json.load(open(a.reward))["rewards"][0]
seq=rew.get("full_prefix") or (rew.get("recent_actions",[])+[rew["action"]])
arc=arc_agi.Arcade(); env=arc.make(a.game); obs=env.reset(); best=obs.levels_completed; hit=None
for i,act in enumerate(seq):
    obs=env.step(ACTS[act-1])
    if obs is None or getattr(obs,"frame",None) is None: break
    if obs.levels_completed>best: hit=i; best=obs.levels_completed; break
    best=max(best,obs.levels_completed)
res={"game":a.game,"replay_len":len(seq),"reached_level_at_step":hit,"best_levels":int(best),
     "win_levels":int(obs.win_levels) if obs else None,"SOLVED":best>0}
print(f"[e93b/{a.game}] replay {len(seq)} actions -> level {best} at step {hit} | SOLVED={best>0}")
import pathlib; outp=a.out or f"experiments/results/e93b_replay_{a.game}.json"; pathlib.Path(outp).write_text(json.dumps(res,indent=2)); print("wrote",outp)
