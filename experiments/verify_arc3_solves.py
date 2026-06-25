"""Reproduce + verify all saved ARC-AGI-3 solves: replay each captured winning sequence against the
live game and confirm it completes a level. Exit 0 iff all pass. Needs py>=3.12 + arc-agi==0.9.9."""
import json, sys
from pathlib import Path
import numpy as np
import arc_agi
from arcengine import GameAction
ACTS=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5,GameAction.ACTION6,GameAction.ACTION7]
d=json.load(open(Path(__file__).resolve().parent/"results"/"e99_deep_sweep.json"))
arc=arc_agi.Arcade(); ok=n=0
for g in sorted(d["solved"]):
    seq=d["results"].get(g,{}).get("sequence")
    if not seq: print(f"  {g}: NO SAVED SEQUENCE"); continue
    env=arc.make(g); o=env.reset(); base=o.levels_completed; hit=False
    for a in seq:
        o=env.step(ACTS[a-1])
        if o is None or getattr(o,"frame",None) is None: break
        if o.levels_completed>base: hit=True; break
    n+=1; ok+=int(hit)
    print(f"  {g}: replay {len(seq)} actions -> {'PASS (level completed)' if hit else 'FAIL'}")
print(f"\n{ok}/{n} saved ARC-AGI-3 solves verified")
sys.exit(0 if ok==n and n>0 else 1)
