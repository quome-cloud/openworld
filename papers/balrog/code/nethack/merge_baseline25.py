"""Merge the parallel chunk results into nethack_results_baseline25.json.
Dedupe rule (pre-declared): one entry per seed, first-completed wins
(completion order approximated by file write order captured in each
chunk/суite file; entries carry wallclock, we keep the first encountered
scanning main file then w0,w1,w2,w3,w4)."""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MAIN = os.path.join(RESULTS, "nethack_results_baseline25.json")

doc = json.load(open(MAIN))
seen = {e["seed"] for e in doc["episodes"]}
dupes = []
for fn in sorted(glob.glob(os.path.join(
        RESULTS, "nethack_results_baseline25_w*.json"))):
    sub = json.load(open(fn))
    for e in sub["episodes"]:
        if e["seed"] in seen:
            dupes.append((e["seed"], fn))
            continue
        seen.add(e["seed"])
        doc["episodes"].append(e)
doc["episodes"].sort(key=lambda e: e["seed"])
prog = [e["progression"] for e in doc["episodes"]]
doc["score"] = 100.0 * sum(prog) / len(prog)
doc["n"] = len(prog)
if dupes:
    doc["dedupe_dropped"] = [f"{s} from {os.path.basename(f)}" for s, f in dupes]
json.dump(doc, open(MAIN, "w"), indent=1)
print(f"merged: n={doc['n']} score={doc['score']:.2f} dupes_dropped={len(dupes)}")
for e in doc["episodes"]:
    print(f"  seed {e['seed']}: {e['progression']*100:5.2f} depth {e['depth_max']:2d} "
          f"{(e.get('role') or '?'):12s} {e['end_reason'][:60]}")
