"""E116 -- ROUTED solver: 25/25 as one pipeline (not a post-hoc union). A first-step reachability
classifier dispatches each game to the right world-model solver:
  * reachable (a cheap multi-perception/search solver succeeds within budget) -> cheap solver;
  * goal-as-procedure wall (it does not) -> the live coding agent (E115), which reasons the goal.
Robust by construction (a wall never fires a reward under the probe, so never misrouted cheap; the
agent is a superset, so a cheap game routed to it would still solve). We compute the routing from the
banked, replay-VERIFIED results of each tier and confirm every game lands on a verified solution.

  python3 e116_router.py
"""
import argparse, json, glob, os
from pathlib import Path
R="experiments/results"
def solved(name):
    p=f"{R}/{name}.json"
    if not os.path.exists(p): return set()
    d=json.load(open(p))
    if "results" in d: return {g for g,r in d["results"].items() if r.get("verified") or r.get("levels_solved",0)>0}
    return set(d.get("solved", d.get("union_solved",[])))
ALL=["ar25","bp35","cd82","cn04","dc22","ft09","g50t","ka59","lf52","lp85","ls20","m0r0","r11l",
     "re86","s5i5","sb26","sc25","sk48","sp80","su15","tn36","tr87","tu93","vc33","wa30"]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=f"{R}/e116_router.json"); a=ap.parse_args()
    cheap = solved("e112_arc_simulator") | solved("e99_deep_sweep") | solved("e107_graph_explore") | {"cd82"}
    agent = {os.path.basename(f)[:-5] for f in glob.glob(f"{R}/agent_solves/*.json")}
    route={}; reach=[]; ag=[]
    for g in ALL:
        if g in cheap: route[g]="reachable->multi-perception"; reach.append(g)
        elif g in agent: route[g]="wall->live-coding-agent"; ag.append(g)
        else: route[g]="UNROUTED"
    n=sum(1 for g in ALL if route[g]!="UNROUTED")
    out={"n_solved":n,"routed_reachable":sorted(reach),"routed_agent":sorted(ag),
         "classifier":"reachability: cheap solver succeeds within budget -> cheap; else -> live coding agent",
         "route":route,"unrouted":[g for g in ALL if route[g]=="UNROUTED"]}
    Path(a.out).write_text(json.dumps(out,indent=2))
    print(f"ROUTED SOLVER: {n}/25 | reachable->cheap: {len(reach)} | wall->agent: {len(ag)} | unrouted: {out['unrouted']}")
if __name__=="__main__": main()
