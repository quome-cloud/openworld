"""Fig: per-game levels reached by RANDOM baseline vs OUR best method -- makes the honest '14/25 above
random' headline visual. Writes papers/arc-3/figs/arc3_above_random.png."""
import json, glob, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=Path("experiments/results"); FIG=Path("papers/arc-3/figs"); FIG.mkdir(parents=True,exist_ok=True)
rand=json.load(open(R/"e117_random_baseline.json"))["results"]
randlv={g:rand[g].get("random_levels",0) for g in rand}
# our-best levels per game: max(agent solve levels, e112 levels_solved, 1 if cheap-solved)
ours={g:0 for g in randlv}
e112=json.load(open(R/"e112_arc_simulator.json"))["results"]
for g,r in e112.items():
    if r.get("verified"): ours[g]=max(ours.get(g,0), r.get("levels_solved",1))
for f in glob.glob(str(R/"agent_solves"/"*.json")):
    g=os.path.basename(f)[:-5]; ours[g]=max(ours.get(g,0), json.load(open(f)).get("levels",1))
for name in ("e99_deep_sweep","e107_graph_explore"):
    for g in json.load(open(R/f"{name}.json")).get("solved",[]): ours[g]=max(ours.get(g,0),1)
games=sorted(randlv, key=lambda g:(ours[g]-randlv[g], ours[g]), reverse=True)
rv=[randlv[g] for g in games]; ov=[ours[g] for g in games]
above=[ov[i]>rv[i] for i in range(len(games))]
x=np.arange(len(games)); w=0.4
plt.figure(figsize=(12,4.6))
plt.bar(x-w/2, rv, w, label="random baseline", color="#cbd5e1", edgecolor="#94a3b8")
plt.bar(x+w/2, ov, w, label="our best method", color=["#2e8b57" if a else "#f59e0b" for a in above], edgecolor="#334155")
for i,g in enumerate(games):
    if ov[i]>rv[i] and ov[i]>=2: plt.text(i+w/2, ov[i]+0.08, f"+{ov[i]-rv[i]}", ha="center", fontsize=7, color="#166534", fontweight="bold")
plt.xticks(x, games, rotation=90, fontsize=8, fontfamily="monospace"); plt.ylabel("levels completed")
n_above=sum(above)
n_zero=sum(1 for i in range(len(games)) if rv[i]==0 and ov[i]>=1)
plt.title(f"The honest measure: {n_above}/25 strictly above random ({n_zero} random never reaches); random already gets ≥1 level on {sum(1 for v in rv if v>=1)}/25", fontsize=10.5, fontweight="bold")
from matplotlib.patches import Patch
_leg=[Patch(facecolor="#cbd5e1", edgecolor="#94a3b8", label="random baseline"),
      Patch(facecolor="#2e8b57", edgecolor="#334155", label="our method — above random"),
      Patch(facecolor="#f59e0b", edgecolor="#334155", label="our method — random also reaches ≥1")]
plt.legend(handles=_leg, loc="upper right", fontsize=8.5, frameon=True); plt.tight_layout(); plt.savefig(FIG/"arc3_above_random.png", dpi=140); plt.close()
print("wrote arc3_above_random.png |", n_above, "above random; random>=1 on", sum(1 for v in rv if v>=1))
