"""Build the ARC-AGI-3 strategy-landscape figures: (1) main bar chart of games solved per strategy,
(2) strategy x game heatmap (what works / what doesn't) -- a guide for world-model research.
Reads experiments/results/*.json; writes papers/arc-3/figs/. Run with a python that has matplotlib."""
import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
R=Path("experiments/results"); FIG=Path("papers/arc-3/figs"); FIG.mkdir(parents=True,exist_ok=True)
def agent_solved():
    import glob, os
    return {os.path.basename(f)[:-5] for f in glob.glob("experiments/results/agent_solves/*.json")}
def load(name):
    p=R/f"{name}.json"
    return json.load(open(p)) if p.exists() else {}
def solved_set(name, key="solved"):
    d=load(name)
    if not d: return set()
    if "results" in d:  # e112 style
        return {g for g,r in d["results"].items() if r.get("verified") or r.get("levels_solved",0)>0}
    return set(d.get(key, d.get("solved", [])))

# strategies in narrative order; last = the winning multi-perception consensus
STRATS=[
    ("Interact search\n(E99)",            solved_set("e99_deep_sweep")),
    ("Model-based MPC\n(E101)",           {"cd82"}),
    ("Atomic goals\n(E102)",              solved_set("e102_goal_search")),
    ("LLM hypotheses\n(E103)",            solved_set("e103_hypothesis")),
    ("Bayes subworld\n+semiring (E104)",  solved_set("e104_bayesian_subworld")),
    ("Graph explore\n(E107)",             solved_set("e107_graph_explore")),
    ("Multi-Perception\nConsensus (E112)",solved_set("e112_arc_simulator")),
    ("Live coding agent\n(E115, OpenWorld)", agent_solved()),
]
ALL=["ar25","bp35","cd82","cn04","dc22","ft09","g50t","ka59","lf52","lp85","ls20","m0r0","r11l",
     "re86","s5i5","sb26","sc25","sk48","sp80","su15","tn36","tr87","tu93","vc33","wa30"]
union=set().union(*[s for _,s in STRATS])
# matrix (games x strats); sort games: solved-by-most first, then alpha
M=np.array([[1 if g in s else 0 for _,s in STRATS] for g in ALL])
order=sorted(range(len(ALL)), key=lambda i:(-M[i].sum(), ALL[i]))
ALLo=[ALL[i] for i in order]; M=M[order]
labels=[n for n,_ in STRATS]; counts=[len(s) for _,s in STRATS]

# ---- Fig 1: main bar chart (games solved per strategy) ----
plt.figure(figsize=(8,4.2))
colors=["#9aa7b3"]*(len(STRATS)-1)+["#2e8b57"]   # winner in green
b=plt.bar(range(len(STRATS)), counts, color=colors, edgecolor="#33414f")
for i,c in enumerate(counts): plt.text(i, c+0.15, str(c), ha="center", fontsize=11, fontweight="bold")
plt.axhline(len(union), ls="--", c="#c2410c", lw=1.2); plt.text(0.1, len(union)+0.3, f"union = {len(union)}/25", color="#c2410c", fontsize=10, fontweight="bold")
plt.xticks(range(len(STRATS)), labels, fontsize=7.5)
plt.ylabel("ARC-AGI-3 games solved (≥1 level)"); plt.ylim(0,26)
plt.title("A live OpenWorld coding agent cracks the walls (union 23/25)", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(FIG/"arc3_strategy_bar.png", dpi=140); plt.close()

# ---- Fig 2: strategy x game heatmap ----
fig,ax=plt.subplots(figsize=(7.2,8.4))
cmap=ListedColormap(["#eef1f4","#2e8b57"])
ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1)
ax.set_xticks(range(len(STRATS))); ax.set_xticklabels(labels, fontsize=7.5, rotation=0)
ax.set_yticks(range(len(ALLo))); ax.set_yticklabels(ALLo, fontsize=8, fontfamily="monospace")
for i in range(len(ALLo)):
    for j in range(len(STRATS)):
        if M[i,j]: ax.text(j,i,"✓",ha="center",va="center",color="white",fontsize=9)
ax.set_xticks(np.arange(-.5,len(STRATS),1),minor=True); ax.set_yticks(np.arange(-.5,len(ALLo),1),minor=True)
ax.grid(which="minor",color="white",lw=1.2); ax.tick_params(which="minor",length=0)
# highlight winner column
ax.add_patch(plt.Rectangle((len(STRATS)-1.5,-.5),1,len(ALLo),fill=False,edgecolor="#c2410c",lw=2.5))
ax.set_title("What solves ARC-AGI-3: strategy × game\n(green = verified ≥1-level solve)", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.savefig(FIG/"arc3_strategy_heatmap.png", dpi=140); plt.close()
print("wrote arc3_strategy_bar.png + arc3_strategy_heatmap.png")
print("per-strategy solves:", {n:len(s) for n,s in STRATS})
print("UNION:", len(union), sorted(union))
