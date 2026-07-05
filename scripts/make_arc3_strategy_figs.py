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
def random_solved():
    import json,os
    p="experiments/results/e117_random_baseline.json"
    if not os.path.exists(p): return set()
    d=json.load(open(p)); return set(d.get("random_geq1_level",[]))
def agent_solved():
    import glob, os
    return {os.path.basename(f)[:-5] for f in glob.glob("experiments/results/agent_solves/*.json")}
def router_solved():
    # E116 reachability router: route each game to the cheap/perceptual solver if it succeeds within
    # budget, else to the live coding agent. The union of both routed buckets is the pipeline's solve set.
    d=load("e116_router")
    return set(d.get("routed_reachable",[])) | set(d.get("routed_agent",[])) if d else set()
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
    ("Random play\n(baseline)",            random_solved()),
    ("Interact search\n(E99)",            solved_set("e99_deep_sweep")),
    ("Model-based MPC\n(E101)",           {"cd82"}),
    ("Atomic goals\n(E102)",              solved_set("e102_goal_search")),
    ("LLM hypotheses\n(E103)",            solved_set("e103_hypothesis")),
    ("Bayes subworld\n+semiring (E104)",  solved_set("e104_bayesian_subworld")),
    ("Graph explore\n(E107)",             solved_set("e107_graph_explore")),
    ("Multi-perception\n(E112)",solved_set("e112_arc_simulator")),
    ("Expert consensus\nvote (E120)",     solved_set("e120_expert_consensus")),
    ("Qwen-30B agent\n(E118)",            solved_set("e118_qwen_agent")),
    # E127 source-SIMULATED: reconstruct the engine source-free and certify vs the real env. It produces
    # CERTIFIED engines, not verified solves (the solve step was never run), and its reconstruction
    # fidelity collapses on the gap games -> 0 verified solves (empty column), a stated negative result.
    ("Source-sim\nreconstruct (E127)",    set()),
    ("Live coding agent\n(E115)", agent_solved()),
    # E116 ROUTER: route each game to the cheapest solver that clears it (cheap/perceptual within budget,
    # else the live coding agent) -> the union as ONE pipeline. This is the headline: 25/25.
    ("ROUTER pipeline\n(E116)", router_solved()),
    # additional goal-discovery negatives -- each scores a static state / hypothesis and banks 0 verified
    # source-free solves (empty columns in the GD band), reinforcing the goal-as-procedure diagnosis.
    ("Bayes goal-synth\n(E119)",          set()),
    ("Codex deep search\n(E124)",         set()),
    ("Lookahead search\n(E131/E132)",     set()),
]
ALL=["ar25","bp35","cd82","cn04","dc22","ft09","g50t","ka59","lf52","lp85","ls20","m0r0","r11l",
     "re86","s5i5","sb26","sc25","sk48","sp80","su15","tn36","tr87","tu93","vc33","wa30"]
union=set().union(*[s for _,s in STRATS])
# Category per strategy (same index order as STRATS) -> groups the columns so the figure shows HOW the
# goal-directed methods fail: the "Goal-discovery" block is a contiguous, almost-empty band.
SP="Search & perception"; GD="Goal-discovery (fails)"; RA="Reasoning agent"
CATS=[SP,SP,SP, GD,GD,GD, SP,SP, GD, RA, GD, RA, RA, GD,GD,GD]   # one per STRATS entry
CAT_ORDER=[SP,GD,RA]
CAT_COLOR={SP:"#2563eb", GD:"#c2410c", RA:"#15803d"}
# column order = group by category (CAT_ORDER), preserving narrative order within each group
corder=[j for cat in CAT_ORDER for j in range(len(STRATS)) if CATS[j]==cat]
STRATS=[STRATS[j] for j in corder]; CATS=[CATS[j] for j in corder]
# matrix (games x strats, reordered columns); sort games: solved-by-most first, then alpha
M=np.array([[1 if g in s else 0 for _,s in STRATS] for g in ALL])
order=sorted(range(len(ALL)), key=lambda i:(-M[i].sum(), ALL[i]))
ALLo=[ALL[i] for i in order]; M=M[order]
labels=[n for n,_ in STRATS]; counts=[len(s) for _,s in STRATS]
# contiguous [start,end] column span of each category, for brackets/separators
spans={c:(min(j for j in range(len(CATS)) if CATS[j]==c), max(j for j in range(len(CATS)) if CATS[j]==c)) for c in CAT_ORDER}
winner_col=labels.index("ROUTER pipeline\n(E116)") if any("ROUTER" in l for l in labels) else len(labels)-1

# ---- Fig 1: main bar chart (games solved per strategy) ----
plt.figure(figsize=(8,4.2))
colors=[CAT_COLOR[c] for c in CATS]              # bars tinted by method class
b=plt.bar(range(len(STRATS)), counts, color=colors, edgecolor="#33414f")
for i,c in enumerate(counts): plt.text(i, c+0.15, str(c), ha="center", fontsize=11, fontweight="bold")
plt.axhline(len(union), ls="--", c="#c2410c", lw=1.2); plt.text(0.1, len(union)+0.3, f"union = {len(union)}/25", color="#c2410c", fontsize=10, fontweight="bold")
plt.xticks(range(len(STRATS)), [l.replace("\n"," ") for l in labels], fontsize=7, rotation=40, ha="right")
plt.ylabel("ARC-AGI-3 games solved (≥1 level)"); plt.ylim(0,26)
plt.title(f"A live coding agent cracks the walls (union {len(union)}/25)", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.savefig(FIG/"arc3_strategy_bar.png", dpi=140); plt.close()

# ---- Fig 2: strategy x game heatmap (columns grouped by category) ----
nS=len(STRATS); nG=len(ALLo)
fig,ax=plt.subplots(figsize=(8.4,9.6))
cmap=ListedColormap(["#eef1f4","#2e8b57"])
ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1)
# vertical strategy labels (single line) -> no horizontal overlap; tinted by category
ax.set_xticks(range(nS))
xt=ax.set_xticklabels([l.replace("\n"," ") for l in labels], fontsize=7.2, rotation=90, ha="center")
for t,c in zip(xt, CATS): t.set_color(CAT_COLOR[c])
ax.set_yticks(range(nG)); ax.set_yticklabels(ALLo, fontsize=8, fontfamily="monospace")
for i in range(nG):
    for j in range(nS):
        if M[i,j]: ax.text(j,i,"✓",ha="center",va="center",color="white",fontsize=9)
ax.set_xticks(np.arange(-.5,nS,1),minor=True); ax.set_yticks(np.arange(-.5,nG,1),minor=True)
ax.grid(which="minor",color="white",lw=1.2); ax.tick_params(which="minor",length=0)
# separators between category groups + winner-column highlight
for c in CAT_ORDER[1:]:
    ax.axvline(spans[c][0]-0.5, color="#334155", lw=1.6)
ax.add_patch(plt.Rectangle((winner_col-.5,-.5),1,nG,fill=False,edgecolor="#c2410c",lw=2.5))
# category brackets + labels BELOW the strategy labels (x in data coords, y in axes fraction)
tr=ax.get_xaxis_transform()
for c in CAT_ORDER:
    s,e=spans[c]; mid=(s+e)/2
    ax.plot([s-0.4,e+0.4],[-0.52,-0.52], transform=tr, color=CAT_COLOR[c], lw=2.6, clip_on=False)
    ax.text(mid, -0.55, c, transform=tr, ha="center", va="top", fontsize=9.5, fontweight="bold",
            color=CAT_COLOR[c], clip_on=False)
ax.text((spans[GD][0]+spans[GD][1])/2, -0.60,
        "score a static state \N{RIGHTWARDS ARROW} cannot rank an ordered procedure",
        transform=tr, ha="center", va="top", fontsize=7.5, style="italic", color=CAT_COLOR[GD], clip_on=False)
ax.set_title("What solves ARC-AGI-3: strategy \N{MULTIPLICATION SIGN} game\n"
             "(green = verified \N{GREATER-THAN OR EQUAL TO}1-level solve; columns grouped by method class)",
             fontsize=11, fontweight="bold")
plt.tight_layout(); plt.savefig(FIG/"arc3_strategy_heatmap.png", dpi=140, bbox_inches="tight"); plt.close()
print("wrote arc3_strategy_bar.png + arc3_strategy_heatmap.png")
print("per-strategy solves:", {n:len(s) for n,s in STRATS})
print("UNION:", len(union), sorted(union))
