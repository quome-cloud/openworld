"""Visual playbacks: animated GIFs per showcase episode, rendered ONLY from the logged
clean-protocol transitions (results/transitions/): left pane = evolving room-graph map
(rooms visited, doors colored by state, objects seen, current room highlighted),
right pane = scrolling command/response transcript + step counter + score.

Money shot: the same treasure game twice — pass-1 death (trap taken) vs pass-2
memory-informed success (banner when the remembered-fatal object is in sight).
"""
import json, os, sys, io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fable_tw.cleanagents import Belief, OPP

DIR_OFF = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
DOOR_COLOR = {"open": "#2a9d3a", "closed": "#e69f00", "locked": "#d0342c", None: "#888888"}
OUT = os.path.join("results", "animations")
os.makedirs(OUT, exist_ok=True)


def room_positions(belief):
    """Grid layout by compass moves from the start room."""
    pos = {}
    if not belief.start and belief.rooms:
        belief.start = next(iter(belief.rooms))
    if not belief.start:
        return pos
    pos[belief.start] = (0, 0)
    frontier = [belief.start]
    while frontier:
        r = frontier.pop(0)
        R = belief.rooms.get(r, {"exits": {}})
        for d, ex in R["exits"].items():
            dest = ex.get("dest")
            if dest and dest not in pos and d in DIR_OFF:
                dx, dy = DIR_OFF[d]
                cand = (pos[r][0] + dx, pos[r][1] + dy)
                while cand in pos.values():
                    cand = (cand[0] + dx, cand[1] + dy)
                pos[dest] = cand
                frontier.append(dest)
    return pos


def snip(text, n=64):
    t = " ".join((text or "").split())
    return t[: n - 1] + "…" if len(t) > n else t


def render_frame(belief, transcript, step, score, title, banner=None, banner_color="#d0342c"):
    fig = plt.figure(figsize=(14, 7.2), dpi=80)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.04,
                          left=0.02, right=0.99, top=0.90, bottom=0.03)
    axm = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1])
    for ax in (axm, axt):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.965)

    pos = room_positions(belief)
    # edges
    for r, R in belief.rooms.items():
        if r not in pos:
            continue
        for d, ex in R["exits"].items():
            dest = ex.get("dest")
            if dest in pos:
                x0, y0 = pos[r]; x1, y1 = pos[dest]
                col = DOOR_COLOR[ex.get("door_state")] if ex.get("door") else "#666666"
                lw = 2.6 if ex.get("door") else 1.4
                axm.plot([x0, x1], [y0, y1], color=col, lw=lw, zorder=1)
    # nodes
    for r, (x, y) in pos.items():
        cur = (r == belief.cur)
        fc = "#ffd54d" if cur else "#eef2f7"
        ec = "#b8860b" if cur else "#7a8aa0"
        axm.add_patch(FancyBboxPatch((x - 0.38, y - 0.16), 0.76, 0.32,
                                     boxstyle="round,pad=0.02", fc=fc, ec=ec,
                                     lw=2.2 if cur else 1.0, zorder=2))
        axm.text(x, y + 0.045, r, ha="center", va="center", fontsize=7.4,
                 fontweight="bold" if cur else "normal", zorder=3)
        objs = [it for it, info in belief.items.items() if info["room"] == r][:3]
        conts = [c for c, i in belief.containers.items() if i["room"] == r][:2]
        lab = " ".join([f"[{c}]" for c in conts] + objs)
        if lab:
            axm.text(x, y - 0.085, snip(lab, 30), ha="center", va="center",
                     fontsize=5.8, color="#444444", zorder=3)
    if pos:
        xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
        axm.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
        axm.set_ylim(min(ys) - 0.55, max(ys) + 0.55)
    inv = ", ".join(sorted(belief.inventory)) or "(empty)"
    axm.text(0.01, 0.012, f"inventory: {snip(inv, 90)}", transform=axm.transAxes,
             fontsize=7.5, color="#1f3d7a", va="bottom")

    # transcript pane
    axt.text(0.03, 0.975, f"step {step}   score {score}", transform=axt.transAxes,
             fontsize=10, fontweight="bold", va="top", color="#333333")
    y = 0.925
    for cmd, resp in transcript[-12:]:
        axt.text(0.03, y, f"> {cmd}", transform=axt.transAxes, fontsize=8.4,
                 va="top", color="#0b5394", family="monospace", fontweight="bold")
        y -= 0.034
        axt.text(0.06, y, snip(resp, 70), transform=axt.transAxes, fontsize=7.4,
                 va="top", color="#555555", family="monospace")
        y -= 0.040
    if banner:
        axt.add_patch(FancyBboxPatch((0.02, 0.015), 0.96, 0.085, boxstyle="round,pad=0.01",
                                     transform=axt.transAxes, fc=banner_color, ec="none", alpha=0.92))
        axt.text(0.5, 0.057, banner, transform=axt.transAxes, fontsize=9.6,
                 ha="center", va="center", color="white", fontweight="bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def animate(trans_file, out_name, title, memory_avoid=None, ep_rec=None):
    data = json.load(open(trans_file))
    trans = data["transitions"]
    b = Belief()
    frames, transcript = [], []
    score = 0
    b.update(trans[0]["obs"], None)
    frames.append(render_frame(b, transcript, 0, 0, title))
    remembered_seen = False
    for i, t in enumerate(trans):
        transcript.append((t["cmd"], t["obs_next"]))
        b.update(t["obs_next"], t["cmd"])
        score += t["reward"] if isinstance(t["reward"], (int, float)) else 0
        banner, bcol = None, "#d0342c"
        if memory_avoid:
            vis = [o for o in memory_avoid if o in b.items and b.items[o]["room"] == b.cur]
            if vis:
                remembered_seen = True
            if remembered_seen:
                banner = f"REMEMBERED (pass 1): taking the {sorted(memory_avoid)[0]} is FATAL — avoiding"
                bcol = "#b3261e"
        if t["done"]:
            if data.get("won"):
                banner, bcol = f"EPISODE WON in {i+1} steps", "#2a7a2a"
            elif t["cmd"].startswith("take "):
                banner = f"TRAP SPRUNG: took the {t['cmd'][5:].split(' from ')[0]} → episode LOST"
                bcol = "#b3261e"
        frames.append(render_frame(b, transcript, i + 1, score, title, banner, bcol))
    # hold the final frame
    frames += [frames[-1]] * 4
    path = os.path.join(OUT, out_name)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=550, loop=0)
    print(f"{path}: {len(frames)} frames, {os.path.getsize(path)//1024} KB")


if __name__ == "__main__":
    T = "results/transitions"
    # showcase: first game of each task from pass 1
    coin = sorted(os.listdir(f"{T}/pass1/coin_collector"))[0]
    cook = sorted(os.listdir(f"{T}/pass1/the_cooking_game"))[0]
    animate(f"{T}/pass1/coin_collector/{coin}", "coin_collector.gif",
            f"coin_collector — {coin.rsplit('.',1)[0]} — CLEAN protocol (frontier exploration)")
    animate(f"{T}/pass1/the_cooking_game/{cook}", "cooking_game.gif",
            f"the_cooking_game — {cook.rsplit('.',1)[0]} — CLEAN protocol (recipe → collect → cook → cut → eat)")
    # money shot: same treasure game, pass-1 death vs pass-2 memory-informed win
    g = "seed_54472"
    animate(f"{T}/pass1/treasure_hunter/{g}.json", "treasure_pass1_death.gif",
            f"treasure_hunter — {g} — pass 1 (memoryless): the trap")
    ep = None
    for fn in sorted(os.listdir("results/memory/pass2/treasure_hunter")):
        r = json.load(open(f"results/memory/pass2/treasure_hunter/{fn}"))
        if r["game"].startswith(g):
            ep = r
            break
    avoid = set(ep["memory_in"]["avoid"]) if ep else set()
    animate(f"{T}/pass2/treasure_hunter/{g}.json", "treasure_pass2_memory.gif",
            f"treasure_hunter — {g} — pass 2 (ledger-informed): the payoff", memory_avoid=avoid, ep_rec=ep)
