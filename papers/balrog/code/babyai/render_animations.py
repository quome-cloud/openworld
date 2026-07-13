"""Render clean-arm episode animations: the agent's served 7x7 egocentric view
('what the agent sees') side by side with the reconstructed allocentric belief
map ('what the agent has mapped'), plus mission text, step counter, last action.

Episodes are replayed exactly from their recorded seeds (env construction +
reset(seed) is deterministic, CleanAgent is deterministic in the obs stream);
the replayed action sequence is written to the metadata JSON alongside each GIF.

Output: results/animations/<task>_ep<NN>_seed<seed>.gif (+ animations_meta.json)
"""

from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

from balrog_env import make_task_env
from clean_agent import CleanAgent
from symbolic_model import DIR_TO_VEC, IDX_TO_OBJECT, IDX_TO_COLOR

ACTION_NAMES = ["turn left", "turn right", "go forward", "pick up", "drop", "toggle"]

TS = 30  # tile size px

RGB = {
    "red": (214, 66, 62), "green": (76, 175, 92), "blue": (66, 106, 222),
    "purple": (148, 76, 204), "yellow": (222, 196, 66), "grey": (146, 146, 152),
}
C_BG = (24, 24, 28)
C_PANEL = (34, 34, 40)
C_UNKNOWN = (48, 48, 56)
C_EMPTY = (232, 232, 228)
C_WALL = (98, 98, 106)
C_GRIDLINE = (208, 208, 204)
C_AGENT = (255, 96, 32)
C_TEXT = (235, 235, 235)
C_DIM = (160, 160, 170)
C_OK = (98, 212, 120)
C_TRAIL = (255, 190, 150)

FONT_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


def font(sz, bold=False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_PATH, sz)
    except OSError:
        return ImageFont.load_default()


def draw_cell(d, x0, y0, kind, color=None, state=None):
    """Draw one tile at pixel (x0, y0). kind in {unknown, empty, wall, door,
    key, ball, box}."""
    x1, y1 = x0 + TS, y0 + TS
    if kind == "unknown":
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=C_UNKNOWN)
        for k in range(0, TS, 6):  # subtle hatch
            d.line([x0 + k, y0, x0, y0 + k], fill=(58, 58, 66))
            d.line([x1 - 1, y1 - 1 - k, x1 - 1 - k, y1 - 1], fill=(58, 58, 66))
        return
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=C_EMPTY, outline=C_GRIDLINE)
    if kind == "empty":
        return
    if kind == "wall":
        d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=C_WALL, outline=(78, 78, 86))
        return
    c = RGB[color]
    pad = 4
    if kind == "door":
        if state == 0:      # open: frame only, doorway clear
            d.rectangle([x0 + 1, y0 + 1, x1 - 2, y1 - 2], outline=c, width=3)
            d.rectangle([x1 - 8, y0 + 3, x1 - 4, y1 - 4], fill=c)
        elif state == 2:    # locked: solid + keyhole
            d.rectangle([x0 + 1, y0 + 1, x1 - 2, y1 - 2], fill=c)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            d.ellipse([cx - 3, cy - 4, cx + 3, cy + 2], fill=(20, 20, 20))
            d.rectangle([cx - 1, cy, cx + 1, cy + 7], fill=(20, 20, 20))
        else:               # closed
            d.rectangle([x0 + 1, y0 + 1, x1 - 2, y1 - 2], fill=c)
            d.rectangle([x0 + 5, y0 + 5, x1 - 6, y1 - 6], outline=(20, 20, 20))
            d.ellipse([x1 - 11, (y0 + y1) // 2 - 2, x1 - 7, (y0 + y1) // 2 + 2],
                      fill=(20, 20, 20))
    elif kind == "ball":
        d.ellipse([x0 + pad, y0 + pad, x1 - pad - 1, y1 - pad - 1], fill=c)
    elif kind == "key":
        cx = x0 + TS // 2
        d.ellipse([cx - 6, y0 + 4, cx + 4, y0 + 14], outline=c, width=3)
        d.rectangle([cx - 2, y0 + 13, cx + 1, y1 - 5], fill=c)
        d.rectangle([cx + 1, y1 - 12, cx + 6, y1 - 9], fill=c)
        d.rectangle([cx + 1, y1 - 7, cx + 6, y1 - 4], fill=c)
    elif kind == "box":
        d.rectangle([x0 + pad, y0 + pad, x1 - pad - 1, y1 - pad - 1],
                    outline=c, width=3)
        d.line([x0 + pad, (y0 + y1) // 2, x1 - pad - 1, (y0 + y1) // 2],
               fill=c, width=2)


def draw_agent(d, x0, y0, dir_, small=False):
    """Triangle pointing along dir_ (0=E,1=S,2=W,3=N)."""
    m = 8 if small else 5
    x1, y1 = x0 + TS, y0 + TS
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    pts = {
        0: [(x1 - m, cy), (x0 + m, y0 + m), (x0 + m, y1 - m)],
        1: [(cx, y1 - m), (x0 + m, y0 + m), (x1 - m, y0 + m)],
        2: [(x0 + m, cy), (x1 - m, y0 + m), (x1 - m, y1 - m)],
        3: [(cx, y0 + m), (x0 + m, y1 - m), (x1 - m, y1 - m)],
    }[dir_]
    d.polygon(pts, fill=C_AGENT, outline=(120, 30, 0))


def cell_kind(v):
    """belief cell value -> (kind, color, state)."""
    if v is None:
        return ("empty", None, None)
    t, c, s = v
    return (t, c, s if t == "door" else None)


def render_frame(obs, belief_snap, trail, extent, mission, step, max_steps,
                 last_action, status):
    """Compose one frame."""
    (bx0, by0, bx1, by1) = extent
    mw, mh = (bx1 - bx0 + 1), (by1 - by0 + 1)

    left_w = 7 * TS
    right_w = mw * TS
    gap = 24
    margin = 16
    header_h = 64
    label_h = 26
    panel_h = max(7, mh) * TS
    W = margin * 2 + left_w + gap + right_w
    H = header_h + label_h + panel_h + margin + 22

    img = Image.new("RGB", (W, H), C_BG)
    d = ImageDraw.Draw(img)

    f_big = font(15, bold=True)
    f_med = font(13)
    f_lab = font(13, bold=True)

    # header
    d.text((margin, 10), f"mission: {mission}", font=f_big, fill=C_TEXT)
    step_txt = f"step {step}/{max_steps}"
    act_txt = f"last action: {last_action if last_action is not None else '—'}"
    d.text((margin, 34), step_txt, font=f_med, fill=C_DIM)
    d.text((margin + 130, 34), act_txt, font=f_med, fill=C_DIM)
    if status:
        d.text((margin + 360, 34), status, font=f_big, fill=C_OK)

    # labels
    ly = header_h
    d.text((margin, ly), "what the agent sees", font=f_lab, fill=C_TEXT)
    d.text((margin + left_w + gap, ly), "what the agent has mapped",
           font=f_lab, fill=C_TEXT)
    d.text((margin + left_w + gap + 200, ly), "(unknown = hatched)",
           font=f_med, fill=C_DIM)

    # ---- left panel: served egocentric view (forward = up) ----
    ox, oy = margin, header_h + label_h
    d.rectangle([ox - 3, oy - 3, ox + left_w + 2, oy + 7 * TS + 2],
                outline=(70, 70, 80), fill=C_PANEL)
    for vy in range(7):
        for vx in range(7):
            t, c, s = int(obs["image"][vx][vy][0]), int(obs["image"][vx][vy][1]), \
                int(obs["image"][vx][vy][2])
            px, py = ox + vx * TS, oy + vy * TS
            if t == 0:
                draw_cell(d, px, py, "unknown")
            elif t == 1:
                draw_cell(d, px, py, "empty")
            else:
                draw_cell(d, px, py, IDX_TO_OBJECT[t], IDX_TO_COLOR[c], s)
    # agent marker at (3,6) facing up; carried object already drawn at cell
    draw_agent(d, ox + 3 * TS, oy + 6 * TS, 3, small=True)

    # ---- right panel: belief map ----
    cells, pos, dir_, carrying = belief_snap
    ox2 = margin + left_w + gap
    d.rectangle([ox2 - 3, oy - 3, ox2 + right_w + 2, oy + mh * TS + 2],
                outline=(70, 70, 80), fill=C_PANEL)
    for wy in range(by0, by1 + 1):
        for wx in range(bx0, bx1 + 1):
            px, py = ox2 + (wx - bx0) * TS, oy + (wy - by0) * TS
            if (wx, wy) not in cells:
                draw_cell(d, px, py, "unknown")
            else:
                kind, col, st = cell_kind(cells[(wx, wy)])
                draw_cell(d, px, py, kind, col, st)
    # exploration trail
    for (tx, ty) in trail:
        px, py = ox2 + (tx - bx0) * TS + TS // 2, oy + (ty - by0) * TS + TS // 2
        d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=C_TRAIL)
    draw_agent(d, ox2 + (pos[0] - bx0) * TS, oy + (pos[1] - by0) * TS, dir_)

    # carried-object chip
    if carrying is not None:
        cy0 = oy + mh * TS + 6
        d.text((ox2, cy0), f"carrying: {carrying[1]} {carrying[0]}",
               font=f_med, fill=RGB[carrying[1]])

    return img


def replay_and_render(task, episode_idx, seed, out_dir):
    env = make_task_env(task)
    obs, info = env.reset(seed=seed)
    mission = obs["mission"]
    max_steps = env.unwrapped.max_steps
    agent = CleanAgent(mission)

    # pass 1: replay, capturing per-step obs + belief snapshots
    # (Belief.integrate is idempotent for a fixed obs/pose, so pre-integrating
    # for the snapshot does not perturb the agent.)
    snaps = []      # (obs, belief_snapshot, last_action_name)
    trail = []
    actions = []

    def snapshot():
        b = agent.belief
        return (dict(b.cells), b.pos, b.dir, b.carrying)

    agent.belief.integrate(obs)
    trail.append(agent.belief.pos)
    snaps.append((obs, snapshot(), None))

    solved = False
    steps = 0
    for _ in range(max_steps):
        a = agent.act(obs)
        actions.append(int(a))
        obs, reward, term, trunc, info = env.step(int(a))
        steps += 1
        agent.belief.integrate(obs)
        trail.append(agent.belief.pos)
        snaps.append((obs, snapshot(), ACTION_NAMES[a]))
        if reward > 0:
            solved = True
        if term or trunc:
            break
    env.close()

    # fixed map extent across all frames = final belief bbox + agent trail
    cells_final = snaps[-1][1][0]
    xs = [p[0] for p in cells_final] + [p[0] for p in trail]
    ys = [p[1] for p in cells_final] + [p[1] for p in trail]
    extent = (min(xs), min(ys), max(xs), max(ys))

    frames = []
    durations = []
    for i, (o, snap, act_name) in enumerate(snaps):
        status = ""
        if i == len(snaps) - 1:
            status = f"SOLVED in {steps} steps" if solved else "FAILED"
        frames.append(render_frame(o, snap, trail[:i + 1], extent, mission,
                                   i, max_steps, act_name, status))
        durations.append(700 if i == 0 else (2200 if i == len(snaps) - 1 else 340))

    name = f"{task}_ep{episode_idx:02d}_seed{seed}"
    gif_path = os.path.join(out_dir, name + ".gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    return {
        "task": task, "episode": episode_idx, "seed": seed, "mission": mission,
        "solved": solved, "steps": steps, "max_steps": max_steps,
        "actions": actions,
        "action_names": [ACTION_NAMES[a] for a in actions],
        "gif": os.path.basename(gif_path), "frames": len(frames),
    }


def main():
    out_dir = "results/animations"
    os.makedirs(out_dir, exist_ok=True)

    # pick, per task, the clean episode with the largest exploration overhead
    # (clean steps - privileged optimal steps)
    priv = {(r["task"], r["seed"]): r for r in
            json.load(open("results/privileged_results.json"))}
    clean = json.load(open("results/clean_results.json"))
    best = {}
    for r in clean:
        d = r["steps_used"] - priv[(r["task"], r["seed"])]["steps_used"]
        if r["task"] not in best or d > best[r["task"]][0]:
            best[r["task"]] = (d, r)

    meta = {"note": "clean-protocol replays (deterministic from seed); "
                    "episodes chosen per task = max exploration overhead vs "
                    "privileged-optimal plan", "episodes": []}
    for task, (d, r) in best.items():
        m = replay_and_render(task, r["episode"], r["seed"], out_dir)
        m["exploration_overhead_steps"] = d
        # consistency check vs the recorded suite run
        assert m["solved"] and m["steps"] == r["steps_used"], \
            f"replay mismatch on {task}: {m['steps']} vs {r['steps_used']}"
        meta["episodes"].append(m)
        print(f"{task}: {m['gif']} ({m['frames']} frames, +{d} exploration steps, "
              f"steps {m['steps']}/{m['max_steps']}) replay==recorded OK", flush=True)

    with open(os.path.join(out_dir, "animations_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("done ->", out_dir)


if __name__ == "__main__":
    main()
