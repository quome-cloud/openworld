"""Render OpenWorld atlas cards for the OPUS (primary, 16/25) and CODEX/GPT-5.5 (12/25) source-free
arms, into papers/arc-3/maps/<arm>/<game>.svg -- the same recipe as build_arc3_fable_maps.py (replay
the banked solution, treat the discovered masked-frame state-transition graph as a World, render_card),
just pointed at a different arm's archive and only for the games that arm fully solved.

    /Users/jim/.arcv/bin/python scripts/build_arc3_arm_maps.py opus
    /Users/jim/.arcv/bin/python scripts/build_arc3_arm_maps.py codex
    /Users/jim/.arcv/bin/python scripts/build_arc3_arm_maps.py opus codex     # both
"""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCR = ROOT / "scratch_arc"
MAPS = ROOT / "papers" / "arc-3" / "maps"
sys.path.insert(0, str(SCR / "full_lf52"))          # arc3_harness
sys.path.insert(0, str(ROOT))                        # openworld core
os.chdir(ROOT)                                       # environment_files/ resolves here

from openworld import World, FunctionTransition, CodeObjective, render_card, to_spec  # noqa: E402

# arm -> (archive filename, card description label)
ARMS = {
    "opus":  ("arc3_fullgame_sourcefree.json",       "Claude Opus 4.8 source-free solve"),
    "codex": ("arc3_fullgame_sourcefree_codex.json", "GPT-5.5 (Codex) source-free solve"),
    "fable": ("arc3_fullgame_sourcefree_fable.json", "Claude Fable source-free solve"),
}


def act_key(a):
    if isinstance(a, int):
        return str(a), {}, str(a)
    a = list(a)
    if a[0] == 6:
        return "6", {"x": int(a[1]), "y": int(a[2])}, f"6:{int(a[1])},{int(a[2])}"
    return str(a[0]), {}, str(a[0])


def trace(game, actions):
    from arc3_harness import Game
    g = Game(game); g.reset()
    frames = [g.frame.copy()]; deltas = []; keys = []; base = g.levels; last = g.levels
    for a in actions:
        name, params, key = act_key(a)
        g.step(*([int(name)] if not params else [6, params["x"], params["y"]]))
        frames.append(g.frame.copy()); deltas.append(g.levels - last); keys.append((name, params, key))
        last = g.levels
        if g.done:
            break
    return frames, deltas, keys, g.levels - base


def state_ids(frames, indexed=False):
    F = np.stack([f.reshape(64, 64) for f in frames])
    if not indexed:
        mask = (F[1:] != F[:-1]).mean(axis=0) <= 0.95
        m = (F * mask).astype(np.int16)
        return [hashlib.blake2b(m[i].tobytes(), digest_size=8).hexdigest() for i in range(len(F))]
    return [f"{hashlib.blake2b(F[i].tobytes(), digest_size=8).hexdigest()}#{i}" for i in range(len(F))]


def build_table(sids, deltas, keys):
    table, ok = {}, True
    for i, (_, _, k) in enumerate(keys):
        key = (sids[i], k); val = (sids[i + 1], int(deltas[i]))
        if key in table and table[key] != val:
            ok = False
        table[key] = val
    return table, ok


def make_world(game, init_sid, table, label):
    def fn(state, action):
        name, params = action["name"], action.get("params") or {}
        k = name if not params else f"{name}:{params['x']},{params['y']}"
        nxt = table.get((state["sid"], k))
        if nxt is None:
            return {"sid": state["sid"], "levels": state["levels"], "miss": state.get("miss", 0) + 1}
        return {"sid": nxt[0], "levels": state["levels"] + nxt[1], "miss": state.get("miss", 0)}
    actions = sorted({k for (_, k) in table.keys()})
    w = World(name=f"arc3-{game}",
              description=f"Discovered world model for ARC-AGI-3 game {game} ({label}): masked-frame "
                          f"state, learned-table dynamics, reward = levels_completed.",
              initial_state={"sid": init_sid, "levels": 0, "miss": 0},
              actions=actions, transition=FunctionTransition(fn))
    w._objective = CodeObjective("def reward(state):\n    return float(state.get('levels', 0))\n",
                                 name="levels_completed", func_name="reward",
                                 description="induced reward: ARC-AGI-3 levels completed")
    return w


def _levelup_graph(deltas):
    lu = [i for i, d in enumerate(deltas) if d > 0]
    nodes = [{"id": 0, "label": ["start", "level 0"], "initial": True}]
    edges = []; prev = -1
    for k, idx in enumerate(lu):
        win = (k == len(lu) - 1)
        nodes.append({"id": k + 1, "label": (["win", f"level {k+1}"] if win else [f"level {k+1}"]),
                      "initial": False})
        edges.append({"src": k, "dst": k + 1, "action": f"{idx - prev} steps"}); prev = idx
    return {"kind": "state", "nodes": nodes, "edges": edges, "truncated": False}


def _level_curve(deltas, npts=40):
    cum = np.cumsum([int(d) for d in deltas]) if deltas else np.array([0])
    idxs = np.linspace(0, len(cum) - 1, min(npts, len(cum))).astype(int)
    return [int(cum[i]) for i in idxs]


def build_arm(arm):
    fname, label = ARMS[arm]
    arch = json.load(open(ROOT / "experiments" / "results" / fname))
    sols, pg = arch.get("solutions", {}), arch.get("per_game", {})
    # only games this arm FULLY solved (reached the win level)
    full = [g for g in sorted(sols) if sols[g]
            and pg.get(g, {}).get("levels", 0) > 0
            and pg.get(g, {}).get("levels") == pg.get(g, {}).get("win")]
    outdir = MAPS if arm == "fable" else MAPS / arm
    outdir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for g in full:
        try:
            frames, deltas, keys, depth = trace(g, sols[g])
            sids = state_ids(frames, indexed=False)
            table, det = build_table(sids, deltas, keys)
            if not det:
                sids = state_ids(frames, indexed=True); table, _ = build_table(sids, deltas, keys)
            world = make_world(g, sids[0], table, label)
            spec = to_spec(world)
            spec["preview"] = {"steps": len(deltas), "action": "solve trace",
                               "series": {"levels": _level_curve(deltas)},
                               "graph": _levelup_graph(deltas)}
            render_card(spec, path=str(outdir / f"{g}.svg"))
            ok += 1
            print(f"  {arm}/{g}: depth={depth} states={len(set(sids))} -> maps/{arm}/{g}.svg", flush=True)
        except Exception as e:
            print(f"  {arm}/{g}: ERROR {e}", flush=True)
    print(f"{arm}: rendered {ok}/{len(full)} atlas cards -> {outdir.relative_to(ROOT)}")


if __name__ == "__main__":
    arms = [a for a in sys.argv[1:] if a in ARMS] or ["opus", "codex"]
    for a in arms:
        build_arm(a)
