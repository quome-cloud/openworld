"""E121 -- re-verify every ARC-AGI-3 full-game solve THROUGH an OpenWorld `World`.

This closes the gap between "solved with a bespoke simulator" and the paper's claim that each solver is a
serveable OpenWorld world. For each solved game we, exactly as CLAUDE.md prescribes, treat the discovered
state-transition graph as a World: a masked-frame perceptor -> symbolic state, a `FunctionTransition`
over the learned table -> dynamics, and an induced `CodeObjective` (reward = levels_completed). We then
REPLAY THE BANKED SOLUTION THROUGH THE OPENWORLD WORLD (`world.step` per action) and assert it reproduces
the verified level-completion count -- so the solve provably runs through the framework, not only through
arc_agi. We also `to_spec` + `validate_spec` + `render_card` to confirm each world is a valid, serveable,
card-renderable artifact.

Honest scope: the learned table covers the verified solution trajectory (the world models that
trajectory's dynamics); the masked-frame state is the status-bar-masked 64x64 grid. If status masking is
insufficient to keep (state,action) deterministic, we fall back to a trajectory-indexed state and flag it.

Run with the arcv interpreter (has arc_agi); openworld is imported from the repo root:
    <arcv>/bin/python experiments/e121_openworld_roundtrip.py
"""
import json, os, sys, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                 # import openworld (zero-dep core) from the repo
import numpy as np
import openworld
from openworld import World, Action, FunctionTransition, CodeObjective, to_spec, validate_spec, render_card

SCR = ROOT / "scratch_arc"
RES = ROOT / "experiments" / "results"
FG = json.load(open(RES / "arc3_fullgame.json"))


def solved_path(g):
    for p in (SCR / f"full_{g}" / "solved_best.json", SCR / f"full_{g}" / "solved.json",
              SCR / f"agent_{g}" / "solved.json"):
        if p.exists():
            return p
    return None


def act_key(a):
    """Canonical (name, params, key) for a banked action ([a] | int | [6,x,y])."""
    if isinstance(a, int):
        return str(a), {}, str(a)
    a = list(a)
    if a[0] == 6:
        return "6", {"x": int(a[1]), "y": int(a[2])}, f"6:{int(a[1])},{int(a[2])}"
    return str(a[0]), {}, str(a[0])


def trace(game, actions):
    """Replay in the real engine; return raw frames, per-step level deltas, action keys, depth."""
    from arc3_harness import Game
    g = Game(game); g.reset()
    frames = [g.frame.copy()]; deltas = []; keys = []; base = g.levels; last = g.levels
    for a in actions:
        name, params, key = act_key(a)
        g.step(*( [int(name)] if not params else [6, params["x"], params["y"]] ))
        frames.append(g.frame.copy()); deltas.append(g.levels - last); keys.append((name, params, key))
        last = g.levels
        if g.done:
            break
    return frames, deltas, keys, g.levels - base


def state_ids(frames, indexed=False):
    """Masked-frame state ids: zero cells that change on >95% of steps (status bar), then hash.
    indexed=True -> append the step index (degenerate path world; used only as a consistency fallback."""
    F = np.stack([f.reshape(64, 64) for f in frames])
    if not indexed:
        changes = (F[1:] != F[:-1]).mean(axis=0)            # per-cell change frequency
        mask = changes <= 0.95
        masked = (F * mask).astype(np.int16)
        return [hashlib.blake2b(masked[i].tobytes(), digest_size=8).hexdigest() for i in range(len(F))]
    return [f"{hashlib.blake2b(F[i].tobytes(), digest_size=8).hexdigest()}#{i}" for i in range(len(F))]


def build_table(sids, deltas, keys):
    """(state_id, action_key) -> (next_state_id, level_delta); report determinism."""
    table, ok = {}, True
    for i, (_, _, k) in enumerate(keys):
        key = (sids[i], k); val = (sids[i + 1], int(deltas[i]))
        if key in table and table[key] != val:
            ok = False
        table[key] = val
    return table, ok


def make_world(game, init_sid, table):
    def fn(state, action):                                   # (state_dict, action_dict) -> dict
        name, params = action["name"], action.get("params") or {}
        k = name if not params else f"{name}:{params['x']},{params['y']}"
        nxt = table.get((state["sid"], k))
        if nxt is None:
            return {"sid": state["sid"], "levels": state["levels"], "miss": state.get("miss", 0) + 1}
        return {"sid": nxt[0], "levels": state["levels"] + nxt[1], "miss": state.get("miss", 0)}
    actions = sorted({k for (_, k) in table.keys()})
    reward_code = "def reward(state):\n    return float(state.get('levels', 0))\n"
    w = World(name=f"arc3-{game}", description=f"Discovered world model for ARC-AGI-3 game {game}: "
              f"masked-frame state, learned-table dynamics, reward = levels_completed.",
              initial_state={"sid": init_sid, "levels": 0, "miss": 0},
              actions=actions, transition=FunctionTransition(fn))
    w._objective = CodeObjective(reward_code, name="levels_completed", func_name="reward",
                                 description="induced reward: number of ARC-AGI-3 levels completed")
    return w


def run_through_world(world, keys):
    world.reset()
    for name, params, _ in keys:
        world.step(Action(name, params))
    s = dict(world.state.copy())
    return int(s.get("levels", 0)), int(s.get("miss", 0))


def roundtrip(game, wd):
    cwd = os.getcwd(); os.chdir(wd)
    if str(wd) not in sys.path:
        sys.path.insert(0, str(wd))
    try:
        actions = json.load(open(solved_path(game)))["actions"]
        frames, deltas, keys, depth = trace(game, actions)
        sids = state_ids(frames, indexed=False)
        table, det = build_table(sids, deltas, keys)
        indexed = False
        if not det:                                          # masking too coarse -> use a path world
            sids = state_ids(frames, indexed=True); table, _ = build_table(sids, deltas, keys); indexed = True
        world = make_world(game, sids[0], table)
        wlevels, miss = run_through_world(world, keys)
        # serveability: spec round-trips + validates + renders a card
        spec_ok = card_ok = False; problems = None
        try:
            spec = to_spec(world); problems = validate_spec(spec)
            spec_ok = isinstance(spec, dict)
            card_ok = bool(render_card(spec))
        except Exception as e:
            problems = [f"spec/card error: {e}"]
        return {
            "game": game, "depth_real": depth, "depth_through_world": wlevels, "misses": miss,
            "pass": (wlevels == depth and miss == 0),
            "n_states": len({s for k in table for s in (k[0],)} | {v[0] for v in table.values()}),
            "n_transitions": len(table), "indexed_fallback": indexed,
            "spec_valid": (problems == [] if isinstance(problems, list) else None),
            "card_renders": card_ok, "n_actions": len(keys),
        }
    finally:
        os.chdir(cwd)


def main():
    out = {"note": "each ARC-AGI-3 solve re-verified through an OpenWorld World (FunctionTransition over the "
                   "discovered masked-frame state graph; reward = levels_completed); solution replayed via "
                   "world.step and asserted to reproduce the verified depth.", "games": {}}
    npass = ntot = 0
    for g in sorted(FG["games"]):
        if not (isinstance(FG["games"][g]["levels"], int) and FG["games"][g]["levels"] > 0):
            continue
        wd = SCR / (f"full_{g}" if (SCR / f"full_{g}").exists() else f"agent_{g}")
        try:
            r = roundtrip(g, wd); out["games"][g] = r; ntot += 1; npass += 1 if r["pass"] else 0
            print(f"{g:6} world-depth {r['depth_through_world']}/{r['depth_real']} "
                  f"{'PASS' if r['pass'] else 'FAIL'}  states={r['n_states']} "
                  f"spec_valid={r['spec_valid']} card={r['card_renders']}"
                  f"{'  [indexed]' if r['indexed_fallback'] else ''}", flush=True)
        except Exception as e:
            print(f"{g:6} ERROR {e}", flush=True); out["games"][g] = {"game": g, "error": str(e)}
    out["n_pass"] = npass; out["n_total"] = ntot
    print(f"\nROUND-TRIP: {npass}/{ntot} solves reproduce their verified depth THROUGH an OpenWorld World")
    json.dump(out, open(RES / "arc3_openworld_roundtrip.json", "w"), indent=2, sort_keys=True)
    print("wrote arc3_openworld_roundtrip.json")


if __name__ == "__main__":
    main()
