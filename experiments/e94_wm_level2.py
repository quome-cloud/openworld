"""E94 -- use the CODE WORLD MODEL to attack level 2 (which blind real-env search E93c could not).

Real-env search is forward-only and resets to level 1, so it can't tree-search level 2. A code world
model can: synthesize level-2 dynamics, then BFS *in the model* (free state-restore + backtracking)
to generate diverse candidate action sequences, and VERIFY the best in the real env (synthesize ->
plan -> verify). Reaches level 2 = code world model helped where blind search didn't.

  python3 e94_wm_level2.py --game sp80 --prefix 5,2,...   # prefix = the verified level-1 solution
"""
import argparse
import json
import random
import re
from collections import deque
from pathlib import Path

import numpy as np
import arc_agi
from arcengine import GameAction

import e86_arc3 as E
import arc3_graph as GR

HERE = Path(__file__).resolve().parent
ACTS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4,
        GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]


def grid(obs):
    a = np.asarray(obs.frame)
    return a[-1].reshape(64, 64) if a.ndim == 3 else a.reshape(64, 64)


def replay(env, prefix):
    obs = env.reset()
    for a in prefix:
        obs = env.step(ACTS[a - 1])
        if obs is None or getattr(obs, "frame", None) is None:
            return None
    return obs


def collect_level2(env, prefix, steps, seed):
    """From the level-2 start (after the level-1 solution), gather (frame,action,next) transitions."""
    rng = random.Random(seed); trans = []; base = None
    obs = replay(env, prefix)
    if obs is None:
        return [], None
    base_level = obs.levels_completed; g = grid(obs); start = g.copy(); avail = list(obs.available_actions)
    for _ in range(steps):
        a = rng.choice(avail); obs = env.step(ACTS[a - 1])
        if obs is None or getattr(obs, "frame", None) is None:
            obs = replay(env, prefix); g = grid(obs); continue
        ng = grid(obs)
        trans.append({"frame": g.tolist(), "action": a, "next": ng.tolist()})
        g = ng
        if obs.levels_completed != base_level or str(obs.state) != "GameState.NOT_FINISHED":
            obs = replay(env, prefix); g = grid(obs)  # stay in level 2
    return trans, start, avail


def synth_model(trans, rounds=4):
    if len(trans) < 12:
        return None, 0.0
    cut = len(trans) * 3 // 4
    acc, code = E.synthesize(trans[:cut] + trans, lambda p: E.claude_cli(p, timeout=600), rounds=rounds)
    # (synthesize splits internally; we just want the best code)
    ns = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<m>", "exec"), ns)  # noqa: S102
        return ns["predict"], acc
    except Exception:  # noqa: BLE001
        return None, acc


def tree_search(predict, start, avail, depth, max_nodes=20000):
    """BFS in the model from level-2 start; return diverse distinct-object-config states + their paths."""
    start = np.asarray(start)
    q = deque([(start, [])]); seen = {GR.objects_sig(start) if hasattr(GR, "objects_sig") else start.tobytes()}
    cands = []; nodes = 0
    def sig(f):
        objs, _ = GR.objects(f)
        return tuple(sorted((o["color"], o["size"], int(o["centroid"][0]) // 4, int(o["centroid"][1]) // 4) for o in objs))
    seen = {sig(start)}
    while q and nodes < max_nodes:
        st, seq = q.popleft()
        for a in avail:
            try:
                ns = np.asarray(predict(st, a))
            except Exception:  # noqa: BLE001
                continue
            if ns.shape != (64, 64):
                continue
            nodes += 1
            s = sig(ns)
            if s not in seen and len(seq) < depth:
                seen.add(s); nseq = seq + [a]
                q.append((ns, nseq)); cands.append(nseq)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sp80")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--collect", type=int, default=400)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--exec_cands", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    prefix = [int(x) for x in args.prefix.split(",") if x.strip()]
    arc = arc_agi.Arcade(); env = arc.make(args.game)
    win = env.reset().win_levels

    print(f"[e94/{args.game}] collecting level-2 transitions...", flush=True)
    trans, start, avail = collect_level2(env, prefix, args.collect, args.seed)
    print(f"[e94/{args.game}] {len(trans)} level-2 transitions; synthesizing model...", flush=True)
    predict, fid = synth_model(trans)
    res = {"game": args.game, "level2_transitions": len(trans), "level2_model_fidelity": round(fid, 4),
           "win_levels": int(win)}
    if predict is None:
        res["error"] = "no level-2 model"; print(f"[e94/{args.game}] no usable level-2 model", flush=True)
    else:
        print(f"[e94/{args.game}] level-2 model fidelity {fid:.3f}; tree-searching candidates...", flush=True)
        cands = tree_search(predict, start, avail, args.depth)
        print(f"[e94/{args.game}] {len(cands)} distinct-config candidate plans; executing in real env...", flush=True)
        best = 1; solved2 = False
        cands.sort(key=len, reverse=True)  # try deeper/novel ones first
        for i, plan in enumerate(cands[:args.exec_cands]):
            obs = replay(env, prefix)
            if obs is None:
                continue
            for a in plan:
                obs = env.step(ACTS[a - 1])
                if obs is None or getattr(obs, "frame", None) is None:
                    break
                if obs.levels_completed > best:
                    best = obs.levels_completed; solved2 = True
                    res["level2_plan"] = prefix + plan
                    print(f"[e94/{args.game}] LEVEL {best} via model-planned candidate {i}!", flush=True)
                    break
            if solved2:
                break
        res["best_levels"] = int(best); res["reached_level2"] = solved2
        print(f"[e94/{args.game}] best {best}/{win} reached_level2={solved2}", flush=True)
    out = Path(args.out) if args.out else HERE / "results" / f"e94_wm_level2_{args.game}.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
