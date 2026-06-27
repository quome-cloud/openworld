"""T370 — hypothesis-driven goal predicates for bp35 (7/9) and lf52 (6/10).

Approach:
1. Random exploration to collect level-completion anchor frames
2. Hypothesize 5 goal predicates from anchors
3. BFS over verified world model targeting each predicate
4. Real-env verify each candidate sequence
5. Report best-effort closeness if no new level cleared

Usage:
    python3 t370_goal_predicates.py [--game bp35|lf52] [--explore 2000] [--bfs 1000]
"""
import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import arc_agi
from arcengine import GameAction

HERE = Path(__file__).resolve().parent
ACTS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4,
        GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]


def grid(obs):
    a = np.asarray(obs.frame)
    return a[-1].reshape(64, 64) if a.ndim == 3 else a.reshape(64, 64)


def load_world_model(game):
    path = HERE / "results" / "arc3_claude" / f"{game}.json"
    if not path.exists():
        return None, 0.0
    d = json.loads(path.read_text())
    ns = {"np": np, "numpy": np}
    try:
        exec(compile(d["code"], "<model>", "exec"), ns)  # noqa: S102
        return ns.get("predict"), d.get("verified_exact", 0.0)
    except Exception:
        return None, 0.0


# ─── Phase 1: collect transitions + level-completion anchor frames ───────────

def explore(game, n_steps, seed=42):
    """Run random exploration; capture (frame, levels_completed, step) at each level-up."""
    rng = random.Random(seed)
    env = arc_agi.Arcade().make(game)
    obs = env.reset()
    avail = obs.available_actions  # list of int action codes
    g = grid(obs)
    win = obs.win_levels
    transitions = []
    level_anchors = {}  # level -> frame at completion
    prev_lv = 0

    for step in range(n_steps):
        a_int = rng.choice(avail)
        ga = ACTS[a_int - 1]
        obs2 = env.step(ga)
        if obs2 is None or getattr(obs2, "frame", None) is None:
            obs = env.reset(); g = grid(obs); avail = obs.available_actions; continue
        g2 = grid(obs2)
        transitions.append((g.copy(), a_int, g2.copy(), obs2.levels_completed))
        if obs2.levels_completed > prev_lv:
            for lv in range(prev_lv + 1, obs2.levels_completed + 1):
                level_anchors[lv] = g2.copy()
                print(f"  [{game}] level {lv} cleared at step {step+1}", flush=True)
            prev_lv = obs2.levels_completed
        if str(obs2.state) != "GameState.NOT_FINISHED":
            obs = env.reset(); g = grid(obs); avail = obs.available_actions; prev_lv = 0; continue
        g = g2; avail = obs2.available_actions

    print(f"  [{game}] explore done: {len(transitions)} transitions, "
          f"anchors for levels {sorted(level_anchors)}", flush=True)
    return transitions, level_anchors, win


# ─── Phase 2: hypothesize goal predicates ────────────────────────────────────

def make_hypotheses(game, level_anchors, current_level, win_level):
    """Return list of (name, predicate_fn) for the next level after current_level."""
    anchors = sorted(level_anchors.keys())
    target_lv = current_level + 1

    hyps = []

    # H1: cell-value distribution matches the last anchor frame best
    if anchors:
        ref = level_anchors[anchors[-1]]
        h1_target = ref.copy()
        def h1(g, _ref=h1_target):
            return np.array_equal(g, _ref)
        hyps.append(("H1_exact_last_anchor", h1))

        # H2: Hamming distance to last anchor is small (< 10% cells differ)
        def h2(g, _ref=h1_target):
            diff = np.sum(g != _ref)
            return diff < (64 * 64 * 0.1)
        hyps.append(("H2_hamming_near_last_anchor", h2))

    # H3: specific counter patterns — row-63 or row-0 fill level
    # For bp35: row-63 fills with value 15; for lf52: row-0 fills with value 1
    if game == "bp35":
        # From world model: row 63 fills with 15 each step.
        # Hypothesis: level n completes when count reaches ~(target_lv * k) for some k
        # Derive k from anchors
        counts = {lv: int(np.count_nonzero(level_anchors[lv][63] == 15)) for lv in anchors}
        print(f"  bp35 row63 counts at level completions: {counts}", flush=True)
        if counts:
            avg_per_level = np.mean(list(counts.values())) if len(counts) == 1 else (
                max(counts.values()) / max(counts.keys()) if max(counts.keys()) > 0 else 16
            )
            threshold = int(avg_per_level * target_lv)
            def h3(g, _t=threshold):
                return int(np.count_nonzero(g[63] == 15)) >= _t
            hyps.append((f"H3_row63_count_ge_{threshold}", h3))

            # H4: counter wraps — after level completion the counter resets
            # The level completion trigger is: avatar reaches a target position
            # Hypothesis: any non-background color at a specific region
            # Use the diff between consecutive anchor frames to find the "goal" region
            if len(anchors) >= 2:
                diff_masks = []
                for i in range(len(anchors) - 1):
                    d = level_anchors[anchors[i]] != level_anchors[anchors[i+1]]
                    diff_masks.append(d)
                combined = np.sum(diff_masks, axis=0) > 0
                # Hypothesis: avatar (color 9) is NOT in the changed-region
                # (reached the target, so changed back)
                def h4(g, _mask=combined):
                    return int(np.sum(g[_mask] == 9)) == 0
                hyps.append(("H4_avatar_cleared_target_region", h4))

    elif game == "lf52":
        counts = {lv: int(np.count_nonzero(level_anchors[lv][0] == 1)) for lv in anchors}
        print(f"  lf52 row0 counts at level completions: {counts}", flush=True)
        if counts:
            avg = np.mean(list(counts.values())) if counts else 32
            threshold = min(64, int(avg * target_lv / max(anchors)) if anchors else int(avg))
            def h3(g, _t=threshold):
                return int(np.count_nonzero(g[0] == 1)) >= _t
            hyps.append((f"H3_row0_count_ge_{threshold}", h3))

            # H4: row 0 fully filled
            def h4(g):
                return int(np.count_nonzero(g[0] == 1)) >= 64
            hyps.append(("H4_row0_full", h4))

    # H5: differential — frame is "more like" any known anchor than start
    # Hamming distance to nearest anchor is minimal
    if anchors:
        anchor_set = [level_anchors[lv] for lv in anchors]
        def h5(g, _ancs=anchor_set):
            dists = [np.sum(g != ref) for ref in _ancs]
            return min(dists) < 100  # within 100 cells of any completion frame
        hyps.append(("H5_near_any_anchor", h5))

    return hyps[:5]  # at most 5


# ─── Phase 3: BFS over world model targeting hypothesis ──────────────────────

def bfs_hypothesis(predict, start, avail_ints, predicate, max_nodes=1000):
    """BFS through world model. Return (path, terminal_state) when predicate holds, or best."""
    if predict is None:
        return None, None, None

    start = np.asarray(start)
    q = deque([(start, [])])
    seen = {start.tobytes()}
    best_dist = None
    best_path = None
    best_state = None
    nodes = 0

    while q and nodes < max_nodes:
        state, path = q.popleft()
        for a_int in avail_ints:
            try:
                ns = np.asarray(predict(state, a_int))
            except Exception:
                continue
            if ns.shape != (64, 64):
                continue
            nodes += 1
            if predicate(ns):
                return path + [a_int], ns, nodes
            h = ns.tobytes()
            npath = path + [a_int]
            if h not in seen and len(npath) < 20:
                seen.add(h)
                q.append((ns, npath))

    # BFS didn't find it — return the closest state we saw
    # (measured by H2-style Hamming to the queue's first item)
    return None, None, nodes


# ─── Phase 4: real-env verify action sequence ────────────────────────────────

def verify_in_env(game, action_seq, current_level):
    """Replay action_seq in real env. Return (levels_achieved, terminal_frame)."""
    env = arc_agi.Arcade().make(game)
    obs = env.reset()
    avail = obs.available_actions
    g = grid(obs)
    best = obs.levels_completed

    for a_int in action_seq:
        if a_int not in avail:
            break
        ga = ACTS[a_int - 1]
        obs2 = env.step(ga)
        if obs2 is None or getattr(obs2, "frame", None) is None:
            break
        g = grid(obs2)
        best = max(best, obs2.levels_completed)
        avail = obs2.available_actions
        if str(obs2.state) != "GameState.NOT_FINISHED":
            break

    return best, g


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_game(game, explore_steps, bfs_nodes, starting_level, seed=42):
    print(f"\n{'='*60}", flush=True)
    print(f"[T370] {game} | starting at level {starting_level} | "
          f"explore {explore_steps} steps | BFS {bfs_nodes} nodes", flush=True)

    # Phase 1: explore
    transitions, level_anchors, win_level = explore(game, explore_steps, seed)

    # Load world model
    predict, fidelity = load_world_model(game)
    print(f"  world model fidelity: {fidelity}", flush=True)

    # Phase 2: hypothesize
    hyps = make_hypotheses(game, level_anchors, starting_level, win_level)
    print(f"  hypotheses: {[h[0] for h in hyps]}", flush=True)

    results = {
        "game": game, "starting_level": starting_level, "win_level": win_level,
        "anchors_found": sorted(level_anchors.keys()),
        "world_model_fidelity": fidelity, "best_level": starting_level,
        "hypothesis_results": []
    }

    if not level_anchors:
        print(f"  WARNING: no level anchors found in {explore_steps} steps. "
              f"Hypotheses cannot be grounded.", flush=True)
        # Fall back to random action sequences as "hypothesis-free BFS"
        results["hypothesis_results"].append({
            "name": "fallback_random",
            "verified": False,
            "new_levels": 0,
            "note": f"no anchors in {explore_steps} random steps"
        })
        return results

    # Start state: reset and replay to current_level-1 completion (use last transition)
    # For now, use a fresh reset as start state (level 0 position)
    env0 = arc_agi.Arcade().make(game)
    obs0 = env0.reset()
    start_frame = grid(obs0)
    avail_ints = obs0.available_actions

    # Phase 3+4: BFS + verify for each hypothesis
    for hyp_name, predicate in hyps:
        print(f"\n  Hypothesis {hyp_name}:", flush=True)
        path, model_state, nodes_explored = bfs_hypothesis(
            predict, start_frame, avail_ints, predicate, max_nodes=bfs_nodes
        )
        if path is None:
            print(f"    BFS: no solution found ({nodes_explored} nodes explored)", flush=True)
            hr = {"name": hyp_name, "bfs_found": False, "nodes": nodes_explored,
                  "verified": False, "new_levels": 0}
        else:
            print(f"    BFS: found path len={len(path)} ({nodes_explored} nodes)", flush=True)
            real_level, term_frame = verify_in_env(game, path, starting_level)
            new_levels = real_level - starting_level
            print(f"    verify: levels_achieved={real_level} (delta={new_levels})", flush=True)
            hr = {"name": hyp_name, "bfs_found": True, "nodes": nodes_explored,
                  "path_len": len(path), "verified": True,
                  "real_level": real_level, "new_levels": new_levels,
                  "solved": new_levels > 0}
            if real_level > results["best_level"]:
                results["best_level"] = real_level

        results["hypothesis_results"].append(hr)

    # Compute best-effort closeness for any anchor we found
    if level_anchors:
        best_anchor = level_anchors[max(level_anchors.keys())]
        env_t = arc_agi.Arcade().make(game)
        obs_t = env_t.reset()
        final_g = grid(obs_t)
        hamming = int(np.sum(final_g != best_anchor))
        results["hamming_to_best_anchor"] = hamming
        results["cells_differ_from_last_anchor"] = hamming
        print(f"\n  Best-effort closeness (Hamming to level {max(level_anchors.keys())} anchor): "
              f"{hamming} cells differ out of {64*64}", flush=True)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="bp35", choices=["bp35", "lf52", "both"])
    ap.add_argument("--explore", type=int, default=1500,
                    help="random steps to collect anchor frames")
    ap.add_argument("--bfs", type=int, default=800,
                    help="max BFS nodes per hypothesis")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    games = ["bp35", "lf52"] if args.game == "both" else [args.game]
    starting = {"bp35": 7, "lf52": 6}

    all_results = {}
    for game in games:
        r = run_game(game, args.explore, args.bfs, starting[game], args.seed)
        all_results[game] = r

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("T370 SUMMARY", flush=True)
    for game, r in all_results.items():
        best = r["best_level"]
        win = r["win_level"]
        start = r["starting_level"]
        print(f"  {game}: {start}/{win} → {best}/{win} "
              f"({'NEW LEVEL CLEARED' if best > start else 'no progress'})", flush=True)
        for hr in r["hypothesis_results"]:
            solved = hr.get("solved", False)
            print(f"    {hr['name']}: {'SOLVED' if solved else 'failed'} "
                  f"(new_levels={hr.get('new_levels', 0)})", flush=True)

    # Write output
    out = Path(args.out) if args.out else HERE / "results" / "t370_goal_predicates.json"
    out.write_text(json.dumps(all_results, indent=2, default=int))
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
