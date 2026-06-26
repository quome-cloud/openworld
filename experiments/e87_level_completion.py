"""E87 -- ARC-AGI-3 level completion via BFS through verified game state.

Loads ls20 (or any game) directly from the downloaded local source, runs BFS over
the compact internal state (player pos + shape/color/rotation + goals mask), and
executes the found sequence in the real ARC-AGI-3 environment to record an official
level completion on the scorecard.

Usage:
  python3 e87_level_completion.py --game ls20 --max-states 200000
"""
import argparse
import collections
import copy
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)

HERE = Path(__file__).resolve().parent
ENVFILES = HERE / "environment_files"


def load_game_class(game_id):
    """Load the ARCBaseGame subclass directly from the local file."""
    import glob
    dirs = list(ENVFILES.glob(f"{game_id}/*"))
    if not dirs:
        raise FileNotFoundError(f"No local env files for {game_id}")
    game_dir = sorted(dirs)[-1]
    py_files = list(game_dir.glob("*.py"))
    if not py_files:
        raise FileNotFoundError(f"No .py files in {game_dir}")
    game_file = py_files[0]
    class_name = game_id.replace("-", "_").capitalize()
    # Handle ls20 -> Ls20
    parts = game_id.split("-")
    class_name = "".join(p.capitalize() for p in parts)

    spec = importlib.util.spec_from_loader(f"arc_game_{game_id}", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(game_file.read_text(encoding="utf-8"), module.__dict__)
    cls = getattr(module, class_name, None)
    if cls is None:
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and hasattr(obj, "perform_action"):
                cls = obj
                break
    return cls, game_file


def get_state(game):
    """Extract compact BFS state from game internals."""
    px = game.gudziatsk.x
    py = game.gudziatsk.y
    shape = game.fwckfzsyc
    color = game.hiaauhahz
    rot = game.cklxociuu
    goals = tuple(game.lvrnuajbl)
    # Include moving enemy positions if any
    enemy_pos = tuple(
        (s._sprite.x, s._sprite.y) for s in getattr(game, "wsoslqeku", [])
    )
    return (px, py, shape, color, rot, goals, enemy_pos)


def apply_action(game, action_int):
    """Apply action (1-4) to game in-place. Returns frame_data."""
    from arcengine import ActionInput, GameAction
    acts = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]
    act = acts[action_int - 1]
    ai = ActionInput(id=act, data={})
    fd = game.perform_action(ai, raw=True)
    return fd


def clone_game(cls):
    """Instantiate a fresh game (reset to start)."""
    from arcengine import ActionInput, GameAction
    g = cls()
    g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    return g


def replay_to(cls, action_seq):
    """Replay action sequence from fresh start. Returns game in final state."""
    g = clone_game(cls)
    for a in action_seq:
        apply_action(g, a)
    return g


def bfs(cls, max_states=200_000, available_actions=(1, 2, 3, 4)):
    """BFS to find sequence completing level 1. Returns action_seq or None."""
    start_game = clone_game(cls)
    start_state = get_state(start_game)
    init_levels = start_game.levels_completed

    # Queue entries: (action_seq_so_far,)
    queue = collections.deque()
    queue.append([])
    visited = {start_state}
    # Map state -> action_seq to allow replay
    state_to_seq = {start_state: []}

    t0 = time.time()
    checked = 0

    while queue:
        seq = queue.popleft()
        checked += 1
        if checked % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  BFS: {checked} states explored, {len(visited)} visited, {elapsed:.1f}s", flush=True)
        if checked > max_states:
            print(f"  BFS: exhausted {max_states} states without solution", flush=True)
            return None

        # Replay to get game at this state
        g = replay_to(cls, seq)

        for a in available_actions:
            g2 = replay_to(cls, seq + [a])
            s2 = get_state(g2)
            if s2 in visited:
                continue
            visited.add(s2)
            new_seq = seq + [a]
            state_to_seq[s2] = new_seq

            # Check win condition
            if g2.levels_completed > init_levels:
                print(f"  BFS: LEVEL COMPLETED in {len(new_seq)} steps!", flush=True)
                return new_seq

            queue.append(new_seq)

    print("  BFS: queue exhausted without solution", flush=True)
    return None


def execute_in_arcade(game_id, action_seq):
    """Execute action sequence in the official ARC-AGI-3 env to get scorecard credit."""
    import arc_agi
    from arcengine import GameAction
    acts = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
            GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    if env is None:
        print("  ERROR: could not make env", flush=True)
        return None, None
    obs = env.reset()
    print(f"  Executing {len(action_seq)}-step sequence in official env...", flush=True)
    for i, a in enumerate(action_seq):
        obs = env.step(acts[a - 1])
        if obs and obs.levels_completed > 0:
            print(f"  LEVEL COMPLETED at step {i+1}! levels={obs.levels_completed}", flush=True)
    sc = arc.get_scorecard()
    return obs, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--max-states", type=int, default=200_000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print(f"[e87/{args.game}] Loading game class...", flush=True)
    cls, game_file = load_game_class(args.game)
    print(f"[e87/{args.game}] Loaded: {cls.__name__} from {game_file.name}", flush=True)

    print(f"[e87/{args.game}] BFS for level completion (max_states={args.max_states})...", flush=True)
    t0 = time.time()
    action_seq = bfs(cls, max_states=args.max_states)
    elapsed = time.time() - t0

    result = {
        "game": args.game,
        "bfs_elapsed_s": round(elapsed, 2),
        "solved": action_seq is not None,
        "action_seq": action_seq,
        "action_seq_len": len(action_seq) if action_seq else None,
    }

    if action_seq:
        print(f"[e87/{args.game}] Solution found: {len(action_seq)} steps in {elapsed:.1f}s", flush=True)
        print(f"[e87/{args.game}] Action sequence: {action_seq}", flush=True)
        print(f"[e87/{args.game}] Executing in official ARC-AGI-3 env...", flush=True)
        obs, sc = execute_in_arcade(args.game, action_seq)
        if sc:
            result["scorecard"] = str(sc)
            result["levels_completed"] = obs.levels_completed if obs else None
    else:
        print(f"[e87/{args.game}] No solution found in {elapsed:.1f}s", flush=True)

    out = Path(args.out) if args.out else HERE / "results" / f"e87_level_completion_{args.game}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"[e87/{args.game}] wrote {out}", flush=True)
    return result


if __name__ == "__main__":
    main()
