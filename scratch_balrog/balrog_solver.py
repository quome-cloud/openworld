"""
E140 recipe solver for BALROG Baba Is AI benchmark.

Recipe: EXPLORE -> MODEL -> GOAL -> PLAN -> SAVE

Two-phase planner:
  Phase 1 — BFS (up to MAX_BFS_NODES): optimal, handles all goto/break_stop tasks.
  Phase 2 — A* with make_win heuristic (up to MAX_ASTAR_NODES): if BFS fails, uses
             a domain heuristic (rule-block alignment cost + agent-to-block distance)
             to guide search through two_room-make_win and similar tasks.

Memory: each plan node stores one env clone (~200KB). BFS capped at 5K nodes (1GB);
A* terminates much earlier due to heuristic guidance.
"""
import heapq
import json
import os
import time
from collections import deque

from baba_harness import Game, ACTIONS

MAX_DEPTH = 200
MAX_BFS_NODES = 5_000       # Phase 1: BFS ceiling
MAX_ASTAR_NODES = 50_000    # Phase 2: A* ceiling (fewer real expansions expected)
NUM_EPISODES = 3


# ── EXPLORE ──────────────────────────────────────────────────────────────────

def explore(game: Game) -> dict:
    """Return a structured snapshot of the initial game state."""
    return {
        "agent_pos": game.agent_pos,
        "rules": game.get_ruleset_text(),
        "win_positions": game.get_win_positions(),
        "stop_positions": game.get_stop_positions(),
        "wall_positions": game.get_wall_positions(),
        "objects": game.get_objects(),
        "grid_size": (game.width, game.height),
    }


# ── MODEL + PLAN ──────────────────────────────────────────────────────────────

def _reconstruct(base_clone: Game, path: list) -> Game:
    """Replay action path from base clone to recover game state."""
    g = base_clone.clone()
    for a in path:
        g.step(a)
    return g


def _heuristic(game: Game) -> float:
    """Inadmissible but useful heuristic for make_win tasks.

    Returns 0 when WIN is achievable immediately (agent adjacent to WIN object).
    Returns lower values for states closer to winning.
    """
    win_pos = game.get_win_positions()
    if win_pos:
        ax, ay = game.agent_pos
        return min(abs(ax - wx) + abs(ay - wy) for wx, wy in win_pos)

    # No WIN rule yet — estimate cost to create one.
    # Use minimum pairwise alignment cost across all (ro, ri, rp) triplets.
    objects = game.get_objects()
    ros = objects.get('rule_object', [])
    ris = objects.get('rule_is', [])
    rps = objects.get('rule_property', [])
    if not (ros and ris and rps):
        return 100  # missing pieces, penalise

    def md(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Best triplet: minimise distance to align (ro adjacent to ri, ri adjacent to rp)
    best = float('inf')
    for ro in ros:
        for ri in ris:
            for rp in rps:
                cost = md(ro, ri) + md(ri, rp)
                best = min(best, cost)

    ax, ay = game.agent_pos
    agent_to_nearest = min(
        md((ax, ay), p)
        for p in ros + ris + rps
    )
    return best + agent_to_nearest


def bfs_plan(game: Game, max_depth: int = MAX_DEPTH,
             max_nodes: int = MAX_BFS_NODES) -> tuple:
    """Phase 1: BFS — optimal, works for all goto/break_stop tasks."""
    game.reset()
    initial_key = game.state_key()
    queue = deque([(game.clone(), [])])
    visited = {initial_key}
    nodes = 0

    while queue and nodes < max_nodes:
        curr, path = queue.popleft()
        nodes += 1
        if len(path) >= max_depth:
            continue

        for action in ACTIONS:
            nxt = curr.clone()
            nxt.step(action)

            if nxt.done and nxt.levels > 0:
                return path + [action], nodes

            if not nxt.done:
                k = nxt.state_key()
                if k not in visited:
                    visited.add(k)
                    queue.append((nxt, path + [action]))

    return None, nodes


def astar_plan(game: Game, max_depth: int = MAX_DEPTH,
               max_nodes: int = MAX_ASTAR_NODES) -> tuple:
    """Phase 2: weighted A* with make_win heuristic (inadmissible, w=3).

    Trades optimality for speed on two_room-make_win / break_stop-make_win.
    Dramatically fewer node expansions than blind BFS when heuristic guides well.
    """
    game.reset()
    initial_key = game.state_key()
    g0 = game.clone()
    h0 = _heuristic(g0)

    # heap: (f=g+w*h, g_cost, tie_breaker, clone, path)
    W = 3
    counter = [0]
    pq = [(h0 * W, 0, 0, g0, [])]
    visited = {initial_key}
    nodes = 0

    while pq and nodes < max_nodes:
        f, g_cost, _, curr, path = heapq.heappop(pq)
        nodes += 1
        if len(path) >= max_depth:
            continue

        for action in ACTIONS:
            nxt = curr.clone()
            nxt.step(action)

            if nxt.done and nxt.levels > 0:
                return path + [action], nodes

            if not nxt.done:
                k = nxt.state_key()
                if k not in visited:
                    visited.add(k)
                    new_g = g_cost + 1
                    h = _heuristic(nxt)
                    counter[0] += 1
                    heapq.heappush(pq, (new_g + W * h, new_g, counter[0], nxt,
                                        path + [action]))

    return None, nodes


def plan(game: Game) -> tuple:
    """Two-phase planner: BFS first, A* fallback for make_win tasks."""
    actions, nodes = bfs_plan(game)
    if actions is not None:
        return actions, nodes, "bfs"

    game.reset()  # reset for A* (bfs_plan left it mid-episode)
    actions, astar_nodes = astar_plan(game)
    return actions, nodes + astar_nodes, "astar" if actions else "failed"


# ── SAVE ──────────────────────────────────────────────────────────────────────

def save_result(result: dict, output_dir: str, task_id: str, episode: int):
    task_clean = task_id.replace("/", "_")
    out_path = os.path.join(output_dir, "babaisai", task_id)
    os.makedirs(out_path, exist_ok=True)
    fname = os.path.join(out_path, f"{task_clean}_run_{episode:02d}.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    return fname


# ── FULL TASK SOLVE ───────────────────────────────────────────────────────────

def solve_task(task_id: str, num_episodes: int = NUM_EPISODES,
               output_dir: str = "results", verbose: bool = True) -> list:
    """Run E140 recipe on one BALROG task for all episodes."""
    results = []

    for ep in range(num_episodes):
        t0 = time.time()
        game = Game(task_id)  # new random seed per episode
        game.reset()

        # EXPLORE
        state = explore(game)
        if verbose:
            print(f"  [{task_id}] ep{ep}: agent={state['agent_pos']} "
                  f"win={state['win_positions']} rules=[{state['rules']}]")

        # MODEL + GOAL + PLAN (BFS → A* fallback)
        actions, nodes, method = plan(game)
        elapsed = time.time() - t0

        if actions is None:
            solved = False
            progression = 0.0
            steps = 0
            if verbose:
                print(f"    -> NO SOLUTION ({nodes} nodes, {elapsed:.1f}s)")
        else:
            steps = len(actions)
            solved = True
            progression = 1.0
            if verbose:
                print(f"    -> SOLVED [{method}] in {steps} steps, "
                      f"{nodes} nodes, {elapsed:.1f}s")

        result = {
            "task_id": task_id,
            "episode": ep,
            "solved": solved,
            "progression": progression,
            "steps": steps,
            "method": method if actions else "failed",
            "actions": actions or [],
            "bfs_nodes": nodes,
            "elapsed_s": round(elapsed, 2),
        }
        results.append(result)
        save_result(result, output_dir, task_id, ep)

    return results


# ── SUITE ─────────────────────────────────────────────────────────────────────

BABAISAI_TASKS = [
    "env/make_win-distr_obj_rule",
    "env/goto_win-distr_obj_rule",
    "env/goto_win",
    "env/goto_win-distr_obj",
    "env/goto_win-distr_rule",
    "env/goto_win-distr_obj-irrelevant_rule",
    "env/make_win-distr_obj",
    "env/make_win-distr_rule",
    "env/make_win",
    "env/make_win-distr_obj-irrelevant_rule",
    "env/two_room-goto_win",
    "env/two_room-goto_win-distr_obj_rule",
    "env/two_room-goto_win-distr_rule",
    "env/two_room-goto_win-distr_obj",
    "env/two_room-goto_win-distr_obj-irrelevant_rule",
    "env/two_room-goto_win-distr_win_rule",
    "env/two_room-break_stop-goto_win-distr_obj_rule",
    "env/two_room-break_stop-goto_win-distr_obj",
    "env/two_room-break_stop-goto_win-distr_rule",
    "env/two_room-break_stop-goto_win-distr_obj-irrelevant_rule",
    "env/two_room-break_stop-goto_win",
    "env/two_room-maybe_break_stop-goto_win-distr_obj_rule",
    "env/two_room-maybe_break_stop-goto_win",
    "env/two_room-maybe_break_stop-goto_win-distr_obj",
    "env/two_room-maybe_break_stop-goto_win-distr_rule",
    "env/two_room-maybe_break_stop-goto_win-distr_obj-irrelevant_rule",
    "env/two_room-make_win-distr_obj_rule",
    "env/two_room-make_win-distr_rule",
    "env/two_room-make_win",
    "env/two_room-make_win-distr_obj-irrelevant_rule",
    "env/two_room-make_win-distr_obj",
    "env/two_room-make_win-distr_win_rule",
    "env/two_room-break_stop-make_win-distr_obj_rule",
    "env/two_room-break_stop-make_win-distr_rule",
    "env/two_room-break_stop-make_win",
    "env/two_room-break_stop-make_win-distr_obj-irrelevant_rule",
    "env/two_room-break_stop-make_win-distr_obj",
    "env/two_room-make_you",
    "env/two_room-make_you-make_win",
    "env/two_room-make_wall_win",
]


def run_suite(output_dir: str = "results", verbose: bool = True):
    """Run E140 recipe on all 40 BALROG Baba Is AI tasks."""
    all_results = {}
    total_solved = 0
    total_episodes = 0

    print(f"Running BALROG Baba Is AI suite ({len(BABAISAI_TASKS)} tasks × "
          f"{NUM_EPISODES} episodes = {len(BABAISAI_TASKS) * NUM_EPISODES} episodes)")
    print(f"Output: {output_dir}/")
    print()

    for task_id in BABAISAI_TASKS:
        print(f"Task: {task_id}")
        task_results = solve_task(task_id, num_episodes=NUM_EPISODES,
                                  output_dir=output_dir, verbose=verbose)
        all_results[task_id] = task_results

        solved = sum(r["solved"] for r in task_results)
        total_solved += solved
        total_episodes += len(task_results)
        print(f"  => {solved}/{NUM_EPISODES} episodes solved")
        print()

    score = total_solved / total_episodes if total_episodes > 0 else 0.0
    print(f"=== SUITE SCORE: {total_solved}/{total_episodes} = {score:.1%} ===")
    print(f"=== SOTA baseline: 75.7% (Gemini-3.1-Pro-Thinking) ===")
    print(f"=== Delta vs SOTA: {score - 0.757:+.1%} ===")

    summary = {
        "score": round(score, 4),
        "total_solved": total_solved,
        "total_episodes": total_episodes,
        "sota_baseline": 0.757,
        "delta_vs_sota": round(score - 0.757, 4),
        "per_task": {
            tid: {
                "solved": sum(r["solved"] for r in rs),
                "total": len(rs),
                "progression_mean": sum(r["progression"] for r in rs) / len(rs),
            }
            for tid, rs in all_results.items()
        },
    }
    with open(os.path.join(output_dir, "suite_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
