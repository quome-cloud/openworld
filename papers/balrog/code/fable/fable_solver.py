"""Fable suite runner for BALROG Baba Is AI.

Per episode:
  1. build env, seed numpy for reproducibility, reset
  2. extract symbolic state; plan with fable_planner (BFS -> macro WA* -> prim WA*)
  3. replay-verify the plan on an env CLONE (any mismatch => misprediction,
     logged, fall back to env-clone BFS/A* search like the original solver)
  4. execute the verified plan on the REAL env instance; solved = levels > 0
  5. checkpoint JSON after every episode

Usage:
  python3 fable_solver.py suite            # full 40-task suite, 3 eps each
  python3 fable_solver.py tasks <t1> <t2>  # specific tasks (env/ prefix opt.)
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from baba_harness import Game
from symbolic_model import ModelUnsupported, extract_rules, extract_state
import fable_planner

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results_fable')
REPORT = os.path.join(HERE, 'FABLE_REPORT.md')
SEED_BASE = 550372
NUM_EPISODES = 3

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


def log_line(msg):
    line = "%s %s" % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(REPORT, 'a') as f:
        f.write("- `%s`\n" % line)


def verify_on_clone(game, actions):
    """Replay actions on a clone; return (verified_win, steps_to_win)."""
    g = game.clone()
    for idx, a in enumerate(actions):
        g.step(a)
        if g.done:
            return (g.levels > 0), idx + 1
    return False, len(actions)


def fallback_env_search(game, max_bfs=5000, max_astar=30000):
    """Env-clone BFS then A*-ish search from the CURRENT env state (the
    original solver's approach, kept as a safety net for mispredictions)."""
    from collections import deque
    t0 = time.time()
    start = game.clone()
    queue = deque([(start, [])])
    visited = {start.state_key()}
    nodes = 0
    while queue and nodes < max_bfs:
        curr, path = queue.popleft()
        nodes += 1
        if len(path) >= 100:
            continue
        for action in ['up', 'right', 'down', 'left']:
            nxt = curr.clone()
            nxt.step(action)
            if nxt.done and nxt.levels > 0:
                return path + [action], nodes, time.time() - t0
            if not nxt.done:
                k = nxt.state_key()
                if k not in visited:
                    visited.add(k)
                    queue.append((nxt, path + [action]))
    return None, nodes, time.time() - t0


def solve_episode(task_id, ep, seed):
    np.random.seed(seed)
    game = Game(task_id)
    game.reset()
    t0 = time.time()
    result = {
        'task_id': task_id, 'episode': ep, 'seed': seed,
        'solved': False, 'progression': 0.0, 'steps': 0,
        'method': 'failed', 'actions': [], 'misprediction': False,
    }

    try:
        W, H, state0 = extract_state(game._env)
        rules0 = extract_rules(state0, W, H)
    except ModelUnsupported as e:
        result['model_unsupported'] = str(e)
        actions, stats = None, {'method': 'model_unsupported'}
    else:
        actions, stats = fable_planner.plan_symbolic(W, H, state0, rules0)
    result.update({k: v for k, v in stats.items() if k != 'method'})
    result['method'] = stats.get('method', 'failed')

    if actions is not None:
        ok, steps_to_win = verify_on_clone(game, actions)
        if not ok:
            result['misprediction'] = True
            log_line("MISPREDICTION %s ep%d: plan len %d failed replay -> "
                     "env-clone fallback" % (task_id, ep, len(actions)))
            actions = None

    if actions is None and not result.get('model_unsupported'):
        fb_actions, fb_nodes, fb_t = (None, 0, 0.0)
        if result['misprediction'] or result['method'] == 'failed':
            fb_actions, fb_nodes, fb_t = fallback_env_search(game)
            result['fallback_nodes'] = fb_nodes
            result['fallback_s'] = round(fb_t, 1)
        if fb_actions is not None:
            actions = fb_actions
            result['method'] = 'env_bfs_fallback'

    if actions is not None:
        # execute on the real env
        for a in actions:
            game.step(a)
            if game.done:
                break
        result['solved'] = game.levels > 0
        result['progression'] = 1.0 if result['solved'] else 0.0
        result['steps'] = len(actions)
        result['actions'] = actions

    result['elapsed_s'] = round(time.time() - t0, 2)
    return result


def save_episode(result):
    task_clean = result['task_id'].replace('/', '_')
    out_dir = os.path.join(RESULTS_DIR, 'babaisai', result['task_id'])
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir,
                         '%s_run_%02d.json' % (task_clean, result['episode']))
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)


def update_summary(all_results):
    per_task = {}
    for tid, eps in all_results.items():
        per_task[tid.replace('env/', '')] = {
            'episodes': len(eps),
            'solved': sum(r['solved'] for r in eps),
            'mean_progression': round(
                sum(r['progression'] for r in eps) / max(len(eps), 1), 4),
            'methods': sorted({r['method'] for r in eps}),
            'mean_elapsed_s': round(
                sum(r['elapsed_s'] for r in eps) / max(len(eps), 1), 2),
            'mispredictions': sum(r['misprediction'] for r in eps),
        }
    n_tasks_total = len(BABAISAI_TASKS)
    prog_sum = sum(v['mean_progression'] for v in per_task.values())
    score_over_attempted = prog_sum / max(len(per_task), 1)
    score_over_suite = prog_sum / n_tasks_total
    summary = {
        'benchmark': 'BALROG / Baba Is AI',
        'suite_size': n_tasks_total,
        'tasks_attempted': len(per_task),
        'episodes_run': sum(v['episodes'] for v in per_task.values()),
        'episodes_solved': sum(v['solved'] for v in per_task.values()),
        'score_over_attempted_pct': round(100 * score_over_attempted, 2),
        'score_over_full_suite_pct': round(100 * score_over_suite, 2),
        'sota_pct': 75.7,
        'planner': ('Fable synthesized symbolic world model + '
                    'BFS -> macro weighted-A* (goal-regression heuristic, '
                    'frozen-block dead-end pruning) -> primitive WA*'),
        'per_task': per_task,
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def load_existing():
    """Resume support: load already-checkpointed episodes."""
    all_results = {}
    base = os.path.join(RESULTS_DIR, 'babaisai')
    if not os.path.isdir(base):
        return all_results
    for tid in BABAISAI_TASKS:
        d = os.path.join(base, tid)
        if not os.path.isdir(d):
            continue
        eps = []
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.json'):
                with open(os.path.join(d, fn)) as f:
                    eps.append(json.load(f))
        if eps:
            all_results[tid] = eps
    return all_results


def run_tasks(task_ids, resume=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = load_existing() if resume else {}
    for ti, task_id in enumerate(task_ids):
        if not task_id.startswith('env/'):
            task_id = 'env/' + task_id
        done_eps = {r['episode'] for r in all_results.get(task_id, [])}
        for ep in range(NUM_EPISODES):
            if ep in done_eps:
                continue
            seed = SEED_BASE + BABAISAI_TASKS.index(task_id) * 100 + ep
            r = solve_episode(task_id, ep, seed)
            save_episode(r)
            all_results.setdefault(task_id, []).append(r)
            log_line("%s ep%d seed=%d -> %s [%s] steps=%d %ss%s" % (
                task_id, ep, seed,
                'SOLVED' if r['solved'] else 'FAILED',
                r['method'], r['steps'], r['elapsed_s'],
                ' MISPRED' if r['misprediction'] else ''))
        summary = update_summary(all_results)
        log_line("  running score over attempted: %.2f%% (%d tasks)" % (
            summary['score_over_attempted_pct'], summary['tasks_attempted']))
    return update_summary(all_results)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'suite'
    if mode == 'suite':
        summary = run_tasks(BABAISAI_TASKS)
    elif mode == 'tasks':
        summary = run_tasks(sys.argv[2:])
    else:
        raise SystemExit('usage: fable_solver.py suite | tasks <ids>')
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_task'},
                     indent=2))
