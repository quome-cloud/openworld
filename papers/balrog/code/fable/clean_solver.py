"""CLEAN test-time protocol for the Fable BALROG Baba Is AI experiment.

Removes every privileged channel from the episode loop:
  * initial state is reconstructed ONLY from the observation array returned by
    env.reset() (the env's native observation_space output: per cell the
    (type_idx, color_idx, 0) encoding of the TOP object) — no attribute access
    on env internals, no get_objects()/get_ruleset_text()/agent_pos;
  * NO verify_on_clone: the plan is executed OPEN-LOOP on the live env;
  * NO env-clone fallback search;
  * score = the real env's own win signal (levels > 0), nothing else.

Legitimate knowledge used by the observation parser (observation format spec,
not state): the OBJECT_TO_IDX / COLOR_TO_IDX / name_mapping tables that define
what the uint8 channels mean, the fact that actions are up/right/down/left,
and the benchmark's max_steps=100. These are the published interface of the
environment, equivalent to knowing what the words in a text observation mean.

Known observability gap (disclosed): the obs encodes only the top object per
cell (encoding_level=1). Cells with stacked objects at reset are therefore
reconstructed as single-object cells. Offline audit over the 120 seeded suite
instances shows the only stacked reset cells are (a) border corners with a
duplicated static Wall (behaviorally inert) and (b) ONE instance
(two_room-make_win-distr_win_rule family) with a distractor RuleObject hidden
UNDER the distractor win-rule RuleProperty block at (11,4). The parser cannot
see (b); if a plan disturbed that cell, open-loop execution could diverge from
the model's prediction. This is an irreducible limitation of the env's obs
encoding, not of the method.
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
from symbolic_model import ModelUnsupported, extract_rules
import fable_planner
from fable_solver import BABAISAI_TASKS, SEED_BASE, NUM_EPISODES

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results_fable', 'clean_protocol')
REPORT = os.path.join(HERE, 'FABLE_REPORT.md')


# ── observation parser (obs -> symbolic state) ────────────────────────────────

def _build_decode_tables():
    """Inverse of the env's published observation encoding tables."""
    from baba.world_object import (
        OBJECT_TO_IDX, COLOR_TO_IDX, name_mapping_inverted,
    )
    idx_to_object = {v: k for k, v in OBJECT_TO_IDX.items()}
    idx_to_color = {v: k for k, v in COLOR_TO_IDX.items()}
    return idx_to_object, idx_to_color, name_mapping_inverted


FLEX_TYPES = {'fball', 'fwall', 'fdoor', 'fkey', 'baba'}


def parse_obs(obs):
    """Reconstruct (W, H, symbolic_state) from the reset() observation array.

    obs: np.ndarray of shape (W, H, 3) uint8, cell (i, j) = top object's
    (type_idx, color_idx, 0). Raises ModelUnsupported for encodings outside
    the suite's feature set (rule_color etc.).
    """
    idx_to_object, idx_to_color, nm_inv = _build_decode_tables()
    W, H, C = obs.shape
    if C != 3:
        raise ModelUnsupported("unexpected obs channels: %d" % C)
    cells = []
    for j in range(H):
        for i in range(W):
            t_idx, c_idx, _ = (int(x) for x in obs[i, j])
            t = idx_to_object.get(t_idx)
            if t == 'empty' or t is None and t_idx == 1:
                cells.append(())
                continue
            if t == 'wall':
                cells.append((('W',),))
                continue
            if t in FLEX_TYPES:
                color = idx_to_color[c_idx]
                cells.append((('O', t, color),))
                continue
            if t == 'rule_object':
                name = idx_to_color[c_idx]          # rule blocks encode name
                cells.append((('RO', nm_inv.get(name, name)),))
                continue
            if t == 'rule_is':
                cells.append((('RI',),))
                continue
            if t == 'rule_property':
                name = idx_to_color[c_idx]
                cells.append((('RP', nm_inv.get(name, name)),))
                continue
            raise ModelUnsupported("obs cell (%d,%d) type %r" % (i, j, t))
    return W, H, tuple(cells)


# ── clean episode loop ────────────────────────────────────────────────────────

def log_line(msg):
    line = "%s %s" % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(REPORT, 'a') as f:
        f.write("- `%s`\n" % line)


def solve_episode_clean(task_id, ep, seed):
    np.random.seed(seed)
    game = Game(task_id)
    obs = game.reset()
    t0 = time.time()
    result = {
        'task_id': task_id, 'episode': ep, 'seed': seed, 'protocol': 'clean',
        'solved': False, 'progression': 0.0, 'steps': 0,
        'method': 'failed', 'actions': [],
    }
    try:
        W, H, state0 = parse_obs(np.asarray(obs))
        rules0 = extract_rules(state0, W, H)
        actions, stats = fable_planner.plan_symbolic(W, H, state0, rules0)
    except ModelUnsupported as e:
        result['model_unsupported'] = str(e)
        actions, stats = None, {'method': 'model_unsupported'}
    result.update({k: v for k, v in stats.items() if k != 'method'})
    result['method'] = stats.get('method', 'failed')

    if actions is not None:
        # OPEN-LOOP execution on the live env; the only env interaction.
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
    with open(os.path.join(out_dir, '%s_run_%02d.json'
                           % (task_clean, result['episode'])), 'w') as f:
        json.dump(result, f, indent=2)


def load_existing():
    all_results = {}
    base = os.path.join(RESULTS_DIR, 'babaisai')
    if not os.path.isdir(base):
        return all_results
    for tid in BABAISAI_TASKS:
        d = os.path.join(base, tid)
        if os.path.isdir(d):
            eps = []
            for fn in sorted(os.listdir(d)):
                if fn.endswith('.json'):
                    with open(os.path.join(d, fn)) as f:
                        eps.append(json.load(f))
            if eps:
                all_results[tid] = eps
    return all_results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = load_existing()
    for ti, task_id in enumerate(BABAISAI_TASKS):
        done_eps = {r['episode'] for r in all_results.get(task_id, [])}
        for ep in range(NUM_EPISODES):
            if ep in done_eps:
                continue
            seed = SEED_BASE + ti * 100 + ep       # identical to privileged run
            r = solve_episode_clean(task_id, ep, seed)
            save_episode(r)
            all_results.setdefault(task_id, []).append(r)
            log_line("[clean] %s ep%d seed=%d -> %s [%s] steps=%d %ss" % (
                task_id, ep, seed, 'SOLVED' if r['solved'] else 'FAILED',
                r['method'], r['steps'], r['elapsed_s']))

    # aggregate + divergence check vs privileged run
    per_task = {}
    total_prog = 0.0
    n_solved = n_eps = 0
    divergences = []
    priv_base = os.path.join(HERE, 'results_fable', 'babaisai')
    for tid, eps in all_results.items():
        prog = sum(r['progression'] for r in eps) / len(eps)
        total_prog += prog
        n_solved += sum(r['solved'] for r in eps)
        n_eps += len(eps)
        per_task[tid.replace('env/', '')] = {
            'solved': sum(r['solved'] for r in eps), 'episodes': len(eps),
            'mean_progression': round(prog, 4),
        }
        for r in eps:
            pf = os.path.join(priv_base, tid, '%s_run_%02d.json' % (
                tid.replace('/', '_'), r['episode']))
            if os.path.exists(pf):
                with open(pf) as f:
                    pr = json.load(f)
                if (pr['solved'] != r['solved']
                        or pr['actions'] != r['actions']):
                    divergences.append({
                        'task': tid, 'episode': r['episode'],
                        'privileged': {'solved': pr['solved'],
                                       'steps': pr['steps']},
                        'clean': {'solved': r['solved'], 'steps': r['steps']},
                        'same_actions': pr['actions'] == r['actions'],
                    })
    score = total_prog / len(BABAISAI_TASKS)
    out = {
        'protocol': ('clean: obs-only initial-state parse, no clone '
                     'verification, no fallback, open-loop execution, '
                     'scored by live env win signal'),
        'final_score_pct': round(100 * score, 2),
        'episodes_solved': n_solved,
        'episodes_run': n_eps,
        'sota_pct': 75.7,
        'delta_vs_sota_pp': round(100 * score - 75.7, 2),
        'divergences_vs_privileged_run': divergences,
        'per_task': per_task,
    }
    with open(os.path.join(HERE, 'results_fable',
                           'clean_protocol_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != 'per_task'},
                     indent=2))


if __name__ == '__main__':
    main()
