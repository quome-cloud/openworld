"""OFFLINE model validation (development tool — never part of scored runs).

Validates the synthesized symbolic model (crafter_model.py) against the real
env in two ways, mirroring the Baba arm's fidelity sweep but adapted to a
stochastic environment:

1. DETERMINISTIC CORE, lock-stepped: from the full pre-step state, predict
   the player-controlled component of the next state (pos, facing, sleeping,
   inventory, achievements, hidden vitals timers, terrain edits) and compare
   exactly. Steps where the outcome depends on randomness or on mob actions
   are EXCLUDED by rule, and the exclusions are counted:
     - sapling draw (p=0.1) -> inventory/achievement for sapling excluded
     - health/food(+cow)/wake excluded when a hostile could interact
       (zombie within 3, arrow within 2, skeleton within 6 pre-step)
   This uses env internals (privileged READ) — allowed here because model
   validation is offline; scored runs never import this module.

2. STOCHASTIC COMPONENTS, distributional: empirical event frequencies over
   many steps vs the source-derived constants in STOCHASTIC_SPEC:
     - cow move rate ~ 0.5 per updated tick
     - zombie chase rate ~ 0.9 when dist<=8
     - zombie melee damage = 2 (7 sleeping), cooldown 5
     - sapling success rate ~ 0.1
     - plant growth only within Manhattan distance < 18 of the player
"""

import json
import os
import sys
import warnings

warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import crafter  # noqa: E402
from crafter.objects import Zombie, Cow, Skeleton, Arrow, Plant  # noqa: E402

from crafter_model import predict_player_step, daylight  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def extract_state(env):
    p = env._player
    return dict(pos=tuple(int(v) for v in p.pos),
                facing=tuple(int(v) for v in p.facing),
                sleeping=p.sleeping,
                inventory=dict(p.inventory),
                achievements=dict(p.achievements),
                _hunger=p._hunger, _thirst=p._thirst,
                _fatigue=p._fatigue, _recover=p._recover)


def make_accessors(env):
    world = env._world

    def mat(pos):
        m, _ = world[pos]
        return m

    def obj(pos):
        _, o = world[pos]
        if o is None or o is env._player:
            return None
        kind = type(o).__name__.lower()
        return dict(kind=kind, health=getattr(o, 'health', 0),
                    ripe=getattr(o, 'ripe', False))
    return mat, obj


def hostile_near(env):
    p = env._player
    for o in env._world.objects:
        d = abs(int(o.pos[0]) - int(p.pos[0])) + abs(int(o.pos[1]) - int(p.pos[1]))
        if isinstance(o, Zombie) and d <= 3:
            return True
        if isinstance(o, Arrow) and d <= 2:
            return True
        if isinstance(o, Skeleton) and d <= 6:
            return True
    return False


def run_validation(episodes=20, steps_per_ep=1200, policy_seed=0):
    rng = np.random.default_rng(policy_seed)
    stats = dict(steps=0, compared=0, mismatches=[], excl_hostile=0,
                 excl_sapling=0, daylight_checked=0, daylight_mismatch=0)
    dist = dict(cow_updates=0, cow_moves=0, zombie_chase_opps=0,
                zombie_chase_moves=0, sapling_draws=0, sapling_hits=0,
                zombie_hits_taken=[], plant_frozen_checked=0,
                plant_frozen_grew=0)
    for ep in range(episodes):
        env = crafter.Env(area=(64, 64), view=(9, 9), size=(256, 256),
                          reward=True, seed=4000 + ep)
        env.reset()
        if ep % 2 == 0:
            # boosted-inventory episodes: make the random policy actually
            # exercise every make/place/collect gate (recipes never fire
            # under a pure random policy otherwise)
            env._player.inventory.update(dict(
                wood=9, stone=9, coal=9, iron=9, sapling=5,
                wood_pickaxe=1, stone_pickaxe=1, iron_pickaxe=1,
                wood_sword=1, stone_sword=1))
        # mixed policy: random with bursts of 'do' and movement to exercise
        # collection/crafting/place paths
        for t in range(steps_per_ep):
            state = extract_state(env)
            mat, obj = make_accessors(env)
            # remember mobs pre-step for distributional stats
            pre_mobs = [(type(o).__name__, tuple(int(v) for v in o.pos),
                         getattr(o, 'health', 0), o)
                        for o in env._world.objects if o is not env._player]
            pre_plants = [(o, o.grown) for o in env._world.objects
                          if isinstance(o, Plant)]
            hostile = hostile_near(env)
            if rng.random() < 0.45:
                action_idx = int(rng.integers(0, 17))
            elif rng.random() < 0.5:
                action_idx = 5  # do
            else:
                action_idx = int(rng.integers(1, 5))  # move
            action = env.action_names[action_idx]
            pred = predict_player_step(state, action, mat, obj)
            _, _, done, _ = env.step(action_idx)
            post = extract_state(env)
            stats['steps'] += 1
            # daylight formula check
            stats['daylight_checked'] += 1
            if abs(daylight(env._step) - env._world.daylight) > 1e-9:
                stats['daylight_mismatch'] += 1
            # comparisons
            excl_keys = set()
            if hostile:
                stats['excl_hostile'] += 1
                excl_keys |= {'health', 'sleeping', 'food'}  # cow kill food ok
            if pred['sapling_branch']:
                stats['excl_sapling'] += 1
                excl_keys |= {'sapling'}
                dist['sapling_draws'] += 1
                if post['inventory']['sapling'] > state['inventory']['sapling']:
                    dist['sapling_hits'] += 1
            ok = True
            notes = []
            if pred['pos'] != post['pos']:
                ok = False; notes.append(('pos', pred['pos'], post['pos']))
            if pred['facing'] != post['facing']:
                ok = False; notes.append(('facing', pred['facing'], post['facing']))
            if 'sleeping' not in excl_keys and pred['sleeping'] != post['sleeping']:
                ok = False; notes.append(('sleeping', pred['sleeping'], post['sleeping']))
            for k in pred['inventory']:
                if k in excl_keys or (k == 'health' and 'health' in excl_keys):
                    continue
                if k == 'health' and pred['inventory'][k] != post['inventory'][k]:
                    ok = False; notes.append((k, pred['inventory'][k], post['inventory'][k]))
                elif k != 'health' and pred['inventory'][k] != post['inventory'][k]:
                    ok = False; notes.append((k, pred['inventory'][k], post['inventory'][k]))
            for k in ('_hunger', '_thirst', '_fatigue', '_recover'):
                if abs(pred[k] - post[k]) > 1e-9 and not (
                        hostile and k in ('_hunger',)):
                    # _hunger reset by cow kill is modelled; only hostile
                    # interference (attacking a cow we predicted dead etc.)
                    ok = False; notes.append((k, pred[k], post[k]))
            for cell, m in pred['terrain_set'].items():
                real, _ = env._world[cell]
                if real != m:
                    ok = False; notes.append(('terrain', cell, m, real))
            for k, v in pred['achievements'].items():
                if k in ('collect_sapling',) and pred['sapling_branch']:
                    continue
                if hostile and k in ('wake_up', 'defeat_zombie',
                                     'defeat_skeleton', 'eat_cow'):
                    continue
                if post['achievements'].get(k, 0) != v:
                    ok = False; notes.append(('ach_' + k, v, post['achievements'].get(k, 0)))
            stats['compared'] += 1
            if not ok:
                stats['mismatches'].append(dict(ep=ep, t=t, action=action,
                                                notes=[str(n) for n in notes]))
            # distributional bookkeeping (uses pre/post mob positions)
            ppos = post['pos']
            for name, pos0, hp0, o in pre_mobs:
                if o.removed:
                    continue
                d0 = abs(pos0[0] - state['pos'][0]) + abs(pos0[1] - state['pos'][1])
                updated = d0 < 18
                pos1 = tuple(int(v) for v in o.pos)
                if name == 'Cow' and updated:
                    dist['cow_updates'] += 1
                    if pos1 != pos0:
                        dist['cow_moves'] += 1
                if name == 'Zombie' and updated and d0 <= 8 and pos1 != pos0:
                    # among zombies that DID move: fraction moving toward the
                    # player's pre-step position (chase w.p. 0.9; a random
                    # mover approaches ~<=0.5 of the time)
                    dist['zombie_chase_opps'] += 1
                    pp = state['pos']
                    if abs(pos1[0] - pp[0]) + abs(pos1[1] - pp[1]) < \
                            abs(pos0[0] - pp[0]) + abs(pos0[1] - pp[1]):
                        dist['zombie_chase_moves'] += 1
            for o, grown0 in pre_plants:
                if o.removed:
                    continue
                d0 = abs(int(o.pos[0]) - state['pos'][0]) + \
                    abs(int(o.pos[1]) - state['pos'][1])
                if d0 >= 18:
                    dist['plant_frozen_checked'] += 1
                    if o.grown != grown0:
                        dist['plant_frozen_grew'] += 1
            if done:
                break
    out = dict(deterministic=dict(
        steps=stats['steps'], compared=stats['compared'],
        mismatch_count=len(stats['mismatches']),
        mismatches=stats['mismatches'][:40],
        excluded_hostile_steps=stats['excl_hostile'],
        excluded_sapling_draws=stats['excl_sapling'],
        daylight=dict(checked=stats['daylight_checked'],
                      mismatches=stats['daylight_mismatch'])),
        distributional=dict(
            cow_move_rate=dict(
                observed=dist['cow_moves'] / max(1, dist['cow_updates']),
                expected=0.5, n=dist['cow_updates'],
                note='cow attempts a random-direction move w.p. 0.5; '
                     'observed rate is lower than 0.5 by the blocked-move '
                     'fraction (attempts into non-walkable cells fail '
                     'silently) — expected_success = 0.5 * P(free cell)'),
            zombie_chase_rate=dict(
                observed=dist['zombie_chase_moves'] / max(1, dist['zombie_chase_opps']),
                expected='>=0.9 among successful moves (chase p=0.9, long/'
                         'short axis 0.8/0.2; blocked chase steps drop out '
                         'of the moved-denominator)',
                n=dist['zombie_chase_opps']),
            sapling_rate=dict(
                observed=dist['sapling_hits'] / max(1, dist['sapling_draws']),
                expected=0.1, n=dist['sapling_draws']),
            plant_frozen_beyond_18=dict(
                checked=dist['plant_frozen_checked'],
                grew_anyway=dist['plant_frozen_grew'], expected=0)))
    return out


if __name__ == '__main__':
    out = run_validation()
    path = os.path.join(HERE, 'results', 'model_validation.json')
    json.dump(out, open(path, 'w'), indent=1)
    d = out['deterministic']
    print(f"deterministic: {d['compared']} steps compared, "
          f"{d['mismatch_count']} mismatches "
          f"({d['excluded_hostile_steps']} hostile-excluded, "
          f"{d['excluded_sapling_draws']} sapling draws), daylight "
          f"{d['daylight']['mismatches']}/{d['daylight']['checked']} wrong")
    print(json.dumps(out['distributional'], indent=1))
