"""Exploration dataset for the source-blind induction leg.

Collects transitions that competent play (conditions A/B) never shows:
random walks, action probing with and without prerequisites, deliberate
deaths by each major cause, and full lifecycle demos (plant ripening, ore
tour, sleep-attack). One .jsonl.gz per episode under results/transitions/,
names prefixed explore_.

Policy provenance: some scripted probes use env internals to TARGET things
(e.g. walk to the nearest lava cell to demonstrate lava death). The logged
dataset itself is exclusively the served channels (obs text + info as the
wrapper returns them), so induction inputs stay clean; the policy that chose
the actions is irrelevant to the induced model and is disclosed in each
file's meta header.
"""

import os
import sys
import warnings

warnings.simplefilter('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from run_suite import make_env, TransitionLogger, RESULTS, CRAFTER_TO_BALROG  # noqa: E402

ACTION_NAMES = list(CRAFTER_TO_BALROG.keys())


def nav_action(env, target, mine=False):
    """Greedy step toward target (env-internal targeting, see header).
    mine=True bores through blocking cells when already facing them."""
    p = env._player.pos
    dx, dy = int(target[0] - p[0]), int(target[1] - p[1])
    if abs(dx) >= abs(dy) and dx != 0:
        a = 'move_right' if dx > 0 else 'move_left'
    elif dy != 0:
        a = 'move_down' if dy > 0 else 'move_up'
    else:
        return 'noop'
    if mine:
        d = MOVE_D[a]
        tgt = (int(p[0] + d[0]), int(p[1] + d[1]))
        m, o = env._world[tgt]
        if o is None and m not in ('grass', 'path', 'sand') and \
                tuple(env._player.facing) == d and m != 'lava':
            return 'do'
    return a


def guarded(policy):
    """Wrap a policy: attack any adjacent zombie/skeleton first."""
    from crafter.objects import Zombie, Skeleton

    def wrapped(env, t, rng, state):
        p = env._player.pos
        for o in env._world.objects:
            if isinstance(o, (Zombie, Skeleton)):
                d = (int(o.pos[0] - p[0]), int(o.pos[1] - p[1]))
                if abs(d[0]) + abs(d[1]) == 1:
                    if tuple(env._player.facing) == d:
                        return 'do'
                    return {(-1, 0): 'move_left', (1, 0): 'move_right',
                            (0, -1): 'move_up', (0, 1): 'move_down'}[d]
        return policy(env, t, rng, state)
    return wrapped


def nearest(env, material):
    world = env._world
    mid = world._mat_ids.get(material)
    if mid is None:
        return None
    xs, ys = np.where(world._mat_map == mid)
    if not len(xs):
        return None
    p = env._player.pos
    i = np.argmin(np.abs(xs - p[0]) + np.abs(ys - p[1]))
    return int(xs[i]), int(ys[i])


MOVE_D = {'move_left': (-1, 0), 'move_right': (1, 0),
          'move_up': (0, -1), 'move_down': (0, 1)}


def lava_guard(env, action, rng):
    """Swap a move-into-lava for a safe move (policy-side guard; the
    deliberate lava-death episode disables it)."""
    if action not in MOVE_D:
        return action
    p = env._player.pos
    d = MOVE_D[action]
    m, _ = env._world[(int(p[0] + d[0]), int(p[1] + d[1]))]
    if m != 'lava':
        return action
    for a2, d2 in MOVE_D.items():
        m2, _ = env._world[(int(p[0] + d2[0]), int(p[1] + d2[1]))]
        if m2 != 'lava':
            return a2
    return 'noop'


def run(name, seed, policy, max_steps, boost=None, note='', guard=True):
    wrapper = make_env(seed)
    env = wrapper.env
    obs = wrapper.reset()
    if boost:
        env._player.inventory.update(boost)
    path = os.path.join(RESULTS, 'transitions', f'explore_{name}.jsonl.gz')
    tlog = TransitionLogger(path, dict(
        seed=seed, condition='exploration', policy=name, note=note,
        boost=boost or {}, actions=ACTION_NAMES))
    rng = np.random.default_rng(seed)
    state = {}
    for t in range(max_steps):
        action = policy(env, t, rng, state)
        if guard:
            action = lava_guard(env, action, rng)
        obs, reward, done, info = wrapper.step(CRAFTER_TO_BALROG[action])
        tlog.log(t + 1, action, reward, done, obs, info)
        if done:
            break
    tlog.close()
    ach = sum(1 for v in info['achievements'].items() if v[1] > 0)
    print(f'explore_{name}: {t+1} steps, done={done}, achievements={ach}')
    return info


def p_random(env, t, rng, state):
    return ACTION_NAMES[int(rng.integers(0, 17))]


def p_random_do_heavy(env, t, rng, state):
    r = rng.random()
    if r < 0.35:
        return 'do'
    if r < 0.55:
        return ACTION_NAMES[int(rng.integers(5, 17))]  # do/sleep/place/make
    return ACTION_NAMES[int(rng.integers(1, 5))]


def p_noop(env, t, rng, state):
    return 'noop'


def p_probe(env, t, rng, state):
    """Systematic probing: every action from every facing, incl. wrong-
    prerequisite crafting (empty inventory) and placement on bad terrain."""
    seq = state.setdefault('seq', [])
    if not seq:
        for a in ACTION_NAMES:
            for d in ('move_left', 'move_right', 'move_up', 'move_down'):
                seq.append(d)
                seq.append(a)
    return seq[t % len(seq)]


def p_lava_seeker(env, t, rng, state):
    tgt = nearest(env, 'lava')
    if tgt is None:
        return p_random(env, t, rng, state)
    return nav_action(env, tgt)


def p_sleep_at_night(env, t, rng, state):
    # wander until night, then sleep in the open -> demonstrates the
    # sleeping-zombie interaction (7 damage) and wake-on-hurt
    phase = env._step % 300
    if 140 <= phase <= 272:
        return 'sleep'
    return ACTION_NAMES[int(rng.integers(1, 5))]


def p_skeleton_magnet(env, t, rng, state):
    from crafter.objects import Skeleton
    sk = [o for o in env._world.objects if isinstance(o, Skeleton)]
    if not sk:
        tgt = nearest(env, 'path')
        return nav_action(env, tgt) if tgt else p_random(env, t, rng, state)
    p = env._player.pos
    s = min(sk, key=lambda o: abs(int(o.pos[0]) - p[0]) + abs(int(o.pos[1]) - p[1]))
    return nav_action(env, tuple(int(v) for v in s.pos))


def p_plant_lifecycle(env, t, rng, state):
    # boost gives saplings: place one, stand next to it until ripe, eat it
    from crafter.objects import Plant
    plants = [o for o in env._world.objects if isinstance(o, Plant)]
    p = env._player.pos
    if not plants:
        return 'place_plant' if t % 3 == 0 else \
            ACTION_NAMES[int(rng.integers(1, 5))]
    o = plants[0]
    d = abs(int(o.pos[0]) - p[0]) + abs(int(o.pos[1]) - p[1])
    if d > 1:
        return nav_action(env, tuple(int(v) for v in o.pos))
    if o.ripe:
        return nav_action(env, tuple(int(v) for v in o.pos)) \
            if tuple(env._player.facing) != \
            (int(o.pos[0] - p[0]), int(o.pos[1] - p[1])) else 'do'
    return 'noop'


def p_ore_tour(env, t, rng, state):
    """With full tools: mine coal -> iron -> diamond -> stone; drink; chop."""
    order = state.setdefault(
        'order', ['coal', 'iron', 'diamond', 'stone', 'tree', 'water'])
    while order:
        tgt = nearest(env, order[0])
        if tgt is None:
            order.pop(0)
            continue
        p = env._player.pos
        d = abs(tgt[0] - p[0]) + abs(tgt[1] - p[1])
        if d > 1:
            return nav_action(env, tgt, mine=True)
        f = (int(p[0] + env._player.facing[0]),
             int(p[1] + env._player.facing[1]))
        if f != tgt:
            return nav_action(env, tgt)
        if order[0] == 'water' and env._player.inventory['drink'] >= 9:
            order.pop(0)
            continue
        return 'do'
    return p_random_do_heavy(env, t, rng, state)


if __name__ == '__main__':
    import sys as _sys
    demos_only = '--demos' in _sys.argv
    if not demos_only:
        run('random_1', 5001, p_random, 800, note='uniform random policy')
    if not demos_only:
        run('random_2', 5002, p_random, 800, note='uniform random policy')
        run('random_3', 5003, p_random, 800, note='uniform random policy')
        run('noop_starve', 5004, p_noop, 1500,
            note='noop-only: vitals decay + starvation death demo')
        run('probe_no_prereq', 5005, p_probe, 900,
            note='systematic action x facing probing with EMPTY inventory: '
                 'demonstrates every failed-precondition no-op')
        run('probe_boosted', 5006, p_random_do_heavy, 1000,
            boost=dict(wood=9, stone=9, coal=9, iron=9, sapling=5,
                       wood_pickaxe=1, stone_pickaxe=1, iron_pickaxe=1,
                       wood_sword=1, stone_sword=1, iron_sword=1),
            note='boosted initial inventory: recipes/placement dynamics')
        run('lava_death', 5007, p_lava_seeker, 600, guard=False,
            note='deliberate lava walk -> instant death demo')
        run('sleep_attack', 5008, p_sleep_at_night, 900,
            note='open-air night sleep -> sleeping zombie damage demo')
        run('skeleton_death', 5009, p_skeleton_magnet, 900,
            note='walk into skeleton fire -> arrow damage demo')
    for s in (5110, 5310, 5410, 5510):
        info = run(f'plant_lifecycle_{s}', s, guarded(p_plant_lifecycle), 1400,
                   boost=dict(sapling=5, iron_sword=1),
                   note='plant place->grow->ripe->eat lifecycle demo')
        if info['achievements'].get('eat_plant'):
            break
    for s in (5011, 5111, 5211, 5311):
        info = run(f'ore_tour_{s}', s, guarded(p_ore_tour), 1800,
                   boost=dict(wood=9, wood_pickaxe=1, stone_pickaxe=1,
                              iron_pickaxe=1, iron_sword=1),
                   note='guided mining of every ore tier + drink/chop demos')
        if info['achievements'].get('collect_diamond'):
            break
