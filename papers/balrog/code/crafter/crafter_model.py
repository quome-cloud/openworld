"""Symbolic world model of Crafter (crafter==1.8.3), synthesized by Fable 5
from reading the environment source (crafter/{env,engine,objects,worldgen}.py,
data.yaml). LLM-free at runtime: pure code + data.

Structure
=========
1. Exact constants (recipes, tool gates, walkability, item caps) — mirrors
   data.yaml, which the env loads into `constants`.
2. Deterministic core — exact one-step predictors for everything in the
   player's own update and for the parts of the world the player edits:
   movement, facing, do/collect, place, make, vitals timers, daylight.
   These are validated by lock-stepped rollouts (validate_model.py).
3. Stochastic transition spec — the randomness in the env, written down as
   distributions with their exact source-derived parameters. These cannot be
   lock-stepped pointwise (they consume the env's private RNG stream); they
   are validated distributionally and handled by the planner via closed-loop
   replanning + worst-case bounds.

Coordinate conventions (match crafter): pos = (x, y); x grows east, y grows
south. facing in {(-1,0) west, (1,0) east, (0,-1) north, (0,1) south}.

Update order inside env.step(action) — source-derived, load-bearing:
  1. step counter += 1; daylight updated from step counter.
  2. player.action set; objects updated in insertion order — the PLAYER is
     always the first object (added in reset before worldgen), so the
     player's action resolves before any mob moves. Only objects with
     Manhattan distance < 2*max(view) = 18 from the player are updated at
     all (mobs and plants beyond 18 are frozen — plants don't grow there).
  3. Every 10th step: chunk balancing (mob spawn/despawn; 12x12 chunks).
  4. Observation rendered (after all updates).
Consequence: mob positions in the observation are exactly the positions the
player acts against next step; a move onto a cell that is free in the obs
cannot be blocked by a mob "moving first". Player one-step outcomes are
therefore deterministic given the pre-step state (sole exception: sapling
collection succeeds with p=0.1).
"""

import math

# ── 1. Exact constants (data.yaml) ──────────────────────────────────────────

MATERIALS = [
    None, 'water', 'grass', 'stone', 'path', 'sand', 'tree', 'lava',
    'coal', 'iron', 'diamond', 'table', 'furnace']
MAT_ID = {m: i for i, m in enumerate(MATERIALS)}

WALKABLE = frozenset({'grass', 'path', 'sand'})
PLAYER_WALKABLE = WALKABLE | {'lava'}   # player can step on lava -> health=0

# Semantic-view object ids append after materials, in this exact order:
SEM_OBJECTS = ['player', 'cow', 'zombie', 'skeleton', 'arrow', 'plant']
ID_TO_ITEM = [str(m) for m in MATERIALS] + SEM_OBJECTS  # index 0 -> 'None'

ITEM_MAX = {  # inventory clamp [0, max] applied at end of every player update
    'health': 9, 'food': 9, 'drink': 9, 'energy': 9,
    'sapling': 9, 'wood': 9, 'stone': 9, 'coal': 9, 'iron': 9, 'diamond': 9,
    'wood_pickaxe': 9, 'stone_pickaxe': 9, 'iron_pickaxe': 9,
    'wood_sword': 9, 'stone_sword': 9, 'iron_sword': 9}
ITEM_INITIAL = {k: (9 if k in ('health', 'food', 'drink', 'energy') else 0)
                for k in ITEM_MAX}

COLLECT = {
    'tree':    dict(require={}, receive={'wood': 1},    leaves='grass', prob=1.0),
    'stone':   dict(require={'wood_pickaxe': 1}, receive={'stone': 1},   leaves='path', prob=1.0),
    'coal':    dict(require={'wood_pickaxe': 1}, receive={'coal': 1},    leaves='path', prob=1.0),
    'iron':    dict(require={'stone_pickaxe': 1}, receive={'iron': 1},   leaves='path', prob=1.0),
    'diamond': dict(require={'iron_pickaxe': 1}, receive={'diamond': 1}, leaves='path', prob=1.0),
    'water':   dict(require={}, receive={'drink': 1},   leaves='water', prob=1.0),
    'grass':   dict(require={}, receive={'sapling': 1}, leaves='grass', prob=0.1),  # STOCHASTIC
}

PLACE = {
    'stone':   dict(uses={'stone': 1}, where={'grass', 'sand', 'path', 'water', 'lava'}, type='material'),
    'table':   dict(uses={'wood': 2},  where={'grass', 'sand', 'path'}, type='material'),
    'furnace': dict(uses={'stone': 4}, where={'grass', 'sand', 'path'}, type='material'),
    'plant':   dict(uses={'sapling': 1}, where={'grass'}, type='object'),
}

MAKE = {
    'wood_pickaxe':  dict(uses={'wood': 1},                       nearby=('table',),           gives=1),
    'stone_pickaxe': dict(uses={'wood': 1, 'stone': 1},           nearby=('table',),           gives=1),
    'iron_pickaxe':  dict(uses={'wood': 1, 'coal': 1, 'iron': 1}, nearby=('table', 'furnace'), gives=1),
    'wood_sword':    dict(uses={'wood': 1},                       nearby=('table',),           gives=1),
    'stone_sword':   dict(uses={'wood': 1, 'stone': 1},           nearby=('table',),           gives=1),
    'iron_sword':    dict(uses={'wood': 1, 'coal': 1, 'iron': 1}, nearby=('table', 'furnace'), gives=1),
}
# 'nearby' = material must appear in the 3x3 window centred on the player.

ACHIEVEMENTS = [
    'collect_coal', 'collect_diamond', 'collect_drink', 'collect_iron',
    'collect_sapling', 'collect_stone', 'collect_wood', 'defeat_skeleton',
    'defeat_zombie', 'eat_cow', 'eat_plant', 'make_iron_pickaxe',
    'make_iron_sword', 'make_stone_pickaxe', 'make_stone_sword',
    'make_wood_pickaxe', 'make_wood_sword', 'place_furnace', 'place_plant',
    'place_stone', 'place_table', 'wake_up']

DIRS = {'west': (-1, 0), 'east': (1, 0), 'north': (0, -1), 'south': (0, 1)}
DIR_ACTION = {(-1, 0): 'move_left', (1, 0): 'move_right',
              (0, -1): 'move_up', (0, 1): 'move_down'}

SWORD_DAMAGE = [('iron_sword', 5), ('stone_sword', 3), ('wood_sword', 2)]
MOB_HP = {'zombie': 5, 'skeleton': 3, 'cow': 3}
ZOMBIE_MELEE = 2          # 7 if player sleeping
ZOMBIE_MELEE_SLEEPING = 7
ZOMBIE_ATTACK_COOLDOWN = 5  # after an attack, 5 ticks pass before the next
ARROW_DAMAGE = 2
UPDATE_RADIUS = 18        # objects farther (Manhattan) than this are frozen
EAT_COW_FOOD = 6          # also resets hunger timer
EAT_PLANT_FOOD = 4
PLANT_RIPE_AGE = 301      # ripe iff grown > 300 (grown +1 per updated step)

VIEW_X, VIEW_Y = 4, 3     # text obs window: x-4..x+4, y-3..y+3 (9x7)
WORLD_AREA = (64, 64)
SPAWN = (32, 32)          # player always spawns at area centre


def damage(inventory):
    """Player melee damage given inventory (max over owned sword tiers, min 1)."""
    for item, dmg in SWORD_DAMAGE:
        if inventory.get(item, 0) > 0:
            return dmg
    return 1


# ── 2. Deterministic core: daylight & vitals ────────────────────────────────

def daylight(step):
    """Exact copy of env._update_time (env.py:135)."""
    progress = (step / 300) % 1 + 0.3
    return 1 - abs(math.cos(math.pi * progress)) ** 3


def is_night(step, threshold=0.5):
    return daylight(step) < threshold

# Night window per 300-step cycle (daylight < 0.5): t%300 in [148, 272].
# Zombie spawn pressure starts once daylight <= 0.833 (target >= 1):
# t%300 in [~127, ~287].


def predict_vitals(v, sleeping):
    """One-step update of (hunger, thirst, fatigue, recover) timers and the
    food/drink/energy/health deltas they trigger. `v` is a dict with keys
    _hunger,_thirst,_fatigue,_recover, food, drink, energy, health, sleeping.
    Mirrors Player._update_life_stats + _degen_or_regen_health exactly.
    Returns a new dict (health delta excludes mob damage, which is applied
    after the player's own update)."""
    v = dict(v)
    v['_hunger'] += 0.5 if sleeping else 1
    if v['_hunger'] > 25:
        v['_hunger'] = 0
        v['food'] -= 1
    v['_thirst'] += 0.5 if sleeping else 1
    if v['_thirst'] > 20:
        v['_thirst'] = 0
        v['drink'] -= 1
    if sleeping:
        v['_fatigue'] = min(v['_fatigue'] - 1, 0)
    else:
        v['_fatigue'] += 1
    if v['_fatigue'] < -10:
        v['_fatigue'] = 0
        v['energy'] += 1
    if v['_fatigue'] > 30:
        v['_fatigue'] = 0
        v['energy'] -= 1
    necessities = (v['food'] > 0, v['drink'] > 0, v['energy'] > 0 or sleeping)
    if all(necessities):
        v['_recover'] += 2 if sleeping else 1
    else:
        v['_recover'] -= 0.5 if sleeping else 1
    if v['_recover'] > 25:
        v['_recover'] = 0
        v['health'] += 1
    if v['_recover'] < -15:
        v['_recover'] = 0
        v['health'] -= 1
    # clamp (applies to all inventory incl. vitals)
    for k in ('food', 'drink', 'energy'):
        v[k] = max(0, min(v[k], 9))
    v['health'] = max(0, min(v['health'], 9))
    return v


# ── 2b. Deterministic core: full player one-step predictor ─────────────────
# Used by validate_model.py for lock-stepped validation, and by the planner
# for legality/effect checks. State is a plain dict extracted from the env
# (or from belief): pos, facing, inventory (incl. vitals), sleeping,
# timers (_hunger.._recover), mat(pos)->material fn, obj(pos)->obj-kind fn.

def predict_player_step(state, action, mat, obj):
    """Predict the player-controlled part of one env step.

    state: dict(pos, facing, sleeping, inventory, _hunger,_thirst,_fatigue,
                _recover, achievements(dict name->count))
    action: crafter action string.
    mat(p) -> material name at p (pre-step), obj(p) -> None or dict(kind=...,
        ripe=bool for plants) for the object at p (pre-step, player excluded).

    Returns dict(pos, facing, sleeping, inventory, timers..., achievements,
                 terrain_set: {pos: material} world edits,
                 object_removed: pos or None       (plant eaten -> not removed;
                                                    fence n/a; mobs die via hp),
                 mob_damage: (pos, dmg) or None    damage dealt to mob at pos,
                 sapling_branch: bool              True if outcome depends on
                                                    the p=0.1 sapling draw)
    Exact except: when sapling_branch is True the inventory/achievement for
    sapling is left at the "no receive" branch and the caller must accept
    either. Mob damage TO the player is not included (happens after)."""
    pos = tuple(state['pos']); facing = tuple(state['facing'])
    inv = dict(state['inventory']); ach = dict(state['achievements'])
    sleeping = state['sleeping']
    timers = {k: state[k] for k in ('_hunger', '_thirst', '_fatigue', '_recover')}
    target = (pos[0] + facing[0], pos[1] + facing[1])
    tmat, tobj = mat(target), obj(target)
    out_terrain = {}
    mob_damage = None
    object_removed = None
    sapling_branch = False
    woke = False

    act = action
    if sleeping:
        if inv['energy'] < 9:
            act = 'sleep'
        else:
            sleeping = False
            ach['wake_up'] = ach.get('wake_up', 0) + 1
            woke = True

    if act == 'noop':
        pass
    elif act.startswith('move_'):
        d = {'left': (-1, 0), 'right': (1, 0), 'up': (0, -1), 'down': (0, 1)}[act[5:]]
        facing = d
        t2 = (pos[0] + d[0], pos[1] + d[1])
        m2, o2 = mat(t2), obj(t2)
        if o2 is None and m2 in PLAYER_WALKABLE:
            pos = t2
        if mat(pos) == 'lava':   # post-move cell (only changes if moved)
            inv['health'] = 0
    elif act == 'do' and tobj is not None:
        dmg = damage(inv)
        kind = tobj['kind']
        if kind == 'plant':
            if tobj.get('ripe'):
                inv['food'] += EAT_PLANT_FOOD
                ach['eat_plant'] = ach.get('eat_plant', 0) + 1
                # plant grown reset to 0, stays alive
        elif kind in ('zombie', 'skeleton', 'cow'):
            mob_damage = (target, dmg)
            if tobj.get('health', MOB_HP[kind]) - dmg <= 0:
                if kind == 'zombie':
                    ach['defeat_zombie'] = ach.get('defeat_zombie', 0) + 1
                elif kind == 'skeleton':
                    ach['defeat_skeleton'] = ach.get('defeat_skeleton', 0) + 1
                elif kind == 'cow':
                    inv['food'] += EAT_COW_FOOD
                    ach['eat_cow'] = ach.get('eat_cow', 0) + 1
                    timers['_hunger'] = 0
        # note: removal of dead mob happens in the MOB's own update
    elif act == 'do':
        if tmat == 'water':
            timers['_thirst'] = 0
        info = COLLECT.get(tmat)
        if info is not None:
            if all(inv.get(k, 0) >= v for k, v in info['require'].items()):
                out_terrain[target] = info['leaves']
                if info['prob'] >= 1.0:
                    for name, amount in info['receive'].items():
                        inv[name] = inv.get(name, 0) + amount
                        ach[f'collect_{name}'] = ach.get(f'collect_{name}', 0) + 1
                else:
                    sapling_branch = True  # p=0.1: receive+achievement or not
    elif act == 'sleep':
        if inv['energy'] < 9:
            sleeping = True
    elif act.startswith('place_'):
        name = act[6:]
        info = PLACE[name]
        if tobj is None and tmat in info['where'] and \
                all(inv.get(k, 0) >= v for k, v in info['uses'].items()):
            for k, v in info['uses'].items():
                inv[k] -= v
            if info['type'] == 'material':
                out_terrain[target] = name
            ach[f'place_{name}'] = ach.get(f'place_{name}', 0) + 1
            # plant: new Plant object appears at target (grown=0)
    elif act.startswith('make_'):
        name = act[5:]
        info = MAKE[name]
        near = {mat((pos[0] + dx, pos[1] + dy))
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        if all(u in near for u in info['nearby']) and \
                all(inv.get(k, 0) >= v for k, v in info['uses'].items()):
            for k, v in info['uses'].items():
                inv[k] -= v
            inv[name] = inv.get(name, 0) + info['gives']
            ach[f'make_{name}'] = ach.get(f'make_{name}', 0) + 1

    # vitals update (uses the possibly-updated sleeping flag)
    v = dict(timers, food=inv['food'], drink=inv['drink'],
             energy=inv['energy'], health=inv['health'])
    v = predict_vitals(v, sleeping)
    inv['food'], inv['drink'], inv['energy'], inv['health'] = \
        v['food'], v['drink'], v['energy'], v['health']
    timers = {k: v[k] for k in ('_hunger', '_thirst', '_fatigue', '_recover')}
    # clamp remaining inventory
    for k in inv:
        inv[k] = max(0, min(inv[k], ITEM_MAX.get(k, 9)))
    # _wake_up_when_hurt: needs last_health bookkeeping; health can only DROP
    # here via lava (=0) or vitals degen; both wake the player.
    # (Handled by comparing health in the validator.)

    return dict(pos=pos, facing=facing, sleeping=sleeping, inventory=inv,
                achievements=ach, terrain_set=out_terrain,
                mob_damage=mob_damage, object_removed=object_removed,
                sapling_branch=sapling_branch, woke=woke, **timers)


# ── 3. Stochastic transition spec (source-derived, per-tick) ────────────────
# Not pointwise-predictable (consumes the env's private RNG). Written down
# here both as documentation and for distributional validation.

STOCHASTIC_SPEC = {
    'sapling_collect': {
        'source': 'objects.py Player._do_material / data.yaml collect.grass',
        'dist': 'Bernoulli(0.1) per do-on-grass; terrain unchanged either way'},
    'cow': {
        'source': 'objects.py Cow.update',
        'dist': 'w.p. 0.5 attempt move in uniform random direction (blocked '
                'moves fail silently); dies at hp<=0 (removed on its next '
                'update); only updates within Manhattan 18 of player'},
    'zombie': {
        'source': 'objects.py Zombie.update',
        'dist': 'if dist(player)<=8: w.p. 0.9 step toward player (long axis '
                'w.p. 0.8 else short axis) else uniform random step; then if '
                'dist<=1: attack (2 dmg, 7 if player sleeping) unless '
                'cooldown>0; cooldown=5 after attack, else cooldown-=1',
        'worst_case': 'closes 1 cell/tick; max melee dps = 2 per 6 ticks '
                      'per zombie (adjacent), first hit immediate'},
    'skeleton': {
        'source': 'objects.py Skeleton.update',
        'dist': 'if dist<=3: retreat step (away, long axis w.p. 0.6); if that '
                'move failed/skipped and dist<=5: w.p. 0.5 shoot arrow toward '
                'player (reload 4 ticks); elif dist<=8: w.p. 0.3 approach; '
                'else w.p. 0.2 random step',
        'worst_case': 'arrow every 5 ticks while in 4..5 band; arrows fly 1 '
                      'cell/tick along a cardinal, 2 dmg on any object hit'},
    'arrow': {
        'source': 'objects.py Arrow.update',
        'dist': 'DETERMINISTIC flight: moves 1 cell/tick along facing; on '
                'object: 2 dmg + arrow removed; on non-walkable material: '
                'removed (destroys table/furnace -> path!); creation is the '
                'only stochastic part (skeleton shoot)'},
    'spawn_balance': {
        'source': 'env.py _balance_chunk/_balance_object (every 10th step, '
                  'per 12x12 chunk)',
        'dist': 'zombie: on grass, target max 3.5-3*daylight (0 by day, 3.5 '
                'at night), spawn w.p. 0.3/check at uniform grass cell of the '
                'chunk if >=6 Manhattan from player; despawn w.p. 0.4 when '
                'over target. skeleton: on path chunks (>=6 path cells), '
                'target 1..2, spawn w.p. 0.1 if >=7 from player. cow: on '
                'grass (>=30 cells), target 1.5+daylight, spawn w.p. 0.01, '
                'despawn w.p. 0.1 when over, >=5 from player.',
        'planner_consequence': 'zombie pressure at night is unbounded over '
                               'time -> shelter or kill; day despawns them.'},
    'worldgen': {
        'source': 'worldgen.py (opensimplex + uniform draws)',
        'dist': 'fixed per episode by seed at reset; deterministic within an '
                'episode. Materials: grass blob at spawn; mountains (stone/'
                'path tunnels/coal p~0.15*|simplex>0|/iron p~0.25*gate/'
                'diamond p~0.006*gate/lava) ; water/sand rings; trees p~0.2 '
                'on grass gate. Objects: cows on grass, zombies dist>10, '
                'skeletons in tunnels.'},
}

# What CANNOT be lock-stepped and why (report §stochasticity):
#   Mob action choices, spawn/despawn events and the sapling draw consume
#   np.random.RandomState draws private to the env, interleaved with
#   rendering draws (LocalView._noise consumes RNG when daylight<0.5).
#   Reproducing them would require bit-mirroring the entire RNG call
#   sequence including night-rendering noise — a full env reimplementation,
#   at which point the "model" is the env. We instead (a) lock-step the
#   deterministic core exactly, (b) validate the stochastic components
#   distributionally (empirical frequencies vs the constants above).
