"""Fable-synthesized fast symbolic world model of Baba Is AI (BALROG suite).

This is a pure-Python, exact reimplementation of the step semantics of
baba/grid.py (BabaIsYouEnv.step / BabaIsYouGrid) for the feature subset that
can occur in the 40-task BALROG babaisai suite. It exists because tree search
over `copy.deepcopy(env)` costs ~13 ms/node; this model steps in ~30-60 us,
a ~250x speedup, which converts "search-budget timeout" failures into solves.

Faithfulness notes (deliberately bug-compatible with the env):
  * Cells are stacks (objects can sit on top of overlappable objects).
  * Rules are extracted from horizontal AND vertical 3-lines around each IS
    block, reading only the TOP object of each cell.
  * Push chains recurse; a pushed block moves iff its own forward cell (after
    its own chain) is empty or overlappable.
  * win/lose are evaluated BEFORE the mover is placed, against the PRE-move
    ruleset; with multiple agents, the LAST-moving agent's result overwrites
    earlier ones (env bug, replicated).
  * A blocked move checks win on the mover's own cell (top = mover itself),
    so "BABA IS WIN" + bump-into-wall wins. Replicated.
  * "X IS Y" (two rule_objects) appends a default-colored Y object ON TOP of
    every cell whose top is an X (env's grid.replace uses grid.set which
    appends). Replicated.
  * Implicit properties: IS YOU and IS PULL imply IS STOP.
  * Ruleset is recomputed after all movement; returned rules are pre-replace
    (matching env, which does not re-extract after replace until next step).

State representation:
  state = tuple over cells (row-major, k = j*W + i) of tuples (stack,
  bottom -> top) of descriptors:
    ('W',)               static wall (baba.world_object.Wall)  - never moves
    ('O', otype, color)  FlexibleWorldObj, otype in
                         {'baba','fball','fkey','fdoor','fwall'}
    ('RO', otype)        RuleObject (pushable)
    ('RI',)              RuleIs     (pushable)
    ('RP', prop)         RuleProperty, prop like 'is_goal'    (pushable)

Anything else (rule colors, non-push rule blocks, open/shut/move/pull rules)
raises ModelUnsupported and the caller falls back to env-clone search.
"""

from collections import namedtuple

DIRS = {'up': (0, -1), 'right': (1, 0), 'down': (0, 1), 'left': (-1, 0)}
ACTIONS = ['up', 'right', 'down', 'left']

DEFAULT_COLOR = {
    'fwall': 'grey', 'fball': 'green', 'fkey': 'blue',
    'fdoor': 'red', 'baba': 'white',
}

MAX_STEPS = 100

Rules = namedtuple('Rules', ['you', 'stop', 'goal', 'defeat', 'push', 'replace'])

EMPTY = frozenset()


class ModelUnsupported(Exception):
    pass


# ── extraction from the real env ─────────────────────────────────────────────

def desc_of(e):
    from baba.world_object import (
        Wall, RuleObject, RuleIs, RuleProperty, RuleColor, RuleBlock,
        FlexibleWorldObj,
    )
    if isinstance(e, RuleColor):
        raise ModelUnsupported("rule_color block")
    if isinstance(e, RuleObject):
        if not e.is_push():
            raise ModelUnsupported("non-push rule block")
        return ('RO', e.object)
    if isinstance(e, RuleIs):
        if not e.is_push():
            raise ModelUnsupported("non-push rule block")
        return ('RI',)
    if isinstance(e, RuleProperty):
        if not e.is_push():
            raise ModelUnsupported("non-push rule block")
        return ('RP', e.property)
    if isinstance(e, RuleBlock):
        raise ModelUnsupported("unknown rule block %r" % e)
    if isinstance(e, FlexibleWorldObj):
        return ('O', e.type, e.color)
    if isinstance(e, Wall):
        return ('W',)
    raise ModelUnsupported("unknown object %r" % e)


def extract_state(env):
    """env: the raw BabaIsYouEnv (harness Game._env). Returns (W, H, state)."""
    W, H = env.width, env.height
    cells = []
    for k in range(W * H):
        stack = tuple(desc_of(e) for e in env.grid.grid[k] if e is not None)
        cells.append(stack)
    return W, H, tuple(cells)


# ── rule extraction (mirrors baba/rule.py extract_ruleset) ────────────────────

def extract_rules(cells, W, H):
    prop = {}
    replace = []
    n = W * H
    for k in range(n):
        c = cells[k]
        if not c or c[-1] != ('RI',):
            continue
        i = k % W
        j = k // W
        # horizontal:  left IS right
        if i - 1 >= 0 and i + 1 < W:
            lc = cells[k - 1]
            rc = cells[k + 1]
            lt = lc[-1] if lc else None
            rt = rc[-1] if rc else None
            if lt is not None and rt is not None and lt[0] == 'RO':
                if rt[0] == 'RP':
                    prop.setdefault(rt[1], set()).add(lt[1])
                elif rt[0] == 'RO':
                    replace.append((lt[1], rt[1]))
        # vertical:  up IS down
        if j - 1 >= 0 and j + 1 < H:
            uc = cells[k - W]
            dc = cells[k + W]
            ut = uc[-1] if uc else None
            dt = dc[-1] if dc else None
            if ut is not None and dt is not None and ut[0] == 'RO':
                if dt[0] == 'RP':
                    prop.setdefault(dt[1], set()).add(ut[1])
                elif dt[0] == 'RO':
                    replace.append((ut[1], dt[1]))

    for unsupported in ('is_move', 'is_pull', 'is_open', 'is_shut'):
        if prop.get(unsupported):
            raise ModelUnsupported("rule property %s" % unsupported)

    you = frozenset(prop.get('is_agent', EMPTY))
    pull = frozenset(prop.get('is_pull', EMPTY))
    stop = frozenset(prop.get('is_stop', EMPTY)) | you | pull
    goal = frozenset(prop.get('is_goal', EMPTY))
    defeat = frozenset(prop.get('is_defeat', EMPTY))
    push = frozenset(prop.get('is_push', EMPTY))
    return Rules(you, stop, goal, defeat, push, tuple(replace))


# ── exact step ────────────────────────────────────────────────────────────────

def step(state, rules, action, step_count, W, H):
    """Exact model of BabaIsYouEnv.step for a directional action.

    Returns (new_state, new_rules, new_step_count, done, win).
    """
    di, dj = DIRS[action]
    cells = list(state)
    you = rules.you
    stop = rules.stop
    goal = rules.goal
    defeat = rules.defeat
    push = rules.push

    win = False
    lose = False

    def top(k):
        c = cells[k]
        return c[-1] if c else None

    def is_push_d(d):
        t0 = d[0]
        if t0 == 'O':
            return d[1] in push
        return t0 in ('RO', 'RI', 'RP')

    def can_overlap_d(d):
        return d[0] == 'O' and d[1] not in stop

    def is_goal_d(d):
        return d[0] == 'O' and d[1] in goal

    def is_defeat_d(d):
        return d[0] == 'O' and d[1] in defeat

    def move_(k):
        i = k % W
        j = k // W
        fi = i + di
        fj = j + dj
        if not (0 <= fi < W and 0 <= fj < H):
            # blocked at grid edge (unreachable in practice: border walls)
            nt = top(k)
            return k, (nt is not None and is_goal_d(nt)), \
                (nt is not None and is_defeat_d(nt))
        fk = fj * W + fi
        ft = top(fk)
        if ft is not None and is_push_d(ft):
            move_(fk)
        ft = top(fk)
        nk = fk if (ft is None or can_overlap_d(ft)) else k
        nt = top(nk)
        w = nt is not None and is_goal_d(nt)
        l = nt is not None and is_defeat_d(nt)
        if nk != k:
            c = cells[k]
            cells[k] = c[:-1]
            cells[nk] = cells[nk] + (c[-1],)
        return nk, w, l

    # agents move in row-major scan order; grid mutates live; has_moved
    # emulated with a set of destination indices (agents are STOP, so two
    # agents can never stack -> index marking is exact).
    if you:
        moved = set()
        for k in range(W * H):
            t = top(k)
            if (t is not None and t[0] == 'O' and t[1] in you
                    and k not in moved):
                nk, w, l = move_(k)
                win, lose = w, l          # overwritten per agent (env bug)
                moved.add(nk)

    nrules = extract_rules(cells, W, H)

    # replace rules: append default-colored obj2 on top of cells topped by obj1
    for (o1, o2) in nrules.replace:
        newd = ('O', o2, DEFAULT_COLOR[o2])
        for k in range(W * H):
            c = cells[k]
            if c and c[-1][0] == 'O' and c[-1][1] == o1:
                cells[k] = c + (newd,)

    sc = step_count + 1
    done = win or lose or sc >= MAX_STEPS
    return tuple(cells), nrules, sc, done, win


# ── helpers ───────────────────────────────────────────────────────────────────

def agent_cells(state, rules):
    """Indices of cells whose TOP object is an agent (matches env scan)."""
    you = rules.you
    if not you:
        return []
    out = []
    for k, c in enumerate(state):
        if c and c[-1][0] == 'O' and c[-1][1] in you:
            out.append(k)
    return out


def render(state, W, H, rules=None):
    """Debug pretty-printer."""
    sym = []
    for j in range(H):
        row = []
        for i in range(W):
            c = state[j * W + i]
            if not c:
                row.append('.'.ljust(9))
                continue
            t = c[-1]
            if t[0] == 'W':
                s = '#'
            elif t[0] == 'O':
                s = t[1].lstrip('f')[:4] + ('*' if len(c) > 1 else '')
            elif t[0] == 'RO':
                s = '[' + t[1].lstrip('f')[:4] + ']'
            elif t[0] == 'RI':
                s = '[is]'
            else:
                s = '[' + t[1].replace('is_', '')[:4] + ']'
            row.append(s.ljust(9))
        sym.append(''.join(row))
    out = '\n'.join(sym)
    if rules is not None:
        out += '\nrules: you=%s stop=%s goal=%s' % (
            sorted(rules.you), sorted(rules.stop), sorted(rules.goal))
    return out
