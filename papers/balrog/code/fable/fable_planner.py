"""Fable planner: goal-regression-guided macro search on the symbolic model.

Three phases, cheapest first (preserving the two-phase spirit of the original
solver, plus a safety net):

  Phase 1  primitive BFS on the symbolic model (optimal; covers everything the
           old env-clone BFS covered, ~250x faster).
  Phase 2  macro weighted-A*: successors are "walk to a push-approach cell
           (BFS reachability) then push once" plus terminal "walk onto a WIN
           cell" moves.  Search depth becomes the number of pushes.
           Heuristic = goal regression over rule assembly: enumerate every
           3-cell line that could host `T IS WIN` for a T with a live
           instance, cost = sum of per-block push lower bounds, with
           frozen-block fixpoint dead-end detection (a block wedged against
           static walls / other frozen blocks on some axis can never change
           that coordinate again -> candidate infeasible; no feasible
           candidate and no rule-rewrite potential -> prune the state).
  Phase 3  primitive weighted-A* with the same heuristic (safety net; also
           handles multi-agent states after YOU-reassignment, which Phase 2
           handles by dropping to primitive successors for those nodes).

All plans are sequences of primitive env actions and are meant to be
replay-verified on an env clone by the caller before execution.
"""

import heapq
import time
from collections import deque

from symbolic_model import (
    ACTIONS, DIRS, MAX_STEPS, ModelUnsupported,
    agent_cells, extract_rules, step,
)

INF = float('inf')

# heuristic tuning
W_ASTAR = 2.0            # weight on h in f = g + W*h
H_CAP = 60               # cap used instead of pruning when rule-rewrite is possible
SLOT_CLEAR_PEN = 3       # slot occupied by another (movable) rule block
UNREACH_PEN = 8          # target in a region the agent can't currently walk to


class Budget:
    def __init__(self, max_nodes, deadline):
        self.max_nodes = max_nodes
        self.deadline = deadline
        self.nodes = 0

    def spend(self):
        self.nodes += 1
        if self.nodes >= self.max_nodes:
            return False
        if self.nodes % 512 == 0 and time.time() > self.deadline:
            return False
        return True


# ── geometry helpers ──────────────────────────────────────────────────────────

def static_wall_cells(state):
    return frozenset(k for k, c in enumerate(state)
                     if any(d[0] == 'W' for d in c))


def passable_for_walk(c, rules):
    """Agent may transit this cell: empty, or overlappable non-goal non-defeat."""
    if not c:
        return True
    t = c[-1]
    return (t[0] == 'O' and t[1] not in rules.stop
            and t[1] not in rules.goal and t[1] not in rules.defeat)


def reachability(state, rules, W, H, start):
    """BFS over walk-passable cells from the agent cell. Returns dist dict."""
    dist = {start: 0}
    dq = deque([start])
    while dq:
        k = dq.popleft()
        d1 = dist[k] + 1
        i = k % W
        j = k // W
        for di, dj in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            fi, fj = i + di, j + dj
            if not (0 <= fi < W and 0 <= fj < H):
                continue
            fk = fj * W + fi
            if fk in dist:
                continue
            if passable_for_walk(state[fk], rules):
                dist[fk] = d1
                dq.append(fk)
    return dist


def walk_actions(state, rules, W, H, start, target):
    """Reconstruct one shortest walk start->target through passable cells."""
    if target == start:
        return []
    dist = {start: 0}
    par = {}
    dq = deque([start])
    while dq:
        k = dq.popleft()
        if k == target:
            break
        d1 = dist[k] + 1
        i = k % W
        j = k // W
        for a in ACTIONS:
            di, dj = DIRS[a]
            fi, fj = i + di, j + dj
            if not (0 <= fi < W and 0 <= fj < H):
                continue
            fk = fj * W + fi
            if fk in dist:
                continue
            if passable_for_walk(state[fk], rules):
                dist[fk] = d1
                par[fk] = (k, a)
                dq.append(fk)
    if target not in dist:
        raise RuntimeError("walk reconstruction failed")
    acts = []
    k = target
    while k != start:
        k, a = par[k]
        acts.append(a)
    acts.reverse()
    return acts


def teleport(state, k_from, k_to):
    """Move the agent (top of k_from) to k_to; equals net effect of walking."""
    if k_from == k_to:
        return state
    cells = list(state)
    mover = cells[k_from][-1]
    cells[k_from] = cells[k_from][:-1]
    cells[k_to] = cells[k_to] + (mover,)
    return tuple(cells)


# ── goal-regression heuristic ─────────────────────────────────────────────────

class Heuristic:
    def __init__(self, W, H, state0):
        self.W = W
        self.H = H
        self.static = static_wall_cells(state0)
        # all 3-cell lines (T, IS, WIN-prop slots) on non-static cells,
        # horizontal (left->right) and vertical (top->bottom), matching the
        # env's rule templates.
        lines = []
        for j in range(H):
            for i in range(W - 2):
                ks = (j * W + i, j * W + i + 1, j * W + i + 2)
                if not any(k in self.static for k in ks):
                    lines.append(ks)
        for i in range(W):
            for j in range(H - 2):
                ks = (j * W + i, (j + 1) * W + i, (j + 2) * W + i)
                if not any(k in self.static for k in ks):
                    lines.append(ks)
        self.lines = lines

    # -- frozen-block fixpoint -------------------------------------------------
    def frozen(self, blocks):
        """blocks: dict k -> desc of all rule blocks.
        Returns (F, xfrozen, yfrozen): F = permanently immovable cells."""
        W = self.W
        F = set(self.static)
        changed = True
        while changed:
            changed = False
            for k in blocks:
                if k in F:
                    continue
                xf = (k - 1) in F or (k + 1) in F
                yf = (k - W) in F or (k + W) in F
                if xf and yf:
                    F.add(k)
                    changed = True
        xfrozen = {k: ((k - 1) in F or (k + 1) in F) for k in blocks}
        yfrozen = {k: ((k - W) in F or (k + W) in F) for k in blocks}
        return F, xfrozen, yfrozen

    def _block_cost(self, k, slot, state, xfrozen, yfrozen, F):
        """Push lower bound to bring block at k to slot."""
        if k == slot:
            return 0
        if slot in F:
            return INF
        W = self.W
        bi, bj = k % W, k // W
        si, sj = slot % W, slot // W
        if bi != si and xfrozen[k]:
            return INF
        if bj != sj and yfrozen[k]:
            return INF
        cost = abs(bi - si) + abs(bj - sj)
        occ = state[slot]
        if occ:
            t = occ[-1]
            if t[0] in ('RO', 'RI', 'RP'):
                cost += SLOT_CLEAR_PEN
        return cost

    def h(self, state, rules, W, H, agent_k, dist):
        """Estimated primitive steps to win. INF => provably-ish dead."""
        # agent type itself is WIN -> bump wins
        acell = state[agent_k]
        atype = acell[-1][1] if acell else None
        if atype is not None and atype in rules.goal:
            return 1

        best = INF

        # option A: an active WIN rule with a visible instance
        if rules.goal:
            for k, c in enumerate(state):
                if c and c[-1][0] == 'O' and c[-1][1] in rules.goal:
                    i, j = k % W, k // W
                    # walk to a neighbor then step on
                    dcell = INF
                    for di, dj in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                        fi, fj = i + di, j + dj
                        if 0 <= fi < W and 0 <= fj < H:
                            nk = fj * W + fi
                            if nk in dist:
                                dcell = min(dcell, dist[nk] + 1)
                    if dcell is INF:
                        ai, aj = agent_k % W, agent_k // W
                        dcell = abs(ai - i) + abs(aj - j) + UNREACH_PEN
                    best = min(best, dcell)

        # option B: assemble `T IS WIN` somewhere
        ro_blocks = {}
        ri_blocks = []
        win_blocks = []
        inst = {}
        for k, c in enumerate(state):
            if not c:
                continue
            t = c[-1]
            if t[0] == 'RO':
                ro_blocks.setdefault(t[1], []).append(k)
            elif t[0] == 'RI':
                ri_blocks.append(k)
            elif t[0] == 'RP':
                if t[1] == 'is_goal':
                    win_blocks.append(k)
            elif t[0] == 'O':
                inst.setdefault(t[1], []).append(k)

        if ri_blocks and win_blocks and ro_blocks:
            all_blocks = {}
            for ks in ro_blocks.values():
                for k in ks:
                    all_blocks[k] = 1
            for k in ri_blocks:
                all_blocks[k] = 1
            for k in win_blocks:
                all_blocks[k] = 1
            F, xf, yf = self.frozen(all_blocks)
            ai, aj = agent_k % W, agent_k // W

            for (sT, sI, sW) in self.lines:
                # cheapest IS and WIN placement for this line
                cI = min(self._block_cost(k, sI, state, xf, yf, F)
                         for k in ri_blocks)
                if cI is INF:
                    continue
                cWn = min(self._block_cost(k, sW, state, xf, yf, F)
                          for k in win_blocks)
                if cWn is INF:
                    continue
                for T, ks in ro_blocks.items():
                    if T not in inst:
                        continue
                    cT = min(self._block_cost(k, sT, state, xf, yf, F)
                             for k in ks)
                    if cT is INF:
                        continue
                    push_cost = cT + cI + cWn
                    # agent engagement: get near some block that must move
                    eng = 0
                    if push_cost > 0:
                        eng = INF
                        for role_ks, slot in ((ks, sT), (ri_blocks, sI),
                                              (win_blocks, sW)):
                            for k in role_ks:
                                if k == slot:
                                    continue
                                if k in dist:
                                    eng = min(eng, max(dist[k] - 1, 0))
                                else:
                                    bi, bj = k % W, k // W
                                    eng = min(eng, abs(ai - bi) + abs(aj - bj)
                                              + UNREACH_PEN)
                        if eng is INF:
                            eng = 0
                    # then reach a T instance from the assembled rule site
                    ti, tj = sT % W, sT // W
                    reach = min(abs(ti - k % W) + abs(tj - k // W)
                                for k in inst[T])
                    best = min(best, push_cost + eng + reach)

        if best is INF:
            # keep alive only if rule-space could still change in ways the
            # regression above cannot see (YOU reassignment, replace rules)
            has_you_block = any(
                c and c[-1] == ('RP', 'is_agent') for c in state)
            n_ro = sum(1 for c in state if c and c[-1][0] == 'RO')
            if (has_you_block or n_ro >= 2) and ri_blocks:
                return H_CAP
            return INF
        return min(best, H_CAP * 4)


# ── plan node bookkeeping ─────────────────────────────────────────────────────

class Node:
    __slots__ = ('state', 'rules', 'g', 'parent', 'seg', '_dist')

    def __init__(self, state, rules, g, parent, seg):
        self.state = state
        self.rules = rules
        self.g = g
        self.parent = parent
        self.seg = seg          # ('acts', [..]) or ('walkpush', target_k, act)
        self._dist = None       # cached reachability map (macro search)

    def extract_plan(self, W, H):
        chain = []
        n = self
        while n is not None:
            chain.append(n)
            n = n.parent
        chain.reverse()
        plan = []
        for prev, cur in zip(chain, chain[1:]):
            kind = cur.seg[0]
            if kind == 'acts':
                plan.extend(cur.seg[1])
            else:
                _, target_k, act = cur.seg
                ags = agent_cells(prev.state, prev.rules)
                start = ags[0]
                plan.extend(walk_actions(prev.state, prev.rules, W, H,
                                         start, target_k))
                plan.append(act)
        return plan


# ── phase 1: primitive BFS ───────────────────────────────────────────────────

def bfs_primitive(state0, rules0, W, H, budget):
    root = Node(state0, rules0, 0, None, None)
    dq = deque([root])
    visited = {state0}
    while dq:
        if not budget.spend():
            return None
        node = dq.popleft()
        if node.g >= MAX_STEPS:
            continue
        for a in ACTIONS:
            try:
                ns, nr, _, done, win = step(node.state, node.rules, a,
                                            node.g, W, H)
            except ModelUnsupported:
                continue
            child = Node(ns, nr, node.g + 1, node, ('acts', [a]))
            if win:
                return child
            if done:
                continue
            if ns not in visited:
                visited.add(ns)
                dq.append(child)
    return None


# ── phase 2/3: weighted A* (macro or primitive successors) ───────────────────

def successors_macro(node, W, H):
    """Macro successors; falls back to primitive for exotic states."""
    state, rules = node.state, node.rules
    ags = agent_cells(state, rules)
    if len(ags) != 1 or rules.replace:
        yield from successors_primitive(node, W, H)
        return
    k0 = ags[0]
    dist = node._dist
    if dist is None:
        dist = reachability(state, rules, W, H, k0)
        node._dist = dist

    # terminal: walk onto an adjacent WIN cell / bump when agent-is-win
    seen_targets = set()
    if rules.goal:
        atype = state[k0][-1][1]
        if atype in rules.goal:
            for a in ACTIONS:
                ns, nr, _, done, win = step(state, rules, a, node.g, W, H)
                if win and node.g + 1 <= MAX_STEPS:
                    yield Node(ns, nr, node.g + 1, node, ('acts', [a])), True
                    return
            # walk to the nearest cell adjacent to a static wall, then bump
            static = static_wall_cells(state)
            best = None
            for nk, d in dist.items():
                if best is not None and d >= best[0]:
                    continue
                i, j = nk % W, nk // W
                for a in ACTIONS:
                    di, dj = DIRS[a]
                    fi, fj = i + di, j + dj
                    if (0 <= fi < W and 0 <= fj < H
                            and fj * W + fi in static):
                        best = (d, nk, a)
                        break
            if best is not None and node.g + best[0] + 1 <= MAX_STEPS:
                d, nk, a = best
                ws = teleport(state, k0, nk)
                ns, nr, _, done, win = step(ws, rules, a, node.g + d, W, H)
                if win:
                    yield (Node(ns, nr, node.g + d + 1, node,
                                ('walkpush', nk, a)), True)
                    return
        for k, c in enumerate(state):
            if c and c[-1][0] == 'O' and c[-1][1] in rules.goal:
                i, j = k % W, k // W
                for a in ACTIONS:
                    di, dj = DIRS[a]
                    ni, nj = i - di, j - dj      # approach cell
                    if not (0 <= ni < W and 0 <= nj < H):
                        continue
                    nk = nj * W + ni
                    if nk not in dist:
                        continue
                    g2 = node.g + dist[nk] + 1
                    if g2 > MAX_STEPS:
                        continue
                    ws = teleport(state, k0, nk)
                    wr = rules
                    ns, nr, _, done, win = step(ws, wr, a, node.g + dist[nk],
                                                W, H)
                    if win:
                        yield (Node(ns, nr, g2, node, ('walkpush', nk, a)),
                               True)

    # pushes: for every pushable block and direction with reachable approach
    for k, c in enumerate(state):
        if not c:
            continue
        t = c[-1]
        pushable = t[0] in ('RO', 'RI', 'RP') or (
            t[0] == 'O' and t[1] in rules.push)
        if not pushable:
            continue
        i, j = k % W, k // W
        for a in ACTIONS:
            di, dj = DIRS[a]
            ai, aj = i - di, j - dj
            if not (0 <= ai < W and 0 <= aj < H):
                continue
            ak = aj * W + ai
            if ak not in dist:
                continue
            key = (ak, a)
            if key in seen_targets:
                continue
            seen_targets.add(key)
            gw = node.g + dist[ak]
            if gw + 1 > MAX_STEPS:
                continue
            ws = teleport(state, k0, ak)
            try:
                ns, nr, _, done, win = step(ws, rules, a, gw, W, H)
            except ModelUnsupported:
                continue
            if win:
                yield Node(ns, nr, gw + 1, node, ('walkpush', ak, a)), True
                continue
            if done:
                continue
            if ns == ws:
                continue                      # push jammed, nothing happened
            yield Node(ns, nr, gw + 1, node, ('walkpush', ak, a)), False


def successors_primitive(node, W, H):
    for a in ACTIONS:
        try:
            ns, nr, _, done, win = step(node.state, node.rules, a,
                                        node.g, W, H)
        except ModelUnsupported:
            continue
        if node.g + 1 > MAX_STEPS:
            continue
        child = Node(ns, nr, node.g + 1, node, ('acts', [a]))
        if win:
            yield child, True
        elif not done:
            yield child, False


def wastar(state0, rules0, W, H, budget, heur, use_macro):
    root = Node(state0, rules0, 0, None, None)
    counter = 0
    ags = agent_cells(state0, rules0)
    if not ags:
        return None
    dist0 = reachability(state0, rules0, W, H, ags[0])
    h0 = heur.h(state0, rules0, W, H, ags[0], dist0)
    if h0 is INF:
        return None
    root._dist = dist0
    pq = [(W_ASTAR * h0, 0, root)]
    best_g = {state0: 0}
    succ_fn = successors_macro if use_macro else successors_primitive
    while pq:
        if not budget.spend():
            return None
        f, _, node = heapq.heappop(pq)
        if best_g.get(node.state, INF) < node.g:
            continue
        for child, win in succ_fn(node, W, H):
            if win:
                return child
            if best_g.get(child.state, INF) <= child.g:
                continue
            best_g[child.state] = child.g
            ags = agent_cells(child.state, child.rules)
            if not ags:
                continue                       # no agent left: dead state
            dist = reachability(child.state, child.rules, W, H, ags[0])
            child._dist = dist
            h = heur.h(child.state, child.rules, W, H, ags[0], dist)
            if h is INF:
                continue                       # goal-regression dead end
            counter += 1
            heapq.heappush(pq, (child.g + W_ASTAR * h, counter, child))
    return None


# ── entry point ───────────────────────────────────────────────────────────────

def plan_symbolic(W, H, state0, rules0,
                  bfs_nodes=20000, macro_nodes=60000, prim_nodes=250000,
                  time_budget_s=420.0):
    """Returns (actions, stats) or (None, stats)."""
    t0 = time.time()
    deadline = t0 + time_budget_s
    stats = {}

    b1 = Budget(bfs_nodes, min(deadline, t0 + 60.0))
    node = bfs_primitive(state0, rules0, W, H, b1)
    stats['bfs_nodes'] = b1.nodes
    if node is not None:
        stats['method'] = 'sym_bfs'
        stats['elapsed_plan_s'] = round(time.time() - t0, 2)
        return node.extract_plan(W, H), stats

    heur = Heuristic(W, H, state0)

    b2 = Budget(macro_nodes, deadline)
    node = wastar(state0, rules0, W, H, b2, heur, use_macro=True)
    stats['macro_nodes'] = b2.nodes
    if node is not None:
        stats['method'] = 'sym_macro_wastar'
        stats['elapsed_plan_s'] = round(time.time() - t0, 2)
        return node.extract_plan(W, H), stats

    b3 = Budget(prim_nodes, deadline)
    node = wastar(state0, rules0, W, H, b3, heur, use_macro=False)
    stats['prim_nodes'] = b3.nodes
    stats['elapsed_plan_s'] = round(time.time() - t0, 2)
    if node is not None:
        stats['method'] = 'sym_prim_wastar'
        return node.extract_plan(W, H), stats

    stats['method'] = 'failed'
    return None, stats
