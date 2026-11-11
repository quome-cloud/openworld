"""Classical planner over the symbolic BabyAI world model.

Structure (mirrors the Baba arm's discipline):
  phase 1 - mission-directed macro planning: navigation is solved exactly by
            Dijkstra/BFS over (pos, dir) with primitive costs; manipulation
            subgoals (pickup / drop / toggle) are enumerated per instruction
            template. Costs are exact primitive step counts.
  phase 2 - fallback: uniform-cost search over the FULL symbolic state with
            primitive actions and goal = verifier success (exact, complete,
            used only if phase 1 fails to produce a verified plan).

Every candidate plan is replay-verified on a clone of the EpisodeModel before
being accepted: acceptance criterion is the model-predicted verifier success
within the step budget.  LLM-free, pure code.
"""

from __future__ import annotations

import heapq
from symbolic_model import (
    LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE, DIR_TO_VEC,
    SymState, SymGoTo, SymPickup, SymOpen, SymPutNext, SymBefore, SymAfter,
    EpisodeModel,
)


class PlanFail(Exception):
    pass


# --------------------------- navigation substrate ---------------------------

def nav_search(state: SymState, starts, goal_test, allow_open_doors=True):
    """Dijkstra over (pos, dir) nodes. starts: [(pos, dir, cost0)].
    Transitions: LEFT/RIGHT (cost 1), FORWARD into empty/open-door cells
    (cost 1), and 'toggle closed unlocked door then forward' (cost 2, only if
    allow_open_doors). Returns (actions, (pos, dir)) for the cheapest node
    satisfying goal_test(pos, dir), or raises PlanFail.

    NOTE toggling a door changes world state; callers replay plans on the
    model, so this stays exact.
    """
    pq = []
    seen = {}
    parent = {}
    for pos, d, c0 in starts:
        node = (pos, d)
        if seen.get(node, 1 << 30) > c0:
            seen[node] = c0
            parent[node] = (None, None)
            heapq.heappush(pq, (c0, node))
    while pq:
        cost, node = heapq.heappop(pq)
        if cost > seen.get(node, 1 << 30):
            continue
        pos, d = node
        if goal_test(pos, d):
            # reconstruct
            acts = []
            cur = node
            while parent[cur][0] is not None:
                prev, a = parent[cur]
                acts.append(a)
                cur = prev
            acts.reverse()
            flat = [x for a in acts for x in (a if isinstance(a, tuple) else (a,))]
            return flat, node
        succs = []
        succs.append(((pos, (d - 1) % 4), LEFT, 1))
        succs.append(((pos, (d + 1) % 4), RIGHT, 1))
        dx, dy = DIR_TO_VEC[d]
        fwd = (pos[0] + dx, pos[1] + dy)
        fobj = state.obj_at(fwd)
        inb = 0 <= fwd[0] < state.width and 0 <= fwd[1] < state.height
        if inb:
            if fobj is None or (fobj.type == "door" and fobj.is_open):
                succs.append(((fwd, d), FORWARD, 1))
            elif (allow_open_doors and fobj.type == "door"
                  and not fobj.is_open and not fobj.is_locked):
                succs.append(((fwd, d), (TOGGLE, FORWARD), 2))
        for nnode, act, w in succs:
            nc = cost + w
            if nc < seen.get(nnode, 1 << 30):
                seen[nnode] = nc
                parent[nnode] = (node, act)
                heapq.heappush(pq, (nc, nnode))
    raise PlanFail("nav: goal unreachable")


def nav_to_face(state, targets, extra_cost0=None):
    """Plan to stand adjacent to any target position, facing it.
    targets: iterable of positions. Returns action list."""
    tset = set(targets)

    def goal(pos, d):
        dx, dy = DIR_TO_VEC[d]
        return (pos[0] + dx, pos[1] + dy) in tset

    starts = extra_cost0 or [(state.agent_pos, state.agent_dir, 0)]
    acts, _ = nav_search(state, starts, goal)
    return acts


# --------------------------- instruction planning ---------------------------

def plan_instr(model: EpisodeModel, instr) -> list:
    """Plan for one instruction on (a clone of) the current model state.
    Returns primitive action list. Does not mutate model."""
    st = model.state

    if isinstance(instr, SymGoTo):
        poss = [p for p in instr.desc.obj_poss]
        if not poss:
            raise PlanFail("goto: no known target positions")
        acts = nav_to_face(st, poss)
        if not acts:
            # already facing the target: the verifier only fires after an
            # action, and turning would change front_pos. PICKUP keeps the
            # front cell facing and cannot invalidate desc.obj_poss (those
            # are only refreshed on DROP).
            acts = [PICKUP]
        return acts

    if isinstance(instr, SymPickup):
        acts_pre = []
        sim = model.clone()
        if sim.state.carrying is not None:
            # must free hands first: drop on an adjacent empty cell
            acts_pre = plan_drop_anywhere(sim.state)
            for a in acts_pre:
                sim.step(a)
        cand = [sim.state.pos_of(oid) for oid in instr.desc.obj_set]
        cand = [p for p in cand if p is not None]
        if not cand:
            raise PlanFail("pickup: no matching object on grid")
        nav = nav_to_face(sim.state, cand)
        return acts_pre + nav + [PICKUP]

    if isinstance(instr, SymOpen):
        best = None
        for oid in instr.desc.obj_set:
            pos = st.pos_of(oid)
            if pos is None:
                continue
            door = st.objs[oid]
            try:
                plan = plan_open_door(model, oid, pos, door)
            except PlanFail:
                continue
            if best is None or len(plan) < len(best):
                best = plan
        if best is None:
            raise PlanFail("open: no door plan found")
        return best

    if isinstance(instr, SymPutNext):
        return plan_putnext(model, instr)

    if isinstance(instr, (SymBefore, SymAfter)):
        if isinstance(instr, SymBefore):
            first, second = instr.instr_a, instr.instr_b
            first_done = instr.a_done == "success"
        else:
            first, second = instr.instr_b, instr.instr_a
            first_done = instr.b_done == "success"
        sim = model.clone()
        acts1 = []
        if not first_done:
            acts1 = plan_instr(sim, first)
            for a in acts1:
                sim.step(a)
        acts2 = plan_instr(sim, second)
        return acts1 + acts2

    raise PlanFail(f"unsupported instruction {type(instr).__name__}")


def plan_drop_anywhere(state: SymState) -> list:
    """Navigate until facing an empty in-bounds cell and drop."""
    def goal(pos, d):
        dx, dy = DIR_TO_VEC[d]
        f = (pos[0] + dx, pos[1] + dy)
        return (0 <= f[0] < state.width and 0 <= f[1] < state.height
                and state.obj_at(f) is None)
    acts, _ = nav_search(state, [(state.agent_pos, state.agent_dir, 0)], goal)
    return acts + [DROP]


def plan_open_door(model: EpisodeModel, oid, pos, door) -> list:
    """Plan to end with a TOGGLE that leaves door `oid` open."""
    st = model.state
    sim = model.clone()
    acts = []
    if door.is_locked:
        car = st.objs.get(st.carrying) if st.carrying is not None else None
        if not (car is not None and car.type == "key" and car.color == door.color):
            if st.carrying is not None:
                d = plan_drop_anywhere(sim.state)
                acts += d
                for a in d:
                    sim.step(a)
            keys = [p for p, o in sim.state.grid.items()
                    if sim.state.objs[o].type == "key"
                    and sim.state.objs[o].color == door.color]
            if not keys:
                raise PlanFail("open: no matching key on grid")
            nav = nav_to_face(sim.state, keys) + [PICKUP]
            acts += nav
            for a in nav:
                sim.step(a)
        nav = nav_to_face(sim.state, [pos]) + [TOGGLE]
        return acts + nav
    elif door.is_open:
        # toggling an open door closes it; toggle twice
        nav = nav_to_face(sim.state, [pos])
        return nav + [TOGGLE, TOGGLE]
    else:
        # NB nav_to_face may route through this very door only if it toggles
        # it open (leaves it open) - final TOGGLE would close it. Replay
        # verification rejects such plans; disallow door-opening en route to
        # keep it clean.
        def goal(p, d):
            dx, dy = DIR_TO_VEC[d]
            return (p[0] + dx, p[1] + dy) == pos
        nav, _ = nav_search(sim.state, [(sim.state.agent_pos, sim.state.agent_dir, 0)],
                            goal, allow_open_doors=True)
        return nav + [TOGGLE]


def plan_putnext(model: EpisodeModel, instr: SymPutNext) -> list:
    """put desc_move next to desc_fixed: pickup a movable instance, drop on a
    cell Manhattan-adjacent to a fixed instance."""
    best = None
    for oid in instr.desc_move.obj_set:
        sim = model.clone()
        st = sim.state
        acts = []
        if st.carrying is not None and st.carrying != oid:
            d = plan_drop_anywhere(st)
            acts += d
            for a in d:
                sim.step(a)
        if st.carrying != oid:
            pos_a = st.pos_of(oid)
            if pos_a is None:
                continue
            try:
                nav = nav_to_face(st, [pos_a]) + [PICKUP]
            except PlanFail:
                continue
            acts += nav
            for a in nav:
                sim.step(a)
        # fixed positions: current on-grid positions of fixed-set objects
        # (excluding the carried object itself)
        fixed_pos = [st.pos_of(f) for f in instr.desc_fixed.obj_set if f != oid]
        fixed_pos = [p for p in fixed_pos if p is not None]
        if not fixed_pos:
            continue
        drop_cells = set()
        for fp in fixed_pos:
            for dx, dy in DIR_TO_VEC:
                c = (fp[0] + dx, fp[1] + dy)
                if (0 <= c[0] < st.width and 0 <= c[1] < st.height
                        and st.obj_at(c) is None):
                    drop_cells.add(c)
        if not drop_cells:
            continue
        try:
            nav = nav_to_face(st, drop_cells)
        except PlanFail:
            continue
        cand = acts + nav + [DROP]
        if best is None or len(cand) < len(best):
            best = cand
    if best is None:
        raise PlanFail("putnext: no plan found")
    return best


# --------------------------- replay verification ----------------------------

def replay_ok(model: EpisodeModel, plan: list):
    """Replay plan on a clone; return (success, steps_used)."""
    sim = model.clone()
    for k, a in enumerate(plan):
        reward, term, trunc = sim.step(a)
        if term and reward > 0:
            return True, k + 1
        if term or trunc:
            return False, k + 1
    return False, len(plan)


# --------------------------- fallback: full-state UCS -----------------------

def ucs_full(model: EpisodeModel, max_nodes=300_000):
    """Uniform-cost search over full symbolic state + verifier progress.
    Exact and complete within the step budget; used as safety net."""
    start = model.clone()
    # verifier progress key: coarse but sufficient for these instr types
    def vkey(m):
        i = m.instr
        parts = []
        def rec(x):
            for attr in ("a_done", "b_done"):
                if hasattr(x, attr):
                    parts.append(str(getattr(x, attr)))
            for attr in ("preCarrying",):
                if hasattr(x, attr):
                    parts.append(str(getattr(x, attr)))
            for attr in ("instr_a", "instr_b"):
                if hasattr(x, attr):
                    rec(getattr(x, attr))
            for attr in ("desc", "desc_move", "desc_fixed"):
                if hasattr(x, attr):
                    parts.append(str(sorted(getattr(x, attr).obj_poss)))
        rec(i)
        return tuple(parts)

    pq = [(0, 0, start, [])]
    seen = set()
    cnt = 0
    nodes = 0
    while pq:
        cost, _, m, plan = heapq.heappop(pq)
        k = (m.state.key(), vkey(m))
        if k in seen:
            continue
        seen.add(k)
        nodes += 1
        if nodes > max_nodes:
            raise PlanFail("ucs: node budget exceeded")
        for a in (LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE):
            m2 = m.clone()
            reward, term, trunc = m2.step(a)
            if term and reward > 0:
                return plan + [a], nodes
            if term or trunc:
                continue
            cnt += 1
            heapq.heappush(pq, (cost + 1, cnt, m2, plan + [a]))
    raise PlanFail("ucs: exhausted")


def solve(model: EpisodeModel):
    """Full solve: phase-1 macro plan (replay-verified) else phase-2 UCS.
    Returns (plan, method)."""
    try:
        plan = plan_instr(model, model.instr)
        ok, _ = replay_ok(model, plan)
        if ok:
            return plan, "macro"
    except PlanFail:
        pass
    plan, _ = ucs_full(model)
    ok, _ = replay_ok(model, plan)
    if not ok:
        raise PlanFail("ucs plan failed replay (model inconsistency)")
    return plan, "ucs"
