"""Blocksworld as a verified-code OpenWorld world model (E78 substrate).

The canonical PlanBench / "can LLMs plan?" domain (Valmeekam et al., NeurIPS 2023),
authored as a single sandboxed `transition(state, action)` -- the SAME verified dynamics
that (a) the agent calls as a tool, (b) the BFS oracle plans over, and (c) the scorer
replays to judge plan validity. One source of truth, training-free, bit-exact.

State (transient `_ok`/`_msg` report the last action's legality; not part of the schema):
  {"on": {x: y}, "table": [blocks], "clear": [blocks], "holding": block|None}
Actions: pickup(x), putdown(x), stack(x,y), unstack(x,y)  -- standard 4-action STRIPS.

Everything here is deterministic and offline (stdlib + openworld core, no LLM, no GPU).
"""

from __future__ import annotations

from collections import deque

from openworld import Action, World, WorldState
from openworld.transition import CodeTransition

# Single source of truth for the dynamics: illegal actions are no-ops that set _ok=False
# with a human-readable reason (so the agent tool can report *why* a move was rejected).
TRANSITION_CODE = '''
def transition(state, action):
    clear = set(state.get("clear", []))
    on = dict(state.get("on", {}))
    table = set(state.get("table", []))
    holding = state.get("holding")
    name = action.get("name")
    p = action.get("params", {})
    x = p.get("x")
    y = p.get("y")

    def out(ok, msg):
        return {"on": dict(on), "table": sorted(table), "clear": sorted(clear),
                "holding": holding, "_ok": ok, "_msg": msg}

    if name == "pickup":
        if holding is not None: return out(False, "hand is not empty")
        if x not in table: return out(False, str(x) + " is not on the table")
        if x not in clear: return out(False, str(x) + " is not clear")
        table.discard(x); clear.discard(x); holding = x
        return out(True, "")
    if name == "putdown":
        if holding != x: return out(False, "not holding " + str(x))
        table.add(x); clear.add(x); holding = None
        return out(True, "")
    if name == "stack":
        if holding != x: return out(False, "not holding " + str(x))
        if y not in clear: return out(False, str(y) + " is not clear")
        if x == y: return out(False, "cannot stack a block on itself")
        on[x] = y; clear.discard(y); clear.add(x); holding = None
        return out(True, "")
    if name == "unstack":
        if holding is not None: return out(False, "hand is not empty")
        if on.get(x) != y: return out(False, str(x) + " is not on " + str(y))
        if x not in clear: return out(False, str(x) + " is not clear")
        del on[x]; clear.add(y); clear.discard(x); holding = x
        return out(True, "")
    return out(False, "unknown action " + str(name))
'''

ACTIONS = ["pickup", "putdown", "stack", "unstack"]
RULES = [
    "A block can be picked up only if it is clear, on the table, and the hand is empty.",
    "A block can be unstacked only if it is clear, on the named block, and the hand is empty.",
    "A held block can be put down on the table, or stacked on a clear block.",
    "The goal is reached when every required on(x,y) and on-table fact holds.",
]
DESCRIPTION = ("Blocksworld: rearrange labelled blocks by pickup/putdown/stack/unstack with "
               "a one-block gripper, respecting clear/handempty preconditions.")

_TRANSITION = CodeTransition(TRANSITION_CODE)


def step(state, action_name, **params):
    """Apply one action through the verified world. Returns the new state dict
    (with transient _ok/_msg). Pure: does not mutate the input."""
    nxt = _TRANSITION.step(WorldState(dict(state)), Action(action_name, params))
    return dict(nxt)


def make_world(initial_state):
    """The OpenWorld World whose transition is the verified Blocksworld dynamics -- the
    exact object served at /step,/rollout and handed to the agent as a tool in E78."""
    return World(
        name="blocksworld",
        description=DESCRIPTION,
        initial_state=dict(initial_state),
        actions=list(ACTIONS),
        rules=list(RULES),
        transition=CodeTransition(TRANSITION_CODE),
    )


# ---------------------------------------------------------------------------
# Goals, canonical keys, legal-action enumeration
# ---------------------------------------------------------------------------

def goal_satisfied(state, goal):
    """Goal is {"on": {x: y}, "table": [blocks]} -- a partial spec the state must entail."""
    on = state.get("on", {})
    table = set(state.get("table", []))
    for x, y in goal.get("on", {}).items():
        if on.get(x) != y:
            return False
    for x in goal.get("table", []):
        if x not in table:
            return False
    return True


def _key(state):
    """Hashable canonical state (clear is derivable, so it is excluded)."""
    on = state.get("on", {})
    return (frozenset(on.items()), frozenset(state.get("table", [])), state.get("holding"))


def legal_actions(state):
    """Every (name, params) legal in `state`, as the BFS branching set."""
    out = []
    clear = set(state.get("clear", []))
    table = set(state.get("table", []))
    on = state.get("on", {})
    holding = state.get("holding")
    if holding is None:
        for x in clear:
            if x in table:
                out.append(("pickup", {"x": x}))
        for x, y in on.items():
            if x in clear:
                out.append(("unstack", {"x": x, "y": y}))
    else:
        out.append(("putdown", {"x": holding}))
        for y in clear:
            if y != holding:
                out.append(("stack", {"x": holding, "y": y}))
    # Deterministic order (set/dict iteration over strings is hash-salted across processes,
    # which would make BFS tie-breaking and random-walk choices irreproducible).
    out.sort(key=lambda np: (np[0], np[1].get("x", ""), np[1].get("y", "")))
    return out


# ---------------------------------------------------------------------------
# BFS oracle (optimal plan) + plan validation
# ---------------------------------------------------------------------------

def bfs_plan(init, goal, max_nodes=200000):
    """Shortest plan (list of (name, params)) from init to a goal-satisfying state, or None.
    Doubles as the optimal-length grader and the upper-bound oracle arm."""
    if goal_satisfied(init, goal):
        return []
    seen = {_key(init)}
    q = deque([(init, [])])
    nodes = 0
    while q and nodes < max_nodes:
        state, plan = q.popleft()
        for name, params in legal_actions(state):
            nodes += 1
            nxt = step(state, name, **params)
            if not nxt.get("_ok"):
                continue
            k = _key(nxt)
            if k in seen:
                continue
            new_plan = plan + [(name, params)]
            if goal_satisfied(nxt, goal):
                return new_plan
            seen.add(k)
            q.append((nxt, new_plan))
    return None


def validate_plan(init, goal, plan):
    """Replay `plan` (list of (name, params)) through the verified world.
    Returns dict: valid (all legal AND goal reached), reached, n_legal, first_illegal."""
    state = dict(init)
    n_legal = 0
    first_illegal = None
    for i, (name, params) in enumerate(plan):
        nxt = step(state, name, **params)
        if not nxt.get("_ok"):
            first_illegal = {"index": i, "action": [name, params], "reason": nxt.get("_msg")}
            break
        state = nxt
        n_legal += 1
    reached = goal_satisfied(state, goal)
    return {"valid": bool(first_illegal is None and reached),
            "reached": bool(reached), "n_legal": n_legal,
            "n_actions": len(plan), "first_illegal": first_illegal,
            "final_state": {"on": dict(state.get("on", {})),
                            "table": sorted(state.get("table", [])),
                            "holding": state.get("holding")}}


# ---------------------------------------------------------------------------
# Instance generation (graded by optimal plan length)
# ---------------------------------------------------------------------------

def _random_config(blocks, rng):
    """A random legal tower configuration: each block on the table or on a clear block."""
    rng.shuffle(blocks)
    on, table, clear = {}, set(), set(blocks)
    placed = []
    for b in blocks:
        spots = [s for s in placed if s in clear]
        if spots and rng.random() < 0.6:
            base = rng.choice(spots)
            on[b] = base
            clear.discard(base)
        else:
            table.add(b)
        placed.append(b)
    return {"on": on, "table": sorted(table), "clear": sorted(clear), "holding": None}


def gen_problem(n_blocks, target_len, rng, tries=400):
    """A solvable instance whose OPTIMAL plan length == target_len, with init/goal/optimal.
    Returns None if no instance of that exact length is found within `tries`."""
    blocks = [chr(ord("a") + i) for i in range(n_blocks)]
    for _ in range(tries):
        init = _random_config(list(blocks), rng)
        goalcfg = _random_config(list(blocks), rng)
        goal = {"on": dict(goalcfg["on"]),
                "table": [b for b in blocks if b in set(goalcfg["table"])]}
        plan = bfs_plan(init, goal)
        if plan is not None and len(plan) == target_len:
            return {"n_blocks": n_blocks, "init": init, "goal": goal,
                    "optimal_len": target_len,
                    "optimal_plan": [[n, p] for n, p in plan]}
    return None
