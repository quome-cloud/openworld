"""plan_in_model: deep lookahead inside the pure WorldSim (no real env).

Because planning is pure (only sim.predict — a dict lookup), backtracking is
free and depth can be large (8+) at negligible cost.  This is the fix for
E131's short-horizon ceiling: depth=8 finds a win that depth=2 cannot see.

The function returns the FULL action path to the best leaf, not just the first
action, because in a pure model we can commit the whole verified sub-plan.
"""

from experiments.e131.lookahead import value as _value


def plan_in_model(sim, start_key, start_levels, actions_of, depth=8, beam=8):
    """Beam lookahead entirely inside sim (pure predict — no real env).

    Args:
        sim          : WorldSim with .predict(state_key, action) -> (next_key, levels) | None
                       and .seen (set of known state keys).
        start_key    : hashable state key for the planning root.
        start_levels : integer level count at the root (for value computation).
        actions_of   : callable (state_key) -> list of actions ([a] or [6,x,y]).
        depth        : maximum lookahead depth (default 8; free in the model).
        beam         : maximum beam width per depth (default 8).

    Returns:
        (plan, value_tuple, leaf_key)
        plan        — list of actions (each a list) forming the full path from
                      start to the best-scoring node.
        value_tuple — (level_delta, novelty) of the best node.
        leaf_key    — state key of the best node reached.
    """
    # Score the start node.
    best_val = _value(start_levels, start_levels, start_key, sim.seen)
    best_path = []
    best_leaf = start_key

    # visited: state keys already in the beam (prevents self-loop inflation and
    # revisiting a state via a different path that can only be equal-or-worse).
    visited = {start_key}

    # beam_nodes: list of (state_key, levels, path_of_actions)
    beam_nodes = [(start_key, start_levels, [])]

    for _d in range(depth):
        if not beam_nodes:
            break

        candidates = []  # (value_tuple, next_key, next_levels, new_path)

        for node_key, node_levels, path in beam_nodes:
            for action in actions_of(node_key):
                nxt = sim.predict(node_key, action)
                if nxt is None:
                    # Unknown transition — knowledge frontier; the node itself
                    # was already scored when it entered the beam.  Skip.
                    continue

                next_key, next_levels = nxt
                new_path = path + [action]
                val = _value(start_levels, next_levels, next_key, sim.seen)

                # Update global best (strictly greater; first best wins ties).
                if val > best_val:
                    best_val = val
                    best_path = new_path
                    best_leaf = next_key

                # Only enqueue states not yet seen (loop / revisit guard).
                if next_key not in visited:
                    candidates.append((val, next_key, next_levels, new_path))
                    visited.add(next_key)

        if not candidates:
            break

        # Beam pruning: keep top-beam by value (stable sort: equal-value
        # candidates keep insertion order, so earliest-found path wins ties).
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam_nodes = [
            (k, lv, p) for (val, k, lv, p) in candidates[:beam]
        ]

    return best_path, best_val, best_leaf
