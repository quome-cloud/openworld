"""Short-horizon lookahead: the explosion is in the HORIZON, not the branching. Per step there are
only ~4-7 actions, so don't infer a whole-level goal -- exhaustively look 2-3 frames ahead over the
deterministic env, score leaves by (level-delta, novelty), commit the best first action, recede.
The FrontierCache is the fast-forward memory: memoized transitions + a replayable path per state."""


def value(frontier_levels, leaf_levels, leaf_key, seen):
    """Leaf score: level-delta dominates; novelty breaks ties. Compared as a tuple."""
    return (leaf_levels - frontier_levels, 0 if leaf_key in seen else 1)


class FrontierCache:
    def __init__(self):
        self.seen = set()
        self.trans = {}        # (state_key, action) -> (next_key, next_levels)
        self.path_to = {}      # state_key -> action list from reset() that reaches it

    def get(self, key, action):
        return self.trans.get((key, action))

    def put(self, key, action, next_key, next_levels, path_to_next):
        self.trans[(key, action)] = (next_key, next_levels)
        self.seen.add(next_key)
        if next_key not in self.path_to:
            self.path_to[next_key] = [list(a) for a in path_to_next]


# ---------------------------------------------------------------------------
# Env interaction helpers
# ---------------------------------------------------------------------------

def _act(env, a):
    """Step env with action a ([act] or [6, x, y]). Returns True if not done."""
    if a[0] == 6:
        env.step(6, a[1], a[2])
    else:
        env.step(a[0])
    return not getattr(env, 'done', False)


def _replay_to(env, path):
    """Reset env and replay every action in path."""
    env.reset()
    for a in path:
        _act(env, a)


# ---------------------------------------------------------------------------
# Depth-d beam expansion
# ---------------------------------------------------------------------------

def best_sequence(env, perceive, frontier_path, frontier_key, frontier_levels,
                  avail, cache, depth=3, beam=4):
    """Exhaustive depth-d beam search from the frontier state.

    Expands action sequences up to `depth` steps.  For each (node, action):
    * cache hit  → fast-forward (no env interaction)
    * cache miss → _replay_to(frontier_path + suffix) then _act, perceive,
                   cache.put (with path_to_next = frontier_path + suffix + [a])

    Scores every leaf (and every intermediate node) with
    value(frontier_levels, leaf_levels, leaf_key, cache.seen); tracks the
    global best; returns the FIRST action of the best sequence + that value.
    Beam pruning (top-`beam` by value) at each depth avoids greedy lock-in.

    Returns (first_action | None, value_tuple).
    """
    if not avail:
        return None, (0, 0)

    best_val = None
    best_first = None

    # Each beam node: (key, levels, suffix)
    # suffix = list of action-lists taken from the frontier; first elem is
    # the first action that would be committed.
    beam_nodes = [(frontier_key, frontier_levels, [])]

    for _d in range(depth):
        candidates = []   # (value_tuple, next_key, next_levels, new_suffix)

        for key, levels, suffix in beam_nodes:
            # ---- build candidate actions for this node ----
            # Non-click: one candidate per avail action id
            action_candidates = [[a] for a in avail if a != 6]

            # Click: need the frame → replay to node, then perceive targets
            if 6 in avail:
                _replay_to(env, frontier_path + suffix)
                s = perceive(env.frame)
                action_candidates += [[6, t["x"], t["y"]]
                                      for t in s.click_targets]
                replayed = True   # env is now at this node's state
            else:
                replayed = False  # env state is unknown / stale

            # ---- expand each candidate action ----
            for a_list in action_candidates:
                a_key = tuple(a_list)          # hashable for cache
                cached = cache.get(key, a_key)

                if cached is not None:
                    next_key, next_levels = cached
                    # env untouched; replayed flag stays as-is
                else:
                    # Replay to node if env is not already there
                    if not replayed:
                        _replay_to(env, frontier_path + suffix)
                    _act(env, a_list)
                    s = perceive(env.frame)
                    next_key = s.key
                    next_levels = getattr(env, 'levels', frontier_levels)
                    path_to_next = frontier_path + suffix + [a_list]
                    cache.put(key, a_key, next_key, next_levels, path_to_next)
                    replayed = False  # env is now at next_key, not at key

                new_suffix = suffix + [a_list]
                val = value(frontier_levels, next_levels, next_key, cache.seen)

                if best_val is None or val > best_val:
                    best_val = val
                    best_first = new_suffix[0]

                candidates.append((val, next_key, next_levels, new_suffix))

        if not candidates:
            break

        # Beam pruning: keep top-`beam` by value (stable sort preserves order
        # among ties, so the first discovered action wins ties at each depth).
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam_nodes = [(k, lv, suf) for (_, k, lv, suf) in candidates[:beam]]

    if best_first is None:
        return None, (0, 0)
    return best_first, best_val
