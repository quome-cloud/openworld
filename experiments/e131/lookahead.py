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
