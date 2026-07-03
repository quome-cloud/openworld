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
    """Step env with action a ([act] or [6, x, y]). Returns True if the step succeeded and the env is
    not done. The real env can raise (e.g. an empty frame on a level-reload/terminal) -- that is caught
    and treated as a dead-end (return False), so a search branch never crashes the whole run."""
    try:
        if a[0] == 6:
            env.step(6, a[1], a[2])
        else:
            env.step(a[0])
    except Exception:
        return False
    return not getattr(env, 'done', False)


def _replay_to(env, path):
    """Reset env and replay every action in path. Returns True if the full path replayed cleanly; False
    if reset or any step failed/terminated (the target state is then unreachable -- caller skips it)."""
    try:
        env.reset()
    except Exception:
        return False
    for a in path:
        if not _act(env, a):
            return False
    return True


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
                if not _replay_to(env, frontier_path + suffix):
                    continue   # this beam node is unreachable (replay desynced/terminated) → skip it
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
                    if not replayed and not _replay_to(env, frontier_path + suffix):
                        replayed = False
                        continue   # node unreachable → skip this candidate
                    alive = _act(env, a_list)
                    replayed = False  # env has moved (or attempted to)
                    if not alive:
                        continue     # dead-end / terminal / empty-frame crash: don't cache or expand
                    s = perceive(env.frame)
                    next_key = s.key
                    next_levels = getattr(env, 'levels', frontier_levels)
                    path_to_next = frontier_path + suffix + [a_list]
                    cache.put(key, a_key, next_key, next_levels, path_to_next)

                new_suffix = suffix + [a_list]
                val = value(frontier_levels, next_levels, next_key, cache.seen)

                if best_val is None or val > best_val:
                    best_val = val
                    best_first = new_suffix[0]

                # done-guard: a freshly-discovered TERMINAL state (game over / level reload) is scored
                # above but NOT promoted into the beam -- re-replaying through done yields unreliable
                # frames. Cache hits are assumed non-terminal (the cache only stores reached states).
                if cached is not None or alive:
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


# ---------------------------------------------------------------------------
# Result dataclass + receding-horizon driver
# ---------------------------------------------------------------------------

class Result:
    """Return value of solve_lookahead."""
    def __init__(self, best_levels, best_actions, cycles, real_steps, cache_size):
        self.best_levels = best_levels
        self.best_actions = best_actions
        self.cycles = cycles
        self.real_steps = real_steps
        self.cache_size = cache_size

    def __repr__(self):
        return (f"Result(best_levels={self.best_levels}, actions={len(self.best_actions)}, "
                f"cycles={self.cycles}, real_steps={self.real_steps}, "
                f"cache_size={self.cache_size})")


def solve_lookahead(env, perceive, seed_actions, win, depth=3, beam=4, budget=4000):
    """Receding-horizon lookahead driver.

    1. Replay seed_actions from reset().
    2. Observe the post-seed state; set best_levels/best_actions baseline (NEVER regress).
    3. Loop:
       - Call best_sequence(...) from the current frontier state.
       - Commit the first action via _act.
       - Append to running action list; update frontier; mark seen.
       - Bank (best_levels / best_actions) whenever levels rise.
       - Stop on: win reached, budget exhausted, env.done, or K=20 consecutive
         cycles with no new state AND no level gain.
    4. Return Result(best_levels, best_actions, cycles, real_steps, cache_size).
    """
    K = 20  # stagnation limit

    # ---- seed phase ----
    env.reset()
    for a in seed_actions:
        _act(env, a)

    # Post-seed observation
    s = perceive(env.frame)
    frontier_key = s.key
    frontier_levels = getattr(env, 'levels', 0)
    frontier_path = [list(a) for a in seed_actions]

    # Initialise cache with seed state as known
    cache = FrontierCache()
    cache.seen.add(frontier_key)
    cache.path_to[frontier_key] = list(frontier_path)

    # Best so far (never regress from seed)
    best_levels = frontier_levels
    best_actions = list(frontier_path)

    actions = list(frontier_path)   # running committed action sequence
    real_steps = 0
    stagnant_cycles = 0
    prev_seen_count = len(cache.seen)

    for cycle in range(budget):
        avail = list(getattr(env, 'avail', []))
        if not avail:
            break

        # Lookahead from current frontier
        first_action, _ = best_sequence(
            env, perceive,
            frontier_path, frontier_key, frontier_levels,
            avail, cache,
            depth=depth, beam=beam,
        )

        if first_action is None:
            break

        # Commit the first action; env must be at frontier_path after best_sequence
        # (best_sequence may have left the env in an unknown state → replay to frontier)
        _replay_to(env, frontier_path)
        _act(env, first_action)
        real_steps += 1

        # Observe new state
        s = perceive(env.frame)
        next_key = s.key
        next_levels = getattr(env, 'levels', frontier_levels)

        # Update cache with this transition
        new_path = frontier_path + [first_action]
        cache.put(frontier_key, tuple(first_action), next_key, next_levels, new_path)

        # Advance frontier
        actions.append(first_action)
        frontier_path = new_path
        frontier_key = next_key
        frontier_levels = next_levels

        # Bank on level rise
        if next_levels > best_levels:
            best_levels = next_levels
            best_actions = list(actions)

        # Win check
        if best_levels >= win and win > 0:
            break
        if getattr(env, 'done', False):
            break
        if best_levels >= win and win > 0:
            break

        # Stagnation: no new state AND no level gain
        new_seen_count = len(cache.seen)
        gained_state = new_seen_count > prev_seen_count
        gained_level = next_levels > (best_levels if actions == best_actions else best_levels)
        # simpler: track whether frontier_levels increased vs last cycle
        if not gained_state:
            stagnant_cycles += 1
        else:
            stagnant_cycles = 0
        prev_seen_count = new_seen_count

        if stagnant_cycles >= K:
            break

    return Result(
        best_levels=best_levels,
        best_actions=best_actions,
        cycles=cycle + 1,
        real_steps=real_steps,
        cache_size=len(cache.trans),
    )
