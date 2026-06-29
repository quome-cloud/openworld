"""plan_in_model: deep lookahead inside the pure WorldSim (no real env).

Because planning is pure (only sim.predict — a dict lookup), backtracking is
free and depth can be large (8+) at negligible cost.  This is the fix for
E131's short-horizon ceiling: depth=8 finds a win that depth=2 cannot see.

The function returns the FULL action path to the best leaf, not just the first
action, because in a pure model we can commit the whole verified sub-plan.

Task 3 additions:  Result, explore, verify, solve_hybrid.
  - explore: bounded real-env walk → fills WorldSim table.
  - verify:  replay frontier + execute plan on real env → (levels, ok).
  - solve_hybrid: EWM loop (explore → plan → verify → refine → commit).
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


# ---------------------------------------------------------------------------
# Task 3: Result, explore, verify, solve_hybrid
# ---------------------------------------------------------------------------

class Result:
    """Return value of solve_hybrid."""
    def __init__(self, best_levels, best_actions, rounds, model_size, real_steps):
        self.best_levels = best_levels
        self.best_actions = best_actions
        self.rounds = rounds
        self.model_size = model_size
        self.real_steps = real_steps

    def __repr__(self):
        return (
            f"Result(best_levels={self.best_levels}, actions={len(self.best_actions)}, "
            f"rounds={self.rounds}, model_size={self.model_size}, "
            f"real_steps={self.real_steps})"
        )


def explore(env, perceive, frontier_path, sim, wm, actions_of, budget):
    """Bounded real-env walk from the frontier; sim.learn every (s,a,s') observed.

    Greedily takes the first *unknown* action at each state.  When all transitions
    from the current state are known the walk stops and restarts from the frontier
    (probing a different branch).  Stops when `budget` real steps have been used
    or no new transitions can be discovered from the frontier.

    Args:
        env           : real env (reset/step interface).
        perceive      : frame -> Stereotype (.key, .click_targets).
        frontier_path : action list from reset() to the frontier state.
        sim           : WorldSim — learns discovered transitions.
        wm            : WorldModel — available for rule learning (optional).
        actions_of    : state_key -> list of candidate actions.
        budget        : max real env steps.
    """
    from experiments.e131.lookahead import _replay_to, _act

    steps = 0
    while steps < budget:
        if not _replay_to(env, frontier_path):
            break  # Frontier is unreachable (reset pollution or terminal)

        s = perceive(env.frame)
        cur_key = s.key
        cur_levels = getattr(env, 'levels', 0)
        walked = False

        # Single walk: move forward as long as there are unknown transitions
        for _ in range(min(budget - steps, 200)):
            acts = actions_of(cur_key)
            unknown = [a for a in acts if not sim.known(cur_key, a)]
            if not unknown:
                break  # All transitions from this state are known; reset & retry

            a = unknown[0]
            if not _act(env, a):
                break  # Step failed or env done

            steps += 1
            walked = True

            s2 = perceive(env.frame)
            nk = s2.key
            nl = getattr(env, 'levels', cur_levels)
            sim.learn(cur_key, a, nk, nl)
            cur_key = nk
            cur_levels = nl

            if getattr(env, 'done', False):
                break

        if not walked:
            break  # No new transitions reachable; exploration exhausted


def verify(env, perceive, frontier_path, plan):
    """Replay frontier_path then execute plan on the real env.

    Args:
        env           : real env.
        perceive      : frame -> Stereotype (only used to read levels via env attribute).
        frontier_path : action list to replay from reset() before executing plan.
        plan          : list of actions to execute after the frontier.

    Returns:
        (real_levels, ok)
        real_levels — env.levels after executing as many plan steps as succeeded.
        ok          — True if no step in plan raised or returned done=False (i.e.
                      the full plan executed without a dead-end; False = truncated).
    """
    from experiments.e131.lookahead import _replay_to, _act

    if not _replay_to(env, frontier_path):
        return 0, False

    levels = getattr(env, 'levels', 0)
    ok = True
    for a in plan:
        if not _act(env, a):
            ok = False
            break
        levels = getattr(env, 'levels', levels)
    return levels, ok


def solve_hybrid(env, perceive, wm, frontier_path, seed_levels, win,
                 depth=8, beam=8, rounds=6, explore_budget=400):
    """EWM loop: explore → plan_in_model → verify → refine → commit.

    Each round:
      1. explore  — bounded real-env walk; fills WorldSim.
      2. plan     — deep beam search inside the pure model; no real env.
      3. verify   — execute the model plan on the real env.
      4. refine   — sim.learn the verified real transitions (corrects mismatches).
      5. commit   — if real levels rose, extend frontier_path and bank best.

    Stops when win is reached, `rounds` rounds complete, or K consecutive rounds
    produce no improvement.  Never regresses below seed_levels.

    Args:
        env           : real game env.
        perceive      : frame -> Stereotype (.key, .click_targets).
        wm            : WorldModel (for optional object-relative rules).
        frontier_path : action list from reset() to the seed frontier.
        seed_levels   : level count at the seed (baseline; never regress).
        win           : target level count (stop when best_levels >= win).
        depth         : lookahead depth in the pure model (default 8).
        beam          : beam width (default 8).
        rounds        : max EWM rounds (default 6).
        explore_budget: max real env steps per explore phase (default 400).

    Returns:
        Result(best_levels, best_actions, rounds, model_size, real_steps).
    """
    from experiments.e132.worldsim import WorldSim
    from experiments.e131.lookahead import _replay_to, _act

    sim = WorldSim()
    frontier_path = list(frontier_path)   # own mutable copy

    # ---- Initialise frontier ----
    if not _replay_to(env, frontier_path):
        return Result(seed_levels, list(frontier_path), 0, 0, 0)

    s0 = perceive(env.frame)
    frontier_key = s0.key
    frontier_levels = getattr(env, 'levels', seed_levels)
    best_levels = max(seed_levels, frontier_levels)
    best_actions = list(frontier_path)
    real_steps = 0
    no_improve_rounds = 0
    K = max(2, rounds // 2)   # stagnation limit

    avail = list(getattr(env, 'avail', []))
    # For click games: seed click targets from the frontier frame; updated on commit
    click_targets = list(s0.click_targets) if 6 in avail else []

    def actions_of(state_key):
        """Candidate actions: directional ids + current frontier click targets."""
        acts = [[a] for a in avail if a != 6]
        acts += [[6, t["x"], t["y"]] for t in click_targets]
        return acts

    rounds_done = 0
    for r in range(rounds):
        rounds_done = r + 1

        # 1. Explore — fill sim from real env
        explore(env, perceive, frontier_path, sim, wm, actions_of, explore_budget)

        if not sim.trans:
            # No transitions discovered (e.g. all actions no-ops on first try)
            no_improve_rounds += 1
            if no_improve_rounds >= K:
                break
            continue

        # 2. Plan in model (pure — no real env)
        plan, val, leaf_key = plan_in_model(
            sim, frontier_key, frontier_levels, actions_of, depth, beam
        )

        if not plan:
            no_improve_rounds += 1
            if no_improve_rounds >= K:
                break
            continue

        # 3. Verify plan on real env
        real_levels, ok = verify(env, perceive, frontier_path, plan)
        real_steps += len(plan)

        # 4. Refine — learn from verified real transitions (corrects model mismatches)
        if _replay_to(env, frontier_path):
            ck = frontier_key
            cl = frontier_levels
            for a in plan:
                if not _act(env, a):
                    break
                s_nxt = perceive(env.frame)
                nk = s_nxt.key
                nl = getattr(env, 'levels', cl)
                sim.learn(ck, a, nk, nl)
                ck = nk
                cl = nl
                real_steps += 1

        # 5. Commit if improved; never regress below seed_levels
        if real_levels > best_levels:
            best_levels = real_levels
            new_fp = frontier_path + plan
            frontier_path = new_fp
            best_actions = list(frontier_path)

            # Update frontier key/levels for the next round
            if _replay_to(env, frontier_path):
                s_f = perceive(env.frame)
                frontier_key = s_f.key
                frontier_levels = real_levels
                if 6 in avail:
                    click_targets[:] = list(s_f.click_targets)

            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        # Win check
        if best_levels >= win and win > 0:
            break

        if no_improve_rounds >= K:
            break

    return Result(
        best_levels=best_levels,
        best_actions=best_actions,
        rounds=rounds_done,
        model_size=len(sim.trans),
        real_steps=real_steps,
    )
