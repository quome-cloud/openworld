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


def explore(env, perceive, frontier_path, sim, wm, actions_of, budget, targets_by_key=None):
    """Bounded BFS from the frontier: from each reached state try EVERY candidate action once
    (learning the full local transition fan-out, so the model branches and plan_in_model can search),
    expanding into newly-discovered states. Records per-state click targets into `targets_by_key` so
    `actions_of` uses each state's OWN targets, not a stale frontier snapshot. Real env touched only
    here; we replay-to-state (deterministic) to backtrack between siblings.

    Args:
        env, perceive, frontier_path : real env + frame->Stereotype + reset-relative path to the frontier.
        sim            : WorldSim — learns discovered transitions.
        wm             : WorldModel — available for rule learning (optional).
        actions_of     : state_key -> candidate actions (reads targets_by_key for clicks).
        budget         : max real-env action attempts.
        targets_by_key : dict state_key -> click targets, populated as states are perceived.
    """
    from experiments.e131.lookahead import _replay_to, _act

    if not _replay_to(env, frontier_path):
        return  # frontier unreachable
    s = perceive(env.frame)
    cur_key = s.key
    cur_levels = getattr(env, "levels", 0)
    if targets_by_key is not None:
        targets_by_key.setdefault(cur_key, list(getattr(s, "click_targets", [])))

    # FORWARD-ONLY walk: take the first UNKNOWN action and move on -- NO per-action reset()+replay.
    # Real-env backtracking is unreliable on multi-level games (the worker corrupts after some steps),
    # so a replay-per-sibling tree search collapses. A forward walk is cheap and reliable; it learns a
    # chain (thin, not a fan-out). Branchy discovery requires a GENERALIZING synthesized model (the EWM
    # agent), not real-env backtracking -- the documented binding constraint.
    for _ in range(budget):
        acts = actions_of(cur_key)
        unknown = [a for a in acts if not sim.known(cur_key, a)]
        if not unknown:
            break                                   # nothing new reachable from here on this walk
        a = unknown[0]
        alive = _act(env, a)
        nl = getattr(env, "levels", cur_levels)
        if not alive:
            if nl > cur_levels:                     # WIN-CAPTURE: final-level win-edge (frame may be empty)
                sim.learn(cur_key, a, ("__win__", nl), nl)
            break                                   # dead-end/terminal ends this forward walk
        s2 = perceive(env.frame)
        nk = s2.key
        sim.learn(cur_key, a, nk, nl)
        if targets_by_key is not None:
            targets_by_key.setdefault(nk, list(getattr(s2, "click_targets", [])))
        cur_key = nk
        cur_levels = nl


def verify(env, perceive, frontier_path, plan):
    """Replay frontier_path then execute plan on the real env.

    Args:
        env           : real env.
        perceive      : frame -> Stereotype (only used to read levels via env attribute).
        frontier_path : action list to replay from reset() before executing plan.
        plan          : list of actions to execute after the frontier.

    Returns:
        (real_levels, ok, n_executed)
        real_levels — env.levels after the executed prefix (level is read AFTER each step, so a
                      final-level WIN that sets done is still measured).
        ok          — True if the full plan executed without a dead-end.
        n_executed  — number of plan steps actually executed (the verified prefix length); commit
                      only this prefix so best_actions always replays to real_levels.
    """
    from experiments.e131.lookahead import _replay_to, _act

    if not _replay_to(env, frontier_path):
        return 0, False, 0

    levels = getattr(env, 'levels', 0)
    ok = True
    n = 0
    for a in plan:
        alive = _act(env, a)
        levels = getattr(env, 'levels', levels)   # read AFTER the step: captures a done-setting win
        n += 1
        if not alive:
            ok = False
            break
    return levels, ok, n


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
    # Per-STATE click targets (the data fix): each state has its own valid targets, recorded as states
    # are perceived during explore/refine/commit. A stale frontier snapshot would mis-explore click games.
    targets_by_key = {frontier_key: list(s0.click_targets) if 6 in avail else []}

    def actions_of(state_key):
        """Candidate actions at THIS state: directional ids + the state's own recorded click targets."""
        acts = [[a] for a in avail if a != 6]
        acts += [[6, t["x"], t["y"]] for t in targets_by_key.get(state_key, [])]
        return acts

    rounds_done = 0
    for r in range(rounds):
        rounds_done = r + 1

        # 1. Explore — fill sim from real env (records per-state targets into targets_by_key)
        explore(env, perceive, frontier_path, sim, wm, actions_of, explore_budget, targets_by_key)

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

        # 3. Verify plan on real env (captures a done-setting final-level win; returns the verified prefix)
        real_levels, ok, n_exec = verify(env, perceive, frontier_path, plan)
        verified_plan = plan[:n_exec]                  # only what actually executed
        real_steps += n_exec

        # 4. Refine — re-learn the verified real transitions (corrects model mismatches); win-edges too
        if _replay_to(env, frontier_path):
            ck = frontier_key
            cl = frontier_levels
            for a in verified_plan:
                alive = _act(env, a)
                nl = getattr(env, 'levels', cl)
                if not alive:
                    if nl > cl:
                        sim.learn(ck, a, ("__win__", nl), nl)   # final-level win-edge
                    break
                nk = perceive(env.frame).key
                sim.learn(ck, a, nk, nl)
                targets_by_key.setdefault(nk, list(perceive(env.frame).click_targets) if 6 in avail else [])
                ck = nk
                cl = nl

        # 5. Commit if improved; never regress. Extend the frontier by the VERIFIED PREFIX only, so
        #    best_actions always replays to real_levels (no unverified trailing actions).
        if real_levels > best_levels:
            best_levels = real_levels
            frontier_path = frontier_path + verified_plan
            best_actions = list(frontier_path)

            # Update frontier key/levels for the next round (replayable since we committed the prefix)
            if _replay_to(env, frontier_path):
                s_f = perceive(env.frame)
                frontier_key = s_f.key
                frontier_levels = getattr(env, 'levels', real_levels)
                targets_by_key.setdefault(frontier_key, list(s_f.click_targets) if 6 in avail else [])

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
