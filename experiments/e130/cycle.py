"""The BSTC behavioral cycle (Thm 4.6), made explicit: extrospect -> introspect+filter -> behave
-> measure tension (sim-vs-real) -> retrosynthesize (learn by acting) -> bank on level gain.
Tension here is over object-state KEYS (0 if the model predicted the observed key, 1 if not) --
the discrete analogue of ||sigma_I - sigma_E||; updates drive it to 0 (learning by action).

v2 (run_cycle_v2): adds receding-horizon multi-step planning via dir_map navigation, adaptive
expert weights (multiplicative bandit), and object-relative lifted dynamics (learn_rule on move).
v1 (run_cycle) is kept intact so all existing tests remain green."""
from dataclasses import dataclass, field
import numpy as np
from experiments.e130 import moral_filter as mf


@dataclass
class Result:
    best_levels: int
    best_actions: list
    tension_trace: list = field(default_factory=list)
    cycles: int = 0
    banked: int = 0


def _do(game, action):
    a = action[0]
    try:
        game.step(a, action[1], action[2]) if a == 6 else game.step(a)
        return not bool(getattr(game, "done", False))
    except Exception:
        return False


def run_cycle(game, world_model, perceive, experts, budget, win, rng, seed_actions=()):
    actions = [list(a) for a in seed_actions]
    for a in actions:                                  # replay the TEIE frontier (O(1), Thm 4.10)
        _do(game, a)
    best = int(getattr(game, "levels", 0))
    best_actions = list(actions)
    trace, steps, cycles = [], 0, 0
    while steps < budget and best < win and not getattr(game, "done", False):
        cycles += 1
        s = perceive(game.frame)
        wp, plan, _ = mf.select(s, best_actions, world_model, experts, rng)
        if not plan:
            break
        for act in plan:
            pred_key, known = world_model.predict(s.key, (wp.kind, wp.y, wp.x))
            if not _do(game, act):
                break
            steps += 1
            obs = perceive(game.frame)
            T = 0.0 if (known and pred_key == obs.key) else 1.0    # sim-vs-real tension
            trace.append(T)
            if T > 0.0:
                world_model.update(s.key, (wp.kind, wp.y, wp.x), obs.key)   # I_gamma: learn by acting
            actions.append([act[0], act[1], act[2]] if act[0] == 6 else [act[0]])
            lv = int(getattr(game, "levels", 0))
            if lv > best:
                best, best_actions = lv, list(actions)
                world_model.bank_subroutine(f"win_to_{lv}", list(actions))
            s = obs
    banked = len(world_model.db)
    return Result(best_levels=best, best_actions=best_actions, tension_trace=trace,
                  cycles=cycles, banked=banked)


def run_cycle_v2(game, world_model, perceive, experts, budget, win, rng,
                 seed_actions=(), avatar=None, dir_map=None):
    """Receding-horizon cycle (v2): selects waypoints with adaptive expert weights, realizes them
    as multi-step directional navigation plans (receding horizon), executes on the real env, and
    updates lifted dynamics (learn_rule) when the avatar moves. The expert that produced the most
    progress (new state keys or level gains) is upweighted via the multiplicative-weights bandit.
    v1 (run_cycle) is kept intact; this function adds the v2 capability additively."""
    from experiments.e130.weights import ExpertWeights
    from experiments.e130 import navigation as nav

    actions = [list(a) for a in seed_actions]
    for a in actions:                                  # replay the frontier (O(1) per Thm 4.10)
        _do(game, a)
    best = int(getattr(game, "levels", 0))
    best_actions = list(actions)
    trace, steps, cycles = [], 0, 0
    seen_keys = set()

    # Build multiplicative-weights bandit over expert function names
    W = ExpertWeights([e.__name__ for e in experts])

    while steps < budget and best < win and not getattr(game, "done", False):
        cycles += 1
        s = perceive(game.frame)
        seen_keys.add(s.key)

        # Select waypoint with adaptive weights; then re-realize with avatar so that
        # "reach" waypoints become multi-step directional plans rather than degenerate clicks.
        # (select() calls realize() internally without avatar, so it always falls back to click;
        # re-realizing here with avatar is the v2 receding-horizon correction.)
        wp, _, _ = mf.select(s, best_actions, world_model, experts, rng,
                              dir_map=dir_map, weights=W)
        plan = mf.realize(wp, s, dir_map=dir_map, avatar=avatar)

        # v2 fallback: when the selected plan is empty (e.g. the waypoint is the avatar's
        # current position) but dir_map is available, navigate toward the nearest non-avatar
        # object (receding-horizon exploration).
        if not plan and dir_map and avatar is not None:
            for o in s.objects:
                if o["color"] != avatar:
                    cand = mf.Waypoint("reach", o["y"], o["x"],
                                       wp.source if (wp and wp.source != "none") else "explore")
                    plan = mf.realize(cand, s, dir_map=dir_map, avatar=avatar)
                    if plan:
                        wp = cand
                        break

        if not plan:
            break

        source = wp.source
        got_progress = False

        # Execute the multi-step plan (the receding horizon: plan computed from current state,
        # then executed action-by-action on the real env)
        for act in plan:
            if steps >= budget:
                break
            pred_key, known = world_model.predict(s.key, (wp.kind, wp.y, wp.x))
            prev_pos = nav.avatar_pos(s, avatar) if avatar is not None else None

            if not _do(game, act):
                break
            steps += 1
            obs = perceive(game.frame)
            T = 0.0 if (known and pred_key == obs.key) else 1.0    # sim-vs-real tension
            trace.append(T)
            if T > 0.0:
                world_model.update(s.key, (wp.kind, wp.y, wp.x), obs.key)

            # Learn lifted (object-relative) rule when the avatar moved
            if avatar is not None and prev_pos is not None:
                new_pos = nav.avatar_pos(obs, avatar)
                if new_pos is not None and new_pos != prev_pos:
                    world_model.learn_rule(avatar, act[0], prev_pos, new_pos)

            # Track action in the format consistent with v1
            actions.append([act[0], act[1], act[2]] if act[0] == 6 else [act[0]])

            lv = int(getattr(game, "levels", 0))
            if lv > best:
                best, best_actions = lv, list(actions)
                world_model.bank_subroutine(f"win_to_{lv}", list(actions))
                got_progress = True

            # New state key = progress signal for the bandit
            if obs.key not in seen_keys:
                seen_keys.add(obs.key)
                got_progress = True

            s = obs

        # Reward the expert that chose this waypoint
        W.reward(source, 1.0 if got_progress else 0.0)

    banked = len(world_model.db)
    return Result(best_levels=best, best_actions=best_actions, tension_trace=trace,
                  cycles=cycles, banked=banked)
