"""The BSTC behavioral cycle (Thm 4.6), made explicit: extrospect -> introspect+filter -> behave
-> measure tension (sim-vs-real) -> retrosynthesize (learn by acting) -> bank on level gain.
Tension here is over object-state KEYS (0 if the model predicted the observed key, 1 if not) --
the discrete analogue of ||sigma_I - sigma_E||; updates drive it to 0 (learning by action)."""
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
