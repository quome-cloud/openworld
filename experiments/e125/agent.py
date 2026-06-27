"""The single-level loop: explore -> synthesize predict() (verifier-gated) -> plan IN SIMULATION -> execute
vs the real env, halting on mismatch -> add the surprising transition + re-synthesize -> repeat. Only verified
plans touch the env. The env decides correctness (a real levels bump)."""
from e125 import explorer, simworld, execute


def solve_level(game_factory, candidates_fn, action_api, game, mask, synth_fn,
                budget_explore=60, budget_plan=20000, rounds=6, traces_dir=None):
    trans = explorer.collect(game_factory, candidates_fn, budget_explore)
    real_actions = budget_explore
    committed = []
    for rnd in range(rounds):
        src, fn = synth_fn(trans, action_api, game, mask, traces_dir=traces_dir)
        if fn is None:
            return {"solved": False, "actions": committed, "rounds_used": rnd, "real_actions": real_actions,
                    "reason": "no verified predict()"}
        init = game_factory(); init.reset()
        for a in committed:
            init.step(*a)
        plan = simworld.plan(fn, init.frame, candidates_fn, budget_plan)
        if plan is None:
            return {"solved": False, "actions": committed, "rounds_used": rnd, "real_actions": real_actions,
                    "reason": "no sim plan"}
        # execute committed+plan against a fresh real game, but verify only the new `plan` segment
        rg = game_factory(); rg.reset()
        for a in committed:
            rg.step(*a)
        res = _exec_from(rg, plan, fn, mask)
        real_actions += len(res["verified_prefix"]) + (1 if res["halt_step"] is not None else 0)
        committed += res["verified_prefix"]
        if res["solved"]:
            return {"solved": True, "actions": committed, "rounds_used": rnd + 1, "real_actions": real_actions}
        if res["new_transitions"]:
            trans = trans + res["new_transitions"]       # add the surprising transition, re-synthesize next round
        else:
            return {"solved": False, "actions": committed, "rounds_used": rnd + 1, "real_actions": real_actions,
                    "reason": "plan exhausted without progress"}
    return {"solved": False, "actions": committed, "rounds_used": rounds, "real_actions": real_actions}


def _exec_from(real_game, plan, predict_fn, mask):
    """execute.execute_plan but the real_game is already advanced to the committed prefix (do not reset)."""
    import numpy as np
    from e125 import verify
    base = real_game.levels; cur = np.asarray(real_game.frame).copy()
    verified, new_trans = [], []
    for i, a in enumerate(plan):
        try:
            pred_nf, _ = predict_fn(cur, list(a))
        except Exception:
            pred_nf = cur
        real_game.step(*a); real_nf = np.asarray(real_game.frame)
        if real_game.levels > base:
            verified.append(a)
            return {"solved": True, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": None}
        if not np.array_equal(verify._masked(pred_nf, mask), verify._masked(real_nf, mask)):
            new_trans.append({"frame": cur.copy(), "action": list(a), "next_frame": real_nf.copy(), "level_up": False})
            return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": i}
        verified.append(a); cur = real_nf.copy()
    return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": None}
