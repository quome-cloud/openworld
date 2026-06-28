"""Execute a sim-planned trajectory against the REAL env, step-by-step, halting the instant the real masked
next-frame diverges from predict()'s -- that divergence is the model-surprise signal (E122-style); the real
transition is recorded so the model can be re-synthesized. Only verified plans touch the env (action-efficient)."""
import numpy as np
from e125 import verify


def execute_plan(real_game, plan, predict_fn, mask, do_reset=True):
    """Run a sim-planned trajectory against the REAL env step-by-step. Halt + record a new transition on either
    surprise: (a) the real masked next-frame diverges from predict()'s, or (b) predict()'s win HYPOTHESIS fires
    (level_up=True) but the real env did NOT advance the level -- a refuted hypothesis, recorded level_up=False
    so re-synthesis revises it. The env is the authority on the win. `do_reset=False` continues from the
    real_game's current state (so the agent can resume after a committed prefix)."""
    if do_reset:
        real_game.reset()
    base = real_game.levels
    cur = np.asarray(real_game.frame).copy()
    verified, new_trans = [], []
    for i, a in enumerate(plan):
        try:
            pred_nf, pred_lu = predict_fn(cur, list(a))
        except Exception:
            pred_nf, pred_lu = cur, False
        real_game.step(*a)
        real_nf = np.asarray(real_game.frame)
        if real_game.levels > base:
            verified.append(a)
            return {"solved": True, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": None}
        if not np.array_equal(verify._masked(pred_nf, mask), verify._masked(real_nf, mask)):
            new_trans.append({"frame": cur.copy(), "action": list(a), "next_frame": real_nf.copy(),
                              "level_up": False})
            return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": i+1}
        if pred_lu:                                       # win hypothesis fired but the env did not level up
            new_trans.append({"frame": cur.copy(), "action": list(a), "next_frame": real_nf.copy(),
                              "level_up": False})
            return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": i+1}
        verified.append(a); cur = real_nf.copy()
    return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": None}
