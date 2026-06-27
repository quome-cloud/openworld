"""Execute a sim-planned trajectory against the REAL env, step-by-step, halting the instant the real masked
next-frame diverges from predict()'s -- that divergence is the model-surprise signal (E122-style); the real
transition is recorded so the model can be re-synthesized. Only verified plans touch the env (action-efficient)."""
import numpy as np
from e125 import verify


def execute_plan(real_game, plan, predict_fn, mask):
    real_game.reset(); base = real_game.levels
    cur = np.asarray(real_game.frame).copy()
    verified, new_trans = [], []
    for i, a in enumerate(plan):
        try:
            pred_nf, _ = predict_fn(cur, list(a))
        except Exception:
            pred_nf = cur
        real_game.step(*a)
        real_nf = np.asarray(real_game.frame)
        if real_game.levels > base:
            verified.append(a)
            return {"solved": True, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": None}
        if not np.array_equal(verify._masked(pred_nf, mask), verify._masked(real_nf, mask)):
            new_trans.append({"frame": cur.copy(), "action": list(a), "next_frame": real_nf.copy(),
                              "level_up": False})
            return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": i+1}
        verified.append(a); cur = real_nf.copy()
    return {"solved": False, "verified_prefix": verified, "new_transitions": new_trans, "halt_step": None}
