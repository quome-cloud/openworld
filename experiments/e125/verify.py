"""The verifier gate: a synthesized predict(frame,action)->(next_frame,level_up) is ACCEPTED only if it
exact-matches every held-out transition (masked next-frame equality + level_up equality). Predicts are
compiled in-process for speed (codex is not adversarial; a predict that errors fails the gate)."""
import copy
import numpy as np
from e125 import objstate as _objstate


def compile_predict(src):
    ns = {"np": np, "__builtins__": __builtins__}
    try:
        exec(src, ns)
        fn = ns.get("predict")
        return fn if callable(fn) else None
    except Exception:
        return None


def compile_goal(src):
    """Compile a synthesized goal_score(frame)->float (an energy/heuristic). Unlike predict(), goal_score is
    NOT gated against data (there is no ground-truth energy) -- it only has to compile and return a number;
    planning descends it. Returns the callable or None."""
    ns = {"np": np, "__builtins__": __builtins__}
    try:
        exec(src, ns)
        fn = ns.get("goal_score")
        return fn if callable(fn) else None
    except Exception:
        return None


def _masked(frame, mask):
    fr = np.asarray(frame)
    return np.where(mask, 0, fr) if mask is not None else fr


def check(predict_fn, transitions, mask):
    """Return (ok, counterexample). ok iff predict_fn reproduces every transition (masked next_frame + level_up)."""
    if predict_fn is None:
        return False, (transitions[0] if transitions else None)
    for t in transitions:
        try:
            nf, lu = predict_fn(np.asarray(t["frame"]), list(t["action"]))
        except Exception:
            return False, t
        if not np.array_equal(_masked(nf, mask), _masked(t["next_frame"], mask)):
            return False, t
        if bool(lu) != bool(t["level_up"]):
            return False, t
    return True, None


# --- decision-equivalent gate over OBJECT states (value-equivalent, not pixel reconstruction) ---

def score_obj(predict_fn, transitions, fields=("color", "y", "x")):
    """(n_matched, fails). Match iff the DECISION-RELEVANT state_key of the predicted next_state equals the
    real one AND level_up matches. fails = [(transition, predicted_next_state|None)]."""
    if predict_fn is None:
        return 0, [(t, None) for t in transitions]
    n, fails = 0, []
    for t in transitions:
        try:
            ns, lu = predict_fn(copy.deepcopy(t["state"]), list(t["action"]))
        except Exception:
            fails.append((t, None)); continue
        if (_objstate.state_key(ns, fields) == _objstate.state_key(t["next_state"], fields)
                and bool(lu) == bool(t["level_up"])):
            n += 1
        else:
            fails.append((t, ns))
    return n, fails


def check_obj(predict_fn, transitions, fields=("color", "y", "x")):
    """(ok, counterexample). ok iff every transition matches on the decision-relevant key + level_up."""
    n, fails = score_obj(predict_fn, transitions, fields)
    return (len(fails) == 0 and predict_fn is not None), (fails[0][0] if fails else None)
