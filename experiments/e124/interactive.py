"""E124 interactive receding-horizon loop (model-predictive control).

Diagnosis (see memory arc3-e124-codex-search): one-shot goal decomposition from a single static frame
doesn't crack deep procedures — codex's predicted waypoints aren't reachable/ordered. Fix: ground codex
every round. Each round codex PLANS the next few steps (a 3-5 step maneuver) from the REAL current frame; the
env executes+verifies it; codex REPLANS from the actually-reached frame. Myopic 1-step planning gets greedily
stuck, so the horizon is a short maneuver, not a single action.

`solve_interactive` is planner-agnostic: pass a mock planner for tests, or the live codex planner
(codex_goalc.plan_ahead) for real runs. The env decides correctness (a replay-verified level-up)."""
import numpy as np
from e124 import search


def _sig(masked_frame):
    return hash(np.asarray(masked_frame).tobytes())


def solve_interactive(game_factory, planner, mask, max_rounds=8, horizon=5):
    """game_factory() -> a fresh game (search replays via reset). planner(frame, history, round, horizon) ->
    a list of candidate plans, each plan a list of action-arg steps like [[3],[3],[5]]. Commit the first plan
    that reaches a NOT-yet-seen state; a level-up at any point (env-verified) returns the winning sequence.
    Returns the solving action sequence or None."""
    committed = []
    seen = set()
    history = []
    for rnd in range(max_rounds):
        raised, frame = search._run_seq(game_factory(), committed, mask)
        if raised:
            return committed
        seen.add(_sig(frame))
        plans = planner(frame, history, rnd, horizon) or []
        best = None
        tried = []
        for plan in plans:
            if not plan:
                continue
            r, mf = search._run_seq(game_factory(), committed + plan, mask)
            if r:
                return committed + plan                 # env-verified solve
            sig = _sig(mf)
            novel = sig not in seen
            tried.append({"plan": plan, "novel": novel})
            if novel and best is None:
                best = (plan, sig)
        if best is None:                                # no plan reached a new state -> stuck
            history.append({"round": rnd, "progressed": False, "tried": tried})
            break
        committed += best[0]
        seen.add(best[1])
        history.append({"round": rnd, "progressed": True, "depth": len(committed), "tried": tried})
    return None
