"""Phase-2 traversal: drive a level by PLANNING in the verified object-world (imagination-primary), executing
verified plans/macros against the REAL env. Imagination plan first; if the ensemble agrees along it, execute it;
else ask an LLM (codex; the Claude fallback is already wired in synth) for a short MACRO and execute that. The
env's g.levels decides the win. Source-free + solution-free -- the LLM never sees game source or banked answers."""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from e125 import simworld, execute, synth, objstate
from e124 import codex_iso

MACRO_SCHEMA = {"type": "object", "additionalProperties": False,
                "required": ["macro", "rationale", "goal_note"],
                "properties": {"macro": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                               "rationale": {"type": "string"}, "goal_note": {"type": "string"}}}


def _macro_prompt(state, action_api, predict_src, goal_src, history):
    hist = "\n".join(f"- tried {h['macro']} -> {h['outcome']}" for h in history[-6:]) or "(none yet)"
    predict_block = f"```python\n{predict_src}\n```" if predict_src else "(unavailable)"
    goal_block = f"```python\n{goal_src}\n```" if goal_src else "(unknown -- hypothesise from object config)"
    return ("You are solving an unknown grid game by acting. You have a VERIFIED world model (predict) and a goal "
            f"energy (goal_score) of the current OBJECT state. Propose a SHORT macro (3-5 actions) toward the win.\n\n"
            f"predict():\n{predict_block}\ngoal_score():\n{goal_block}\n"
            f"Current state: {synth._objs(state)}\nActions: {action_api}\n"
            f"Macros already tried (do not repeat fruitless ones):\n{hist}\n\n"
            "Return JSON {macro: [[a],...], rationale, goal_note}. macro is a list of actions like [[4],[4],[6,3,5]].")


def _max_disagreement(plan, predict_fn, ensemble, initial_state):
    """Replay the plan through predict_fn; at each state measure ensemble disagreement; return the max."""
    if not ensemble or len(ensemble) <= 1:
        return 0.0
    s = initial_state
    worst = 0.0
    for a in plan:
        worst = max(worst, synth.ensemble_disagreement(ensemble, s, a))
        try:
            s, _ = predict_fn(copy.deepcopy(s), list(a))
        except Exception:
            break
    return worst


def traverse_level(game_factory, candidates_fn, wm, action_api, game, macro_runner=None, perceive=None,
                   committed=None, budget_plan=20000, max_macros=8, stall_macros=3, disagreement_thresh=0.0,
                   traces_dir=None):
    """Drive a single level by planning in the object-world (imagination-primary).

    Each round:
      1. plan_obj in imagination using wm["predict_fn"] + wm["goal_fn"].
      2. If a plan exists AND max ensemble_disagreement along it <= disagreement_thresh -> execute via execute_obj.
      3. Else ask macro_runner for a 3-5 action macro and execute that.

    Returns {"solved":bool, "actions":list, "new_transitions":list, "reason":str, "macros_used":int}.
    A surprise halts and returns so the caller can re-synthesize (reason="surprise").
    """
    perceive = perceive or objstate.object_state
    run = macro_runner or codex_iso.run
    committed = list(committed or [])
    predict_fn = wm["predict_fn"]
    goal_fn = wm.get("goal_fn")
    ensemble = wm.get("ensemble") or [predict_fn]
    history = []
    new_trans = []
    macros_used = 0
    stall = 0

    def _state_after(prefix):
        """Replay prefix actions on a fresh game instance; return (game, object_state)."""
        g = game_factory()
        g.reset()
        for a in prefix:
            g.step(*a)
        return g, perceive(g.frame)

    # Initialise seen_states with the current committed prefix's object state so that
    # a round that leaves the committed state KEY unchanged is counted as a stall cycle.
    _, _init_st = _state_after(committed)
    seen_states = {objstate.state_key(_init_st)}

    for _ in range(max_macros):
        _, init_state = _state_after(committed)

        # 1. Attempt imagination plan
        plan = simworld.plan_obj(predict_fn, init_state, candidates_fn, budget_plan, goal_fn=goal_fn)
        use_plan = (plan is not None and
                    _max_disagreement(plan, predict_fn, ensemble, init_state) <= disagreement_thresh)

        if use_plan:
            actions = plan
        else:
            # 2. Macro fallback
            res = run(
                _macro_prompt(init_state, action_api, wm.get("predict_src"), wm.get("goal_src"), history),
                MACRO_SCHEMA, "gpt-5.5", game)
            macro = (res.get("final") or {}).get("macro") or []
            actions = [list(a) for a in macro if a]
            macros_used += 1
            if not actions:
                stall += 1
                if stall >= stall_macros:
                    return {"solved": False, "actions": committed, "new_transitions": new_trans,
                            "reason": "stall", "macros_used": macros_used}
                continue

        # 3. Execute actions against real env (fresh game replayed to committed prefix)
        rg = game_factory()
        rg.reset()
        for a in committed:
            rg.step(*a)
        r = execute.execute_obj(rg, actions, predict_fn, perceive, do_reset=False)
        committed += r["verified_prefix"]

        if r["solved"]:
            return {"solved": True, "actions": committed, "new_transitions": new_trans,
                    "reason": "solved", "macros_used": macros_used}

        if r["new_transitions"]:
            new_trans += r["new_transitions"]
            return {"solved": False, "actions": committed, "new_transitions": new_trans,
                    "reason": "surprise", "macros_used": macros_used}

        # No progress this round — detect state novelty to distinguish genuine progress from cycles.
        _, cur_state = _state_after(committed)
        cur_key = objstate.state_key(cur_state)
        if cur_key in seen_states:
            stall += 1
        else:
            stall = 0
            seen_states.add(cur_key)

        history.append({"macro": actions, "outcome": "no progress"})
        if stall >= stall_macros:
            return {"solved": False, "actions": committed, "new_transitions": new_trans,
                    "reason": "stall", "macros_used": macros_used}

    return {"solved": False, "actions": committed, "new_transitions": new_trans,
            "reason": "max_macros", "macros_used": macros_used}
