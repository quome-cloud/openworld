"""The single-level loop: explore -> synthesize predict()+goal_score (verifier-gated dynamics, autonomous win
hypothesis) -> plan IN SIMULATION by best-first ENERGY DESCENT on goal_score -> execute vs the real env,
halting on mismatch OR a refuted win hypothesis -> add the surprising transition + re-synthesize -> repeat.
Only verified plans touch the env. The env decides correctness (a real levels bump)."""
from e125 import explorer, simworld, execute


def solve_level(game_factory, candidates_fn, action_api, game, mask, synth_fn,
                budget_explore=60, budget_plan=20000, rounds=6, traces_dir=None):
    trans = explorer.collect(game_factory, candidates_fn, budget_explore)
    real_actions = budget_explore
    committed = []
    seen_keys = {(t["frame"].tobytes(), tuple(t["action"])) for t in trans}   # dedup grounded transitions
    last_src = last_goal = None                          # carry the prior verified program forward (per-level)
    for rnd in range(rounds):
        src, fn, goal_fn = synth_fn(trans, action_api, game, mask, traces_dir=traces_dir, seed_src=last_src)
        if fn is None:
            return {"solved": False, "actions": committed, "rounds_used": rnd, "real_actions": real_actions,
                    "reason": "no verified predict()"}
        last_src = src
        if goal_fn is not None:                          # a reused full seed returns goal_fn=None -> keep cache
            last_goal = goal_fn
        goal_fn = goal_fn or last_goal
        init = game_factory(); init.reset()
        for a in committed:
            init.step(*a)
        plan = simworld.plan(fn, init.frame, candidates_fn, budget_plan, goal_fn=goal_fn)
        if plan is None:
            # No predicted win is reachable in sim (the offline win hypothesis never fires). Fall back to
            # GOAL-DIRECTED real-env exploration: descend goal_score toward the hypothesised win to reach and
            # GROUND a real level-up (the online oracle). A grounded level_up=True transition makes the next
            # synth learn a VERIFIED win condition -> the replan then finds it. (Restartable-agent strategy.)
            gd = explorer.goal_directed_collect(game_factory, candidates_fn, fn, goal_fn, budget_explore)
            real_actions += len(gd)
            fresh = [t for t in gd if (t["frame"].tobytes(), tuple(t["action"])) not in seen_keys]
            for t in fresh:
                seen_keys.add((t["frame"].tobytes(), tuple(t["action"])))
            if fresh:
                trans = trans + fresh                    # re-synthesize next round with the new (grounded) data
                continue
            return {"solved": False, "actions": committed, "rounds_used": rnd, "real_actions": real_actions,
                    "reason": "no sim plan"}
        # replay the committed prefix on a fresh real game, then verify only the new `plan` segment
        rg = game_factory(); rg.reset()
        for a in committed:
            rg.step(*a)
        res = execute.execute_plan(rg, plan, fn, mask, do_reset=False)
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


def solve_game(game_factory, candidates_fn, action_api, game, synth_obj_fn, perceive=None, macro_runner=None,
               budget_explore=60, budget_plan=20000, rounds_per_level=4, max_levels=9, max_macros=8,
               traces_dir=None):
    """Per-level loop over the verified object-world. Explore -> synth_obj (seeded from the prior level's
    verified program = rule-library transfer) -> traverse_level (committed=solution so far). On a real level-up
    extend the solution and carry the program forward; on surprise re-synth; on stall stop. The env decides wins."""
    from e125 import explorer, traverse, objstate
    perceive = perceive or objstate.object_state
    solution = []                 # actions through solved levels
    rule_src = None               # last verified predict src (transfer seed for the next level)
    levels = []; real_actions = 0
    for level in range(max_levels):
        trans = explorer.collect_obj(game_factory, candidates_fn, budget_explore, perceive, prefix=solution)
        real_actions += budget_explore
        last_src = rule_src; solved = False; reason = "no model"
        for _ in range(rounds_per_level):
            src, fn, goal_fn, ensemble = synth_obj_fn(trans, action_api, game, seed_src=last_src,
                                                      traces_dir=traces_dir)
            if fn is None:
                reason = "no verified predict()"; break
            last_src = src
            wm = {"predict_src": src, "predict_fn": fn, "goal_src": None, "goal_fn": goal_fn, "ensemble": ensemble}
            res = traverse.traverse_level(game_factory, candidates_fn, wm, action_api, game,
                                          macro_runner=macro_runner, perceive=perceive, committed=list(solution),
                                          budget_plan=budget_plan, max_macros=max_macros, traces_dir=traces_dir)
            real_actions += max(0, len(res["actions"]) - len(solution)) + res["macros_used"]
            reason = res["reason"]
            if res["solved"]:
                solution = res["actions"]; rule_src = src; solved = True; break
            if res["new_transitions"]:
                trans = trans + res["new_transitions"]      # surprise -> re-synthesize (seeded)
            else:
                break                                        # stall/no progress
        levels.append({"level": level, "solved": solved, "reason": reason})
        if not solved:
            break
    return {"levels_solved": sum(1 for l in levels if l["solved"]), "solution": solution,
            "levels": levels, "real_actions": real_actions}
