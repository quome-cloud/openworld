"""Per-game orchestration: probe -> (optional subgoal) -> search each level -> bank replay-verified."""
import json, time
import numpy as np
from e119 import perceive, planner, slm


def _candidates_fn(game, mask):
    avail_dir = [a for a in getattr(game, "avail", [1, 2, 3, 4, 5, 7]) if a != 6]
    has_click = 6 in getattr(game, "avail", [])

    def fn(frame):
        acts = [(a,) for a in avail_dir]
        if has_click:
            acts += [(6, x, y) for (x, y) in perceive.click_candidates(frame)]
        return acts
    return fn


def solve_game(game, llm=None, mode="search", budget=None, logdir=None, make=None):
    budget = budget or {"max_nodes": 4000, "max_depth": 40}
    game.reset()
    win = game.win
    name = type(game).__name__ if not isinstance(getattr(game, "gid", None), str) else game.gid
    actions = []
    log = []
    while not game.done and game.levels < win:
        trans = perceive.probe(game)
        mask = perceive.status_mask([t["before"] for t in trans] + [t["after"] for t in trans])
        key_fn = lambda f, m=mask: perceive.state_key(f, m)
        cands = _candidates_fn(game, mask)
        score_fn = None
        subgoal = None
        if mode in ("slm", "macro+slm") and llm is not None:
            frames = [t["after"] for t in trans]
            oj = perceive.object_json(trans[0]["before"])
            subgoal = slm.propose_subgoal(llm, oj, frames)
            if subgoal is not None:
                pred = slm.compile_predicate(subgoal)
                score_fn = lambda f, p=pred: 1.0 if p(f) else 0.0   # frontier prefers goal-satisfying frames
        # search from the CURRENT progress: replay known actions, then search the next level
        seq = planner.search_level(_PrefixGame(game, actions), cands, key_fn, budget, score_fn)
        rec = {"level_index": game.levels, "subgoal": subgoal, "found": seq is not None,
               "ts": None}
        log.append(rec)
        if seq is None:
            if mode in ("macro", "macro+slm") and llm is not None:
                seq = _macro_fallback(game, actions, trans, llm, key_fn, make)
            if seq is None:
                break
        actions += seq
        # re-apply to advance the real game state for the next iteration
        game.reset()
        for a in actions: game.step(*a)
    # verify before banking. The real arc env's reset() retains completed-level progress
    # (it resets to the current-level checkpoint, not game start), so replaying on the
    # reused, progress-polluted game makes replay_levels' delta collapse to 0. Verify on a
    # FRESH env when we can build one (mirrors arc3_harness.replay's new-Game pattern); fall
    # back to the passed game for mocks that reset truly. reached is the replay-confirmed
    # level count, so it is the level count we report and trust.
    gid = getattr(game, "gid", None)
    verify_game = make(gid) if (make is not None and isinstance(gid, str)) else game
    reached, _ = planner.replay_levels(verify_game, actions)
    verified = reached > 0
    result = {"game": name, "mode": mode, "levels": reached, "win": win,
              "actions": actions, "verified": bool(verified)}
    if logdir is not None and verified:
        import pathlib
        d = pathlib.Path(logdir); d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}_solved.json").write_text(json.dumps(result))
        (d / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in log))
    return result


class _PrefixGame:
    """Wraps a GameLike so search starts AFTER a fixed action prefix (the levels already solved)."""
    def __init__(self, game, prefix):
        self._g = game; self._prefix = list(prefix)
        self.win = game.win; self.reset()
    def reset(self):
        self._g.reset()
        for a in self._prefix: self._g.step(*a)
        self.levels = self._g.levels; self.done = self._g.done; self.frame = self._g.frame
        self.avail = getattr(self._g, "avail", [1, 2, 3, 4, 5, 7])
        return self.frame
    def step(self, a, x=None, y=None):
        self._g.step(a, x, y)
        self.levels = self._g.levels; self.done = self._g.done; self.frame = self._g.frame
        return self.frame


def _macro_fallback(game, actions, trans, llm, key_fn, make):
    """On a stall, synthesize a subgoal, propose+rank macros, and return the FIRST macro whose
    fresh-env replay of (actions+macro) raises levels. Returns the macro (list of action tuples)
    or None (honest stop). The env decides correctness."""
    from e119 import macro
    avail = list(getattr(game, "avail", [1, 2, 3, 4, 5, 7]))
    oj = perceive.object_json(trans[0]["before"])
    diffs = [perceive.contrastive_diff(t["before"], t["after"]) for t in trans]
    try:
        subgoal = slm.propose_subgoal(llm, oj, [t["after"] for t in trans])
    except Exception:
        subgoal = None      # a flaky/parse-failed subgoal must not kill the macro fallback
    cands = macro.propose_macros(llm, game, actions, oj, diffs, subgoal, avail, key_fn)
    if not cands:
        return None
    ranked = macro.rank_macros(cands, game, actions, subgoal, key_fn, seen=set())
    base_levels = _levels_after(make, game, actions)
    for m in ranked:
        reached = _levels_after(make, game, actions + list(m))
        if reached > base_levels:
            return list(m)
    return None


def _levels_after(make, game, action_list):
    """Fresh-env replay (Bug #2): make a new game from gid when possible, else reuse `game`."""
    gid = getattr(game, "gid", None)
    g = make(gid) if (make is not None and isinstance(gid, str)) else game
    reached, _ = planner.replay_levels(g, action_list)
    return reached
