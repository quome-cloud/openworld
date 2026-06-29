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


def solve_game(game, llm=None, mode="search", budget=None, logdir=None):
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
        if mode == "slm" and llm is not None:
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
            break
        actions += seq
        # re-apply to advance the real game state for the next iteration
        game.reset()
        for a in actions: game.step(*a)
    # verify before banking
    reached, _ = planner.replay_levels(game, actions)
    verified = reached >= game.levels and reached > 0
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
