"""E124 entry: run the ablation ladder (blind / blind+macros) on a single level of each
pilot game and record which rungs beat the blind floor. Codex compiles the goal source-free;
the env decides correctness. save_results BEFORE asserts (CLAUDE.md). Milestone 1 (single
level); deep chaining is M2 (deferred pending M1 gate)."""
import os, sys, argparse, json

sys.path.insert(0, os.path.dirname(__file__))

from e124 import search
from e124.codex_goalc import compile_goal, Goal
from common import save_results

# MVP ladder: does codex's macros beat blind?  subgoals/full are Task 6b.
RUNGS = ["blind", "blind_macros"]

# Blind rung gets no macros — a true control.
_EMPTY_GOAL = Goal([], [], None, "", True, [])


def run_one(game_factory, candidates_fn, mask, budget, goal, rungs=RUNGS):
    """Run the ablation ladder on one level.

    Parameters
    ----------
    game_factory : callable
        Called with no arguments; returns a FRESH game instance for each rung.
        (Search mutates game state via reset/replay, so isolation is required.)
    candidates_fn : callable
        frame -> list of action-arg lists (e.g. [[1], [2], [6, x, y]]).
    mask : array-like or None
        Status-bar mask forwarded to search.run.
    budget : int
        Step budget per rung.
    goal : Goal
        Compiled goal from compile_goal (or injected in tests).  The blind rung
        always receives _EMPTY_GOAL so it is a genuine control.
    rungs : list[str]
        Rung names to evaluate.  Defaults to RUNGS = ["blind", "blind_macros"].

    Returns
    -------
    dict[str, int | None]
        {rung: solution_len_or_None}.  None means the rung failed within budget.
    """
    out = {}
    for rung in rungs:
        # Blind rung gets no macros — do not leak codex info into the control.
        g = goal if rung == "blind_macros" else _EMPTY_GOAL
        seq = search.run(game_factory(), g, budget, rung, candidates_fn, mask)
        out[rung] = (len(seq) if seq is not None else None)
    return out


def main():
    ap = argparse.ArgumentParser(description="E124 Milestone-1 ablation ladder")
    ap.add_argument("--games", default="tn36",
                    help="Comma-separated game IDs (e.g. tn36,ka59)")
    ap.add_argument("--budget", type=int, default=4000,
                    help="Search step budget per rung (default 4000)")
    ap.add_argument("--model", default="gpt-5.5",
                    help="Codex model for compile_goal (default gpt-5.5)")
    ap.add_argument("--traces", default="experiments/results/e124_traces",
                    help="Directory for codex call telemetry")
    a = ap.parse_args()

    # Live imports — only reached when the controller runs the real gate.
    # Tests must NOT import or instantiate these.
    from arc3_sandbox import SandboxGame   # noqa: F401 (live path only)
    from e119 import perceive              # noqa: F401 (live path only)

    results = {}
    for gid in a.games.split(","):
        game = SandboxGame(gid)
        game.reset()
        frames = [game.frame]

        mask = perceive.status_mask(frames)

        # Directional pilot: skip click action (6).
        avail = getattr(game, "avail", list(range(1, 8)))
        candidates_fn = lambda fr, _a=avail: [[act] for act in _a if act != 6]

        goal = compile_goal(
            frames,
            action_api="env.step(action) — actions 1-5,7 are directional; 6 is click",
            dynamics="",
            game=gid,
            level=0,
            regime=0,
            model=a.model,
            traces_dir=a.traces,
        )

        results[gid] = run_one(
            game_factory=lambda _gid=gid: SandboxGame(_gid),
            candidates_fn=candidates_fn,
            mask=mask,
            budget=a.budget,
            goal=goal,
        )

    # CLAUDE.md: save_results BEFORE asserts.
    save_results("e124_autonomous_search", {
        "experiment": "e124_autonomous_search",
        "games": results,
    })
    print("[e124]", json.dumps(results))


if __name__ == "__main__":
    main()
