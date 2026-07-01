"""Execute a scratch ARC proposal with compact output.

This helper is source-free with respect to game internals: it imports the
scratch directory's public sandbox wrapper and replays frontier/proposal JSON.
It is intended for budgeted probe loops where verbose per-step summaries are
too noisy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _act(game, action):
    if action[0] == 6:
        return game.step(6, action[1], action[2])
    return game.step(action[0])


def execute(scratch: Path, proposal_path: Path, solved_out: Path, counterexample_out: Path) -> dict:
    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    proposal = json.loads(proposal_path.read_text())
    actions = list(frontier["actions"])

    game = SandboxGame(frontier["game"])
    try:
        game.reset()
        for action in actions:
            _act(game, action)
        start_level = int(game.levels)
        used = []
        for action in proposal["probe_plan"]:
            _act(game, action)
            used.append(action)
            if int(game.levels) > start_level:
                solved = {
                    "game": frontier["game"],
                    "actions": actions + used,
                    "levels": int(game.levels),
                    "win": int(game.win),
                    "proposal_id": proposal.get("proposal_id"),
                }
                solved_out.write_text(json.dumps(solved, indent=2) + "\n")
                return {
                    "proposal_id": proposal.get("proposal_id"),
                    "outcome": "level_up",
                    "start_level": start_level,
                    "end_level": int(game.levels),
                    "steps": len(used),
                    "solved_out": str(solved_out),
                }
            if bool(game.done):
                break
        counterexample = {
            "proposal_id": proposal.get("proposal_id"),
            "start_level": start_level,
            "end_level": int(game.levels),
            "steps": len(used),
            "executed": used,
            "done": bool(game.done),
        }
        counterexample_out.write_text(json.dumps(counterexample, indent=2) + "\n")
        return {
            "proposal_id": proposal.get("proposal_id"),
            "outcome": "no_level_up",
            "start_level": start_level,
            "end_level": int(game.levels),
            "steps": len(used),
            "counterexample_out": str(counterexample_out),
        }
    finally:
        game.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one ARC scratch proposal quietly.")
    parser.add_argument("scratch")
    parser.add_argument("proposal")
    parser.add_argument("--solved-out", default="solved_probe.json")
    parser.add_argument("--counterexample-out", default="counterexample_probe.json")
    args = parser.parse_args()

    scratch = Path(args.scratch).resolve()
    proposal = Path(args.proposal)
    if not proposal.is_absolute():
        proposal = scratch / proposal
    result = execute(
        scratch,
        proposal.resolve(),
        Path(args.solved_out).resolve(),
        Path(args.counterexample_out).resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

