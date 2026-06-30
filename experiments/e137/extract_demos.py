"""CLI helper: build an E137 schema packet from a source-free frontier trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from experiments.arc3_sandbox import SandboxGame
    from experiments.e137.schema_induction import build_packet, replay_records, write_packet
except ImportError:  # flat audited workspace
    from arc3_sandbox import SandboxGame
    from schema_induction import build_packet, replay_records, write_packet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("frontier_json")
    ap.add_argument("out_json")
    ap.add_argument("--priority", default="")
    args = ap.parse_args()

    frontier = json.loads(Path(args.frontier_json).read_text())
    actions = frontier.get("actions") or []
    game = frontier.get("game") or args.game
    env = SandboxGame(game)
    try:
        records = replay_records(env, actions)
    finally:
        env.close()
    packet = build_packet(
        game=game,
        records=records,
        frontier_actions=actions,
        frontier_levels=int(frontier.get("levels", 0)),
        win=int(frontier.get("win", 0)),
        target_games_rank=[x for x in args.priority.split(",") if x],
    )
    write_packet(packet, args.out_json)
    print(
        f"[e137] wrote {args.out_json}: {packet['n_solved_level_demos']} demos, "
        f"{len(packet['candidate_schemas'])} schemas"
    )


if __name__ == "__main__":
    main()

