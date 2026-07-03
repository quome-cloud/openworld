"""Iterative source-free controller using E143 behavioral-memory transfer.

E144 takes the E143 result from "this suffix advances one frontier" to a
full-run loop. At each stage it records the current source-free frontier state,
retrieves compatible suffixes from prior action traces by frame signature, tests
them against the live sandbox, and commits only verifier-confirmed level-ups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from experiments.e143.transition_miner import (
    action_list,
    extract_behavioral_suffixes,
    rank_behavioral_suffixes_by_signature,
    record_trace,
    write_transfer_proposals,
)
from scripts.execute_arc_probe_quiet import execute


def write_frontier(scratch: Path, game: str, actions: Sequence[Sequence[int]]) -> Path:
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / "frontier.json"
    payload = {"game": game, "actions": action_list(actions)}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def prepare_scratch(scratch: Path, game: str, actions: Sequence[Sequence[int]], sandbox_path: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sandbox_path, scratch / "arc3_sandbox.py")
    write_frontier(scratch, game, actions)


def frontier_actions(scratch: Path) -> list[list[int]]:
    frontier = json.loads((scratch / "frontier.json").read_text())
    return action_list(frontier.get("actions", []))


def load_solution_actions(solved_path: Path) -> list[list[int]]:
    solved = json.loads(solved_path.read_text())
    return action_list(solved.get("actions", []))


def current_level(trace: Mapping[str, Any]) -> int:
    states = trace.get("states", [])
    if isinstance(states, list) and states and isinstance(states[-1], Mapping):
        return int(states[-1].get("levels", 0))
    return 0


def filtered_ranked_suffixes(
    scratch: Path,
    trace: Mapping[str, Any],
    solution_paths: Sequence[Path],
    *,
    transfer_limit: int,
    signature_threshold: float,
) -> list[dict[str, Any]]:
    suffixes = extract_behavioral_suffixes(trace, solution_paths, max_suffixes=max(transfer_limit * 6, 24))
    ranked = rank_behavioral_suffixes_by_signature(scratch, trace, suffixes)
    return [
        s
        for s in ranked
        if float(s.get("signature_distance", float("inf"))) <= signature_threshold
    ][:transfer_limit]


def execute_stage_candidates(
    scratch: Path,
    stage_dir: Path,
    proposal_paths: Sequence[Path],
) -> dict[str, Any] | None:
    attempts: list[dict[str, Any]] = []
    for idx, proposal in enumerate(proposal_paths, start=1):
        solved_out = stage_dir / f"solved_{idx:02d}.json"
        counterexample_out = stage_dir / f"counterexample_{idx:02d}.json"
        result = execute(scratch, proposal, solved_out, counterexample_out)
        attempts.append(result)
        if result.get("outcome") == "level_up":
            solved_result = dict(result)
            solved_result["solved_actions"] = load_solution_actions(solved_out)
            solved_result["attempts"] = [dict(a) for a in attempts]
            return solved_result
    if attempts:
        (stage_dir / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
    return None


def run_full_controller(
    game: str,
    trace_dir: Path,
    scratch: Path,
    out_dir: Path,
    *,
    sandbox_path: Path = Path("experiments/arc3_sandbox.py"),
    max_stages: int = 12,
    transfer_limit: int = 24,
    signature_threshold: float = 1000.0,
    solution_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    if solution_paths is None:
        solution_paths = sorted(trace_dir.glob(f"{game}__*.json"))
    else:
        solution_paths = sorted(Path(p) for p in solution_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_scratch(scratch, game, [], sandbox_path)

    actions: list[list[int]] = []
    stages: list[dict[str, Any]] = []
    final_level = 0
    stopped_reason = "max_stages"

    for stage_idx in range(max_stages):
        stage_dir = out_dir / f"stage_{stage_idx:02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        write_frontier(scratch, game, actions)
        trace = record_trace(scratch)
        level_before = current_level(trace)
        final_level = level_before
        (stage_dir / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")

        ranked = filtered_ranked_suffixes(
            scratch,
            trace,
            solution_paths,
            transfer_limit=transfer_limit,
            signature_threshold=signature_threshold,
        )
        (stage_dir / "ranked_suffixes.json").write_text(json.dumps(ranked, indent=2) + "\n")
        if not ranked:
            stopped_reason = "no_compatible_suffix"
            stages.append(
                {
                    "stage": stage_idx,
                    "level_before": level_before,
                    "level_after": level_before,
                    "frontier_actions": len(actions),
                    "outcome": "no_compatible_suffix",
                }
            )
            break

        proposal_paths = write_transfer_proposals(stage_dir / "proposals", trace, ranked)
        solved = execute_stage_candidates(scratch, stage_dir, proposal_paths)
        if solved is None:
            stopped_reason = "no_candidate_level_up"
            stages.append(
                {
                    "stage": stage_idx,
                    "level_before": level_before,
                    "level_after": level_before,
                    "frontier_actions": len(actions),
                    "candidate_count": len(proposal_paths),
                    "outcome": "no_candidate_level_up",
                }
            )
            break

        new_actions = action_list(solved["solved_actions"])
        added = len(new_actions) - len(actions)
        actions = new_actions
        final_level = int(solved.get("end_level", level_before))
        write_frontier(scratch, game, actions)
        stage_record = {
            "stage": stage_idx,
            "level_before": level_before,
            "level_after": final_level,
            "frontier_actions_before": len(actions) - added,
            "frontier_actions_after": len(actions),
            "actions_added": added,
            "candidate_count": len(proposal_paths),
            "proposal_id": solved.get("proposal_id"),
            "steps": solved.get("steps"),
            "outcome": "level_up",
            "solved_out": solved.get("solved_out"),
            "attempts": solved.get("attempts", []),
        }
        stages.append(stage_record)
        (stage_dir / "stage_result.json").write_text(json.dumps(stage_record, indent=2) + "\n")
        if final_level >= 9:
            stopped_reason = "completed_level_9"
            break

    result = {
        "experiment": "E144",
        "game": game,
        "trace_dir": str(trace_dir),
        "scratch": str(scratch),
        "max_stages": max_stages,
        "transfer_limit": transfer_limit,
        "signature_threshold": signature_threshold,
        "levels": final_level,
        "actions": actions,
        "action_count": len(actions),
        "stopped_reason": stopped_reason,
        "stages": stages,
    }
    (out_dir / "full_run_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E144 full-run behavioral-memory controller.")
    parser.add_argument("game")
    parser.add_argument("--trace-dir", default="experiments/results/arc3_traces/solutions")
    parser.add_argument("--scratch-dir", default="/tmp/e144_scratch")
    parser.add_argument("--out-dir", default="/tmp/e144_run")
    parser.add_argument("--sandbox-path", default="experiments/arc3_sandbox.py")
    parser.add_argument("--max-stages", type=int, default=12)
    parser.add_argument("--transfer-limit", type=int, default=24)
    parser.add_argument("--signature-threshold", type=float, default=1000.0)
    args = parser.parse_args()

    result = run_full_controller(
        args.game,
        Path(args.trace_dir).resolve(),
        Path(args.scratch_dir).resolve(),
        Path(args.out_dir).resolve(),
        sandbox_path=Path(args.sandbox_path).resolve(),
        max_stages=args.max_stages,
        transfer_limit=args.transfer_limit,
        signature_threshold=args.signature_threshold,
    )
    print(json.dumps({k: result[k] for k in ("experiment", "game", "levels", "action_count", "stopped_reason")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
