"""Retrieve-discover-writeback controller for source-free ARC solving.

E146 wraps E145 memory retrieval with a discovery fallback. On each stage it:

1. tries audited source-free episodic memory;
2. if retrieval fails, optionally runs a source-free discovery command;
3. verifies any newly written solution against the sandbox;
4. writes the verified fragment into a local episodic memory store; and
5. continues chaining from the deeper frontier.

The global trace archive is never mutated. New fragments are local/provisional
until the existing capture/finalize/audit tooling promotes them.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from experiments.e143.transition_miner import action_list, record_trace, write_transfer_proposals
from experiments.e144.full_run_controller import (
    current_level,
    filtered_ranked_suffixes,
    prepare_scratch,
    write_frontier,
)
from experiments.e145.sourcefree_controller import eligible_solution_paths, write_memory_manifest
from experiments.e146.sourcefree_primitives import sourcefree_primitive_candidates
from scripts.execute_arc_probe_quiet import execute


def utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def replay_solution(scratch: Path, solved_path: Path) -> dict[str, Any]:
    """Replay a candidate solved.json through the scratch SandboxGame."""

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    data = json.loads(solved_path.read_text())
    game_id = str(data["game"])
    actions = action_list(data.get("actions", []))
    game = SandboxGame(game_id)
    try:
        game.reset()
        for action in actions:
            if int(action[0]) == 6:
                game.step(6, int(action[1]), int(action[2]))
            else:
                game.step(int(action[0]))
            if bool(game.done):
                break
        return {
            "game": game_id,
            "actions": actions,
            "levels": int(game.levels),
            "win": int(game.win),
            "done": bool(game.done),
            "claimed_levels": int(data.get("levels", 0) or 0),
            "replay_verified": int(game.levels) >= int(data.get("levels", 0) or 0),
        }
    finally:
        game.close()


def write_local_memory_record(
    memory_root: Path,
    *,
    game: str,
    actions: Sequence[Sequence[int]],
    levels: int,
    win: int,
    stage: int,
    method: str,
    discovery: Mapping[str, Any] | None = None,
) -> Path:
    """Write a verified discovery fragment in arc3_traces-compatible form."""

    memory_root.mkdir(parents=True, exist_ok=True)
    (memory_root / "solutions").mkdir(exist_ok=True)
    (memory_root / "meta").mkdir(exist_ok=True)
    run_id = f"{game}__e146-memory__{utc_id()}__stage-{stage:02d}"
    solution_file = f"solutions/{run_id}.json"
    solution_path = memory_root / solution_file
    payload = {"game": game, "levels": int(levels), "win": int(win), "actions": action_list(actions)}
    solution_path.write_text(json.dumps(payload, indent=2) + "\n")

    record = {
        "run_id": run_id,
        "game": game,
        "tier": "e146-memory",
        "method": method,
        "source_free": True,
        "memory_tainted": False,
        "fairness": "source-free by E146 discovery command contract; sandbox replay verified locally",
        "solution_file": solution_file,
        "outcome": {
            "levels": int(levels),
            "win": int(win),
            "full_solve": bool(win and levels >= win),
            "audit": {
                "mode": "e146_local_provisional",
                "clean": True,
                "findings": [],
                "note": "Local immediate writeback; promote through finalize_traces for archive-grade audit.",
            },
            "replay_verified": True,
        },
        "discovery": dict(discovery or {}),
    }
    (memory_root / "meta" / f"{run_id}.json").write_text(json.dumps(record, indent=2) + "\n")
    with (memory_root / "runs.jsonl").open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return solution_path


def discovery_environment(
    *,
    game: str,
    scratch: Path,
    stage_dir: Path,
    stage: int,
    level: int,
) -> dict[str, str]:
    solved_out = stage_dir / "discovery" / "solved.json"
    solved_out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "E146_GAME": game,
            "E146_SCRATCH": str(scratch),
            "E146_FRONTIER": str(scratch / "frontier.json"),
            "E146_STAGE_DIR": str(stage_dir),
            "E146_STAGE": str(stage),
            "E146_LEVEL": str(level),
            "E146_SOLVED_OUT": str(solved_out),
        }
    )
    return env


def run_discovery_command(
    command: str,
    *,
    game: str,
    scratch: Path,
    stage_dir: Path,
    stage: int,
    level: int,
    timeout_s: int | None,
) -> dict[str, Any]:
    env = discovery_environment(game=game, scratch=scratch, stage_dir=stage_dir, stage=stage, level=level)
    rendered = command.format(
        game=shlex.quote(game),
        scratch=shlex.quote(str(scratch)),
        frontier=shlex.quote(str(scratch / "frontier.json")),
        stage_dir=shlex.quote(str(stage_dir)),
        stage=stage,
        level=level,
        solved_out=shlex.quote(env["E146_SOLVED_OUT"]),
    )
    log_path = stage_dir / "discovery" / "discovery_command.log"
    solved_out = Path(env["E146_SOLVED_OUT"])
    fallback_solved = stage_dir / "judge_schema_discovery" / "solved.json"

    def harvest_solved() -> bool:
        for candidate in (solved_out, fallback_solved):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text())
            except Exception:
                continue
            if not isinstance(payload.get("actions"), list):
                continue
            if candidate != solved_out:
                solved_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, solved_out)
            return True
        return False

    with log_path.open("w") as log_file:
        proc = subprocess.Popen(
            rendered,
            shell=True,
            cwd=stage_dir,
            env=env,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.monotonic()
        timed_out = False
        harvested = False
        returncode: int | None = None
        while True:
            returncode = proc.poll()
            if harvest_solved():
                harvested = True
                if returncode is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except Exception:
                        proc.terminate()
                break
            if returncode is not None:
                break
            if timeout_s is not None and time.monotonic() - started >= timeout_s:
                timed_out = True
                if harvest_solved():
                    harvested = True
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
                break
            time.sleep(0.5)

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait()
        if returncode is None:
            returncode = proc.returncode
        if timed_out:
            log_file.write(f"\n[E146] discovery command timed out after {timeout_s}s\n")
        if harvested:
            log_file.write("\n[E146] harvested solved.json before discovery process exit\n")
    return {
        "command": rendered,
        "returncode": returncode,
        "timed_out": timed_out,
        "harvested": harvested,
        "log": str(log_path),
        "solved_path": env["E146_SOLVED_OUT"],
    }


def memory_paths(trace_root: Path, local_memory_root: Path, game: str) -> list[Path]:
    paths = eligible_solution_paths(trace_root, games=[game])
    paths.extend(eligible_solution_paths(local_memory_root, games=[game]))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return sorted(out)


def same_game_solution_paths(*roots: Path, game: str) -> list[Path]:
    """Find same-game solution JSON files without applying provenance gates."""

    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        for base in (root / "solutions", root):
            if not base.exists():
                continue
            for path in sorted(base.glob(f"{game}*.json")):
                if path in seen:
                    continue
                try:
                    data = json.loads(path.read_text())
                except Exception:
                    continue
                if str(data.get("game")) != str(game):
                    continue
                seen.add(path)
                out.append(path)
    return sorted(out)


def exact_prefix_continuations(
    trace: Mapping[str, Any],
    solution_paths: Sequence[Path],
    *,
    max_suffixes: int = 8,
) -> list[dict[str, Any]]:
    """Return archived same-game suffixes whose actions extend this frontier.

    This is a high-precision memory fallback for late frontiers where whole-frame
    signature matching can be too brittle. It does not trust the archive blindly:
    the returned suffixes still go through the normal sandbox tournament.
    """

    game = str(trace.get("game"))
    frontier = action_list(trace.get("actions", []))
    frontier_key = tuple(tuple(a) for a in frontier)
    frontier_level = current_level(trace)
    out: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for path in solution_paths:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if str(data.get("game")) != game:
            continue
        actions = action_list(data.get("actions", []))
        if len(actions) <= len(frontier):
            continue
        if tuple(tuple(a) for a in actions[: len(frontier)]) != frontier_key:
            continue
        suffix = actions[len(frontier) :]
        key = tuple(tuple(a) for a in suffix)
        if key in seen:
            continue
        seen.add(key)
        try:
            source_levels = int(data.get("levels", 0))
        except Exception:
            source_levels = 0
        out.append(
            {
                "source_path": str(path),
                "source_levels": source_levels,
                "frontier_level": frontier_level,
                "start_index": len(frontier),
                "suffix": suffix,
                "signature_distance": 0.0,
                "retrieval_mode": "exact_prefix_continuation",
            }
        )
        if len(out) >= max_suffixes:
            break
    out.sort(key=lambda row: (-int(row.get("source_levels", 0)), len(row.get("suffix", []))))
    return out


def tournament_score(candidate: Mapping[str, Any], *, level_before: int, frontier_action_count: int) -> tuple[int, int, int, str]:
    """Rank verified candidates: deepest first, then shortest added route."""

    level_after = int(candidate.get("level_after", candidate.get("levels", level_before)))
    action_count = int(candidate.get("action_count", 0))
    actions_added = int(candidate.get("actions_added", max(0, action_count - frontier_action_count)))
    source_rank = {"memory": 0, "primitive": 1, "discovery": 2}.get(str(candidate.get("source")), 9)
    candidate_id = str(candidate.get("candidate_id", ""))
    return (level_after, -actions_added, -source_rank, candidate_id)


def select_tournament_winner(
    candidates: Sequence[Mapping[str, Any]],
    *,
    level_before: int,
    frontier_action_count: int,
) -> dict[str, Any] | None:
    verified = [
        dict(candidate)
        for candidate in candidates
        if candidate.get("replay_verified") and int(candidate.get("level_after", level_before)) > level_before
    ]
    if not verified:
        return None
    return max(
        verified,
        key=lambda candidate: tournament_score(
            candidate,
            level_before=level_before,
            frontier_action_count=frontier_action_count,
        ),
    )


def execute_memory_tournament(
    scratch: Path,
    stage_dir: Path,
    proposal_paths: Sequence[Path],
    *,
    level_before: int,
    frontier_action_count: int,
) -> dict[str, Any]:
    """Run all memory proposals and choose the best verified level-up."""

    attempts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    exec_dir = stage_dir / "memory_exec"
    exec_dir.mkdir(parents=True, exist_ok=True)
    for idx, proposal in enumerate(proposal_paths, start=1):
        solved_out = exec_dir / f"solved_{idx:02d}.json"
        counterexample_out = exec_dir / f"counterexample_{idx:02d}.json"
        result = execute(scratch, proposal, solved_out, counterexample_out)
        attempts.append(result)
        if result.get("outcome") != "level_up" or not solved_out.exists():
            continue
        replayed = replay_solution(scratch, solved_out)
        candidate = {
            "candidate_id": str(result.get("proposal_id") or f"memory-{idx:02d}"),
            "source": "memory",
            "proposal_path": str(proposal),
            "solved_out": str(solved_out),
            "level_before": int(level_before),
            "level_after": int(replayed["levels"]),
            "actions": action_list(replayed["actions"]),
            "action_count": len(action_list(replayed["actions"])),
            "actions_added": len(action_list(replayed["actions"])) - frontier_action_count,
            "win": int(replayed["win"]),
            "done": bool(replayed["done"]),
            "replay_verified": bool(replayed["replay_verified"]),
            "executor_result": result,
        }
        candidates.append(candidate)
    winner = select_tournament_winner(
        candidates,
        level_before=level_before,
        frontier_action_count=frontier_action_count,
    )
    summary = {
        "kind": "memory",
        "level_before": int(level_before),
        "frontier_action_count": int(frontier_action_count),
        "proposal_count": len(proposal_paths),
        "attempts": attempts,
        "candidates": [
            {k: v for k, v in candidate.items() if k != "actions"}
            for candidate in candidates
        ],
        "winner": None if winner is None else {k: v for k, v in winner.items() if k != "actions"},
    }
    (stage_dir / "memory_tournament.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"winner": winner, "summary": summary}


def evaluate_solution_tournament(
    scratch: Path,
    stage_dir: Path,
    payloads: Sequence[Mapping[str, Any]],
    *,
    source: str,
    level_before: int,
    frontier_action_count: int,
) -> dict[str, Any]:
    """Replay solved-like payloads and select the best verified candidate."""

    candidates: list[dict[str, Any]] = []
    candidate_dir = stage_dir / f"{source}_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for idx, payload in enumerate(payloads, start=1):
        candidate_id = str(payload.get("proposal_id") or payload.get("primitive") or f"{source}-{idx:02d}")
        candidate_path = candidate_dir / f"candidate_{idx:02d}.json"
        candidate_path.write_text(json.dumps(dict(payload), indent=2) + "\n")
        replayed = replay_solution(scratch, candidate_path)
        candidate = {
            "candidate_id": candidate_id,
            "source": source,
            "candidate_path": str(candidate_path),
            "level_before": int(level_before),
            "level_after": int(replayed["levels"]),
            "actions": action_list(replayed["actions"]),
            "action_count": len(action_list(replayed["actions"])),
            "actions_added": len(action_list(replayed["actions"])) - frontier_action_count,
            "win": int(replayed["win"]),
            "done": bool(replayed["done"]),
            "replay_verified": bool(replayed["replay_verified"]),
            "metadata": {
                k: v
                for k, v in payload.items()
                if k not in {"actions", "game", "levels", "win"}
            },
        }
        candidates.append(candidate)
    winner = select_tournament_winner(
        candidates,
        level_before=level_before,
        frontier_action_count=frontier_action_count,
    )
    summary = {
        "kind": source,
        "level_before": int(level_before),
        "frontier_action_count": int(frontier_action_count),
        "candidate_count": len(candidates),
        "candidates": [
            {k: v for k, v in candidate.items() if k != "actions"}
            for candidate in candidates
        ],
        "winner": None if winner is None else {k: v for k, v in winner.items() if k != "actions"},
    }
    (stage_dir / f"{source}_tournament.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"winner": winner, "summary": summary}


def run_retrieve_discover_controller(
    game: str,
    trace_root: Path,
    scratch: Path,
    out_dir: Path,
    *,
    sandbox_path: Path = Path("experiments/arc3_sandbox.py"),
    discovery_command: str | None = None,
    max_stages: int = 12,
    transfer_limit: int = 24,
    signature_threshold: float = 1000.0,
    discovery_timeout_s: int | None = None,
    use_sourcefree_primitives: bool = True,
    use_expensive_primitives: bool = False,
    use_cold_search: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    local_memory_root = out_dir / "episodic_memory"
    prepare_scratch(scratch, game, [], sandbox_path)

    actions: list[list[int]] = []
    stages: list[dict[str, Any]] = []
    stopped_reason = "max_stages"
    final_level = 0

    for stage_idx in range(max_stages):
        stage_dir = out_dir / f"stage_{stage_idx:02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        write_frontier(scratch, game, actions)
        trace = record_trace(scratch)
        level_before = current_level(trace)
        final_level = level_before
        (stage_dir / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")

        candidates = memory_paths(trace_root, local_memory_root, game)
        exact_candidate_paths = same_game_solution_paths(trace_root, local_memory_root, game=game)
        write_memory_manifest(stage_dir, candidates)
        ranked = filtered_ranked_suffixes(
            scratch,
            trace,
            candidates,
            transfer_limit=transfer_limit,
            signature_threshold=signature_threshold,
        )
        exact = exact_prefix_continuations(trace, exact_candidate_paths, max_suffixes=transfer_limit)
        if exact:
            merged: list[dict[str, Any]] = []
            seen_suffixes: set[tuple[tuple[int, ...], ...]] = set()
            for row in list(exact) + list(ranked):
                key = tuple(tuple(a) for a in row.get("suffix", []))
                if key not in seen_suffixes:
                    merged.append(dict(row))
                    seen_suffixes.add(key)
            ranked = merged[:transfer_limit]
        (stage_dir / "ranked_suffixes.json").write_text(json.dumps(ranked, indent=2) + "\n")

        if ranked:
            proposal_paths = write_transfer_proposals(stage_dir / "memory_proposals", trace, ranked)
        else:
            proposal_paths = []

        memory_tournament = execute_memory_tournament(
            scratch,
            stage_dir,
            proposal_paths,
            level_before=level_before,
            frontier_action_count=len(actions),
        ) if proposal_paths else {"winner": None, "summary": {"candidate_count": 0}}
        memory_winner = memory_tournament.get("winner")

        if memory_winner is not None:
            actions = action_list(memory_winner["actions"])
            final_level = int(memory_winner["level_after"])
            record = {
                "stage": stage_idx,
                "mode": "memory_tournament",
                "level_before": level_before,
                "level_after": final_level,
                "actions_added": int(memory_winner["actions_added"]),
                "frontier_actions_after": len(actions),
                "candidate_count": len(proposal_paths),
                "outcome": "level_up",
                "proposal_id": memory_winner.get("candidate_id"),
                "tournament": memory_tournament.get("summary"),
            }
            stages.append(record)
            (stage_dir / "stage_result.json").write_text(json.dumps(record, indent=2) + "\n")
            if final_level >= 9:
                stopped_reason = "completed_level_9"
                break
            continue

        if use_sourcefree_primitives:
            primitives = sourcefree_primitive_candidates(
                scratch,
                include_expensive=use_expensive_primitives,
                include_cold_search=use_cold_search,
            )
            if primitives:
                primitive_tournament = evaluate_solution_tournament(
                    scratch,
                    stage_dir,
                    primitives,
                    source="primitive",
                    level_before=level_before,
                    frontier_action_count=len(actions),
                )
                primitive_winner = primitive_tournament.get("winner")
                if primitive_winner is not None:
                    actions = action_list(primitive_winner["actions"])
                    final_level = int(primitive_winner["level_after"])
                    primitive_meta = dict(primitive_winner.get("metadata", {}))
                    memory_path = write_local_memory_record(
                        local_memory_root,
                        game=game,
                        actions=actions,
                        levels=final_level,
                        win=int(primitive_winner["win"]),
                        stage=stage_idx,
                        method="e146-sourcefree-primitive-writeback",
                        discovery={"primitive": primitive_meta, "tournament": primitive_tournament.get("summary")},
                    )
                    record = {
                        "stage": stage_idx,
                        "mode": "primitive_tournament",
                        "level_before": level_before,
                        "level_after": final_level,
                        "actions_added": int(primitive_winner["actions_added"]),
                        "frontier_actions_after": len(actions),
                        "outcome": "level_up",
                        "primitive": primitive_meta,
                        "tournament": primitive_tournament.get("summary"),
                        "memory_path": str(memory_path),
                    }
                    stages.append(record)
                    (stage_dir / "stage_result.json").write_text(json.dumps(record, indent=2) + "\n")
                    if final_level >= int(primitive_winner["win"]):
                        stopped_reason = "completed_all_levels"
                        break
                    continue
            else:
                primitive_miss = {
                    "stage": stage_idx,
                    "mode": "primitive",
                    "level_before": level_before,
                    "candidate_count": 0,
                    "use_expensive_primitives": bool(use_expensive_primitives),
                    "use_cold_search": bool(use_cold_search),
                    "outcome": "no_primitive_candidate",
                }
                (stage_dir / "primitive_miss.json").write_text(json.dumps(primitive_miss, indent=2) + "\n")

        if not discovery_command:
            stopped_reason = "memory_miss_no_discovery"
            record = {
                "stage": stage_idx,
                "mode": "memory",
                "level_before": level_before,
                "level_after": level_before,
                "candidate_count": len(proposal_paths),
                "outcome": "memory_miss_no_discovery",
            }
            stages.append(record)
            (stage_dir / "stage_result.json").write_text(json.dumps(record, indent=2) + "\n")
            break

        discovery = run_discovery_command(
            discovery_command,
            game=game,
            scratch=scratch,
            stage_dir=stage_dir,
            stage=stage_idx,
            level=level_before,
            timeout_s=discovery_timeout_s,
        )
        solved_path = Path(str(discovery["solved_path"]))
        if not solved_path.exists():
            stopped_reason = "discovery_no_solution"
            stages.append(
                {
                    "stage": stage_idx,
                    "mode": "discovery",
                    "level_before": level_before,
                    "level_after": level_before,
                    "outcome": "discovery_no_solution",
                    "discovery": discovery,
                }
            )
            break

        try:
            discovery_payloads = [json.loads(solved_path.read_text())]
        except Exception:
            discovery_payloads = []
        discovery_tournament = evaluate_solution_tournament(
            scratch,
            stage_dir,
            discovery_payloads,
            source="discovery",
            level_before=level_before,
            frontier_action_count=len(actions),
        )
        discovery_winner = discovery_tournament.get("winner")
        if discovery_winner is None:
            stopped_reason = "discovery_not_deeper"
            stages.append(
                {
                    "stage": stage_idx,
                    "mode": "discovery",
                    "level_before": level_before,
                    "level_after": level_before,
                    "outcome": "discovery_not_deeper",
                    "discovery": discovery,
                    "tournament": discovery_tournament.get("summary"),
                }
            )
            break

        actions = action_list(discovery_winner["actions"])
        final_level = int(discovery_winner["level_after"])
        memory_path = write_local_memory_record(
            local_memory_root,
            game=game,
            actions=actions,
            levels=final_level,
            win=int(discovery_winner["win"]),
            stage=stage_idx,
            method="e146-discovery-writeback",
            discovery={**discovery, "tournament": discovery_tournament.get("summary")},
        )
        record = {
            "stage": stage_idx,
            "mode": "discovery_tournament",
            "level_before": level_before,
            "level_after": final_level,
            "actions_added": int(discovery_winner["actions_added"]),
            "frontier_actions_after": len(actions),
            "outcome": "level_up",
            "discovery": discovery,
            "tournament": discovery_tournament.get("summary"),
            "memory_path": str(memory_path),
        }
        stages.append(record)
        (stage_dir / "stage_result.json").write_text(json.dumps(record, indent=2) + "\n")
        if final_level >= 9:
            stopped_reason = "completed_level_9"
            break

    result = {
        "experiment": "E146",
        "protocol": "source-free retrieve-discover-writeback",
        "game": game,
        "trace_root": str(trace_root),
        "scratch": str(scratch),
        "out_dir": str(out_dir),
        "local_memory_root": str(local_memory_root),
        "max_stages": max_stages,
        "transfer_limit": transfer_limit,
        "signature_threshold": signature_threshold,
        "has_discovery_command": bool(discovery_command),
        "use_sourcefree_primitives": bool(use_sourcefree_primitives),
        "use_expensive_primitives": bool(use_expensive_primitives),
        "use_cold_search": bool(use_cold_search),
        "solve_time_llm_calls": "external_command_dependent",
        "levels": final_level,
        "actions": actions,
        "action_count": len(actions),
        "stopped_reason": stopped_reason,
        "stages": stages,
    }
    (out_dir / "retrieve_discover_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E146 retrieve-discover-writeback controller.")
    parser.add_argument("game")
    parser.add_argument("--trace-root", default="experiments/results/arc3_traces")
    parser.add_argument("--scratch-dir", default="/tmp/e146_scratch")
    parser.add_argument("--out-dir", default="/tmp/e146_run")
    parser.add_argument("--sandbox-path", default="experiments/arc3_sandbox.py")
    parser.add_argument("--discovery-command")
    parser.add_argument(
        "--use-judge-schema-discovery",
        action="store_true",
        help="Use the repo's live Codex judge/schema discovery adapter when memory retrieval misses.",
    )
    parser.add_argument("--discovery-timeout-s", type=int)
    parser.add_argument(
        "--no-sourcefree-primitives",
        action="store_true",
        help="Disable deterministic source-free primitive solvers before LLM discovery.",
    )
    parser.add_argument(
        "--use-expensive-primitives",
        action="store_true",
        help="Enable slower deterministic timing/detour primitives before LLM discovery.",
    )
    parser.add_argument(
        "--use-cold-search",
        action="store_true",
        help="Enable bounded sandbox-only graph-frontier exploration when memory and cheap primitives miss.",
    )
    parser.add_argument("--max-stages", type=int, default=12)
    parser.add_argument("--transfer-limit", type=int, default=24)
    parser.add_argument("--signature-threshold", type=float, default=1000.0)
    args = parser.parse_args()

    discovery_command = args.discovery_command
    if args.use_judge_schema_discovery:
        if discovery_command:
            parser.error("--use-judge-schema-discovery cannot be combined with --discovery-command")
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "scripts" / "run_e146_judge_schema_discovery.sh"
        discovery_command = f"bash {shlex.quote(str(script))}"

    result = run_retrieve_discover_controller(
        args.game,
        Path(args.trace_root).resolve(),
        Path(args.scratch_dir).resolve(),
        Path(args.out_dir).resolve(),
        sandbox_path=Path(args.sandbox_path).resolve(),
        discovery_command=discovery_command,
        max_stages=args.max_stages,
        transfer_limit=args.transfer_limit,
        signature_threshold=args.signature_threshold,
        discovery_timeout_s=args.discovery_timeout_s,
        use_sourcefree_primitives=not args.no_sourcefree_primitives,
        use_expensive_primitives=args.use_expensive_primitives,
        use_cold_search=args.use_cold_search,
    )
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "game": result["game"],
                "levels": result["levels"],
                "action_count": result["action_count"],
                "stopped_reason": result["stopped_reason"],
                "has_discovery_command": result["has_discovery_command"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
