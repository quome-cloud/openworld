import json
import time

from experiments.e145.sourcefree_controller import eligible_solution_paths
from experiments.e146.retrieve_discover_controller import (
    discovery_environment,
    memory_paths,
    run_discovery_command,
    select_tournament_winner,
    tournament_score,
    write_local_memory_record,
)
from experiments.e146 import sourcefree_primitives


def test_write_local_memory_record_is_eligible_for_retrieval(tmp_path):
    path = write_local_memory_record(
        tmp_path,
        game="toy",
        actions=[[1], [6, 2, 3]],
        levels=1,
        win=3,
        stage=0,
        method="unit",
    )

    assert path.exists()
    assert eligible_solution_paths(tmp_path, games=["toy"]) == [path]

    record = json.loads((tmp_path / "runs.jsonl").read_text().splitlines()[0])
    assert record["source_free"] is True
    assert record["outcome"]["replay_verified"] is True
    assert record["outcome"]["audit"]["clean"] is True


def test_memory_paths_merges_archive_and_local_without_duplicates(tmp_path):
    archive = tmp_path / "archive"
    local = tmp_path / "local"
    write_local_memory_record(archive, game="toy", actions=[[1]], levels=1, win=2, stage=0, method="archive")
    local_path = write_local_memory_record(local, game="toy", actions=[[2]], levels=1, win=2, stage=0, method="local")

    paths = memory_paths(archive, local, "toy")

    assert local_path in paths
    assert len(paths) == 2


def test_discovery_environment_points_to_frontier_and_solved_out(tmp_path):
    scratch = tmp_path / "scratch"
    stage_dir = tmp_path / "stage"
    env = discovery_environment(game="toy", scratch=scratch, stage_dir=stage_dir, stage=2, level=5)

    assert env["E146_GAME"] == "toy"
    assert env["E146_STAGE"] == "2"
    assert env["E146_LEVEL"] == "5"
    assert env["E146_FRONTIER"] == str(scratch / "frontier.json")
    assert env["E146_SOLVED_OUT"] == str(stage_dir / "discovery" / "solved.json")


def test_discovery_command_harvests_solved_before_process_exit(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    command = (
        "mkdir -p judge_schema_discovery; "
        "printf '%s' '{{\"game\":\"toy\",\"actions\":[[1]],\"levels\":1,\"win\":1}}' "
        "> judge_schema_discovery/solved.json; "
        "sleep 30"
    )

    started = time.monotonic()
    result = run_discovery_command(
        command,
        game="toy",
        scratch=scratch,
        stage_dir=stage_dir,
        stage=0,
        level=0,
        timeout_s=10,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 10
    assert result["harvested"] is True
    solved = json.loads((stage_dir / "discovery" / "solved.json").read_text())
    assert solved["levels"] == 1


def test_tournament_selects_deepest_verified_candidate():
    candidates = [
        {
            "candidate_id": "short",
            "source": "memory",
            "level_after": 2,
            "action_count": 12,
            "actions_added": 4,
            "replay_verified": True,
        },
        {
            "candidate_id": "deep",
            "source": "primitive",
            "level_after": 3,
            "action_count": 20,
            "actions_added": 12,
            "replay_verified": True,
        },
        {
            "candidate_id": "unverified",
            "source": "memory",
            "level_after": 4,
            "action_count": 8,
            "actions_added": 1,
            "replay_verified": False,
        },
    ]

    winner = select_tournament_winner(candidates, level_before=1, frontier_action_count=8)

    assert winner is not None
    assert winner["candidate_id"] == "deep"


def test_tournament_tie_breaks_on_shorter_verified_candidate():
    long = {
        "candidate_id": "long",
        "source": "memory",
        "level_after": 2,
        "action_count": 20,
        "actions_added": 12,
        "replay_verified": True,
    }
    short = {
        "candidate_id": "short",
        "source": "primitive",
        "level_after": 2,
        "action_count": 11,
        "actions_added": 3,
        "replay_verified": True,
    }

    assert tournament_score(short, level_before=1, frontier_action_count=8) > tournament_score(
        long,
        level_before=1,
        frontier_action_count=8,
    )
    winner = select_tournament_winner([long, short], level_before=1, frontier_action_count=8)

    assert winner is not None
    assert winner["candidate_id"] == "short"


def test_sourcefree_primitives_do_not_run_expensive_detour_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(sourcefree_primitives, "center_corridor_candidate", lambda scratch: None)
    monkeypatch.setattr(sourcefree_primitives, "lattice_corridor_candidate", lambda scratch: None)

    def fail_if_called(_scratch):
        raise AssertionError("expensive detour primitive should be opt-in")

    monkeypatch.setattr(sourcefree_primitives, "simple_path_detour_candidate", fail_if_called)

    assert sourcefree_primitives.sourcefree_primitive_candidates(tmp_path) == []
