import json

from experiments.e144.full_run_controller import (
    current_level,
    load_solution_actions,
    write_frontier,
)


def test_current_level_reads_last_recorded_state():
    trace = {"states": [{"levels": 1}, {"levels": 4}]}

    assert current_level(trace) == 4


def test_write_frontier_normalizes_actions(tmp_path):
    write_frontier(tmp_path, "toy", [[1], [6, 3, 4], [99], [2]])

    frontier = json.loads((tmp_path / "frontier.json").read_text())
    assert frontier == {"game": "toy", "actions": [[1], [6, 3, 4], [2]]}


def test_load_solution_actions_normalizes_actions(tmp_path):
    path = tmp_path / "solved.json"
    path.write_text(json.dumps({"actions": [[1], [6, 1, 2], [6, 99, 0], [5]]}))

    assert load_solution_actions(path) == [[1], [6, 1, 2], [5]]

