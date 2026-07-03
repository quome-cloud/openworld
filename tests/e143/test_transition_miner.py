from experiments.e143.transition_miner import (
    build_introspection_world,
    build_transition_graph,
    extract_behavioral_suffixes,
    frame_summary,
    is_critical,
    rank_behavioral_suffixes_by_signature,
    signature_distance,
    transition_delta,
)
import json


def test_frame_summary_extracts_palette_and_small_components():
    frame = [
        [0, 0, 1, 1],
        [0, 2, 2, 1],
        [3, 3, 2, 1],
        [3, 0, 0, 0],
    ]

    summary = frame_summary(frame, levels=2, win=1)

    assert summary["levels"] == 2
    assert summary["win"] == 1
    assert summary["palette"] == {"0": 6, "1": 4, "2": 3, "3": 3}
    assert any(c["c"] == 2 and c["n"] == 3 for c in summary["small"])


def test_transition_delta_marks_palette_and_component_changes_critical():
    before = frame_summary([[0, 0], [1, 1]], levels=0)
    after = frame_summary([[0, 2], [1, 1]], levels=0)

    delta = transition_delta(before, after)

    assert delta["hash_changed"] is True
    assert delta["palette_delta"] == {"0": -1, "2": 1}
    assert is_critical(delta)


def test_introspection_world_promotes_critical_edges():
    s0 = frame_summary([[0, 0], [1, 1]], levels=0)
    s1 = frame_summary([[0, 2], [1, 1]], levels=0)
    delta = transition_delta(s0, s1)
    trace = {
        "game": "toy",
        "proposal_id": "p",
        "states": [s0, s1],
        "edges": [
            {
                "edge_id": "trace-0000",
                "step": 0,
                "action": [6, 1, 0],
                "from": s0["hash"],
                "to": s1["hash"],
                "delta": delta,
                "critical": True,
            }
        ],
    }

    graph = build_transition_graph(trace)
    world = build_introspection_world(graph)

    assert graph["critical_edges"][0]["edge_id"] == "trace-0000"
    assert world["world_id"] == "transition-introspection-toy"
    assert world["cards"][0]["observed_delta"]["palette_delta"]["2"] == 1


def test_behavioral_suffix_extractor_includes_level_local_135_cut(tmp_path):
    actions = [[1] for _ in range(313)]
    solution = tmp_path / "toy_solution.json"
    solution.write_text(json.dumps({"game": "toy", "levels": 6, "actions": actions}))
    trace = {
        "game": "toy",
        "states": [{"levels": 5}],
    }

    suffixes = extract_behavioral_suffixes(trace, [solution], max_suffixes=20)
    starts = {s["start_index"] for s in suffixes}

    assert 178 in starts


def test_behavioral_suffix_extractor_includes_frontier_aligned_cut(tmp_path):
    actions = [[1] for _ in range(105)]
    solution = tmp_path / "su15_solution.json"
    solution.write_text(json.dumps({"game": "su15", "levels": 9, "actions": actions}))
    trace = {
        "game": "su15",
        "frontier_action_count": 85,
        "states": [{"levels": 4}],
    }

    suffixes = extract_behavioral_suffixes(trace, [solution], max_suffixes=20)
    starts = {s["start_index"] for s in suffixes}

    assert 85 in starts


def test_signature_distance_prefers_matching_state():
    target = frame_summary([[0, 0], [1, 2]], levels=3)
    same = frame_summary([[0, 0], [1, 2]], levels=3)
    different = frame_summary([[4, 4], [4, 2]], levels=3)
    wrong_level = frame_summary([[0, 0], [1, 2]], levels=2)

    assert signature_distance(target, same) < signature_distance(target, different)
    assert signature_distance(target, wrong_level) > signature_distance(target, different)


def test_signature_ranking_prefers_longer_suffix_in_same_distance_band(monkeypatch, tmp_path):
    solution = tmp_path / "toy_solution.json"
    actions = [[1] for _ in range(10)]
    solution.write_text(json.dumps({"game": "toy", "actions": actions}))
    trace = {"game": "toy", "states": [frame_summary([[0]], levels=1)]}
    suffixes = [
        {"source_path": str(solution), "start_index": 3, "suffix": [[1]] * 7},
        {"source_path": str(solution), "start_index": 8, "suffix": [[1]] * 2},
    ]

    class FakeGame:
        def __init__(self, _game):
            self.levels = 1
            self.win = 0
            self.done = False
            self.frame = [[0]]

        def reset(self):
            self.frame = [[0]]

        def step(self, *_args):
            self.frame = [[0]]

        def close(self):
            pass


    monkeypatch.setitem(__import__("sys").modules, "arc3_sandbox", type("M", (), {"SandboxGame": FakeGame}))
    ranked = rank_behavioral_suffixes_by_signature(tmp_path, trace, suffixes)

    assert ranked[0]["start_index"] == 3
