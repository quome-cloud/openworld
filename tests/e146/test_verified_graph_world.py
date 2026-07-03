import json

from openworld import Action, from_spec, validate_spec

from experiments.e146.verified_graph_world import (
    openworld_spec_for_verified_graph,
    write_verified_graph_world,
)


def _toy_world():
    return {
        "world_id": "toy__verified-graph__stage-00",
        "game": "toy",
        "stage": 0,
        "level": 2,
        "win": 3,
        "pitch": 6,
        "frontier_action_count": 5,
        "initial_node_id": "n0_0",
        "initial_xy": [0, 0],
        "actions": [1, 2, 3, 4],
        "nodes": [
            {"id": "n0_0", "xy": [0, 0], "color": 9},
            {"id": "n6_0", "xy": [6, 0], "color": 2},
            {"id": "n12_0", "xy": [12, 0], "color": 14},
        ],
        "edges": [
            {
                "from": "n0_0",
                "to": "n6_0",
                "action": 4,
                "from_xy": [0, 0],
                "to_xy": [6, 0],
                "expected_to_xy": [6, 0],
                "from_level": 2,
                "to_level": 2,
                "done": False,
                "terminal": False,
                "verified": True,
            },
            {
                "from": "n6_0",
                "to": "n12_0",
                "action": 4,
                "from_xy": [6, 0],
                "to_xy": [12, 0],
                "expected_to_xy": [12, 0],
                "from_level": 2,
                "to_level": 3,
                "done": False,
                "terminal": True,
                "verified": True,
            },
            {
                "from": "n0_0",
                "to": "n0_6",
                "action": 2,
                "from_xy": [0, 0],
                "to_xy": [0, 6],
                "expected_to_xy": [0, 6],
                "from_level": 2,
                "to_level": 2,
                "done": False,
                "terminal": False,
                "verified": False,
            },
        ],
        "verification": {
            "mode": "unit",
            "source_free": True,
            "verified_node_count": 3,
            "verified_edge_count": 2,
            "max_depth": 4,
        },
    }


def test_openworld_spec_for_verified_graph_is_valid_and_runnable():
    spec = openworld_spec_for_verified_graph(_toy_world())

    assert validate_spec(spec) == []

    world = from_spec(spec, allow_code=True)
    state = world.step(Action("a4"))
    assert state["node_id"] == "n6_0"
    assert state["level"] == 2
    assert state["misses"] == 0

    state = world.step(Action("a4"))
    assert state["node_id"] == "n12_0"
    assert state["level"] == 3
    assert state["terminal"] is True


def test_unverified_edges_are_not_executable():
    spec = openworld_spec_for_verified_graph(_toy_world())
    world = from_spec(spec, allow_code=True)

    state = world.step(Action("a2"))

    assert state["node_id"] == "n0_0"
    assert state["misses"] == 1


def test_write_verified_graph_world_splits_world_and_spec(tmp_path):
    payload = _toy_world()
    payload["openworld_spec"] = openworld_spec_for_verified_graph(payload)

    paths = write_verified_graph_world(payload, tmp_path)

    world_json = json.loads((tmp_path / "verified_graph_world.json").read_text())
    spec_json = json.loads((tmp_path / "verified_graph_world.spec.json").read_text())
    assert paths["world"].endswith("verified_graph_world.json")
    assert "openworld_spec" not in world_json
    assert spec_json["name"] == "arc-toy-verified-graph-stage-00"
