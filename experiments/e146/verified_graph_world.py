"""Verified finite graph worlds for E146 ARC reasoning.

The source-free primitives build useful visible graphs, but a visible edge is
only a hypothesis until the public sandbox reproduces it. This module turns a
frontier frame into a small finite OpenWorld-style transition system whose
edges are admitted only after replay verification through ``SandboxGame``.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from experiments.e143.transition_miner import action_list
from experiments.e146.sourcefree_primitives import (
    _act,
    _center_nodes,
    _frame_list,
    _fresh_verified_level,
    _largest_component_center,
    _reverse_action,
    _simple_paths,
)


Action = list[int]
Node = tuple[int, int]


def _utc_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _node_id(node: Node) -> str:
    return f"n{node[0]}_{node[1]}"


def _graph_for_pitch(
    nodes: Mapping[Node, int],
    *,
    pitch: int,
) -> dict[Node, list[tuple[Node, int]]]:
    graph: dict[Node, list[tuple[Node, int]]] = {p: [] for p in nodes}
    for x, y in nodes:
        for dx, dy, action in ((pitch, 0, 4), (-pitch, 0, 3), (0, pitch, 2), (0, -pitch, 1)):
            dest = (x + dx, y + dy)
            if dest in nodes:
                graph[(x, y)].append((dest, action))
    return graph


def _replay_to_frame(
    scratch: Path,
    game_id: str,
    actions: Sequence[Sequence[int]],
) -> dict[str, Any]:
    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    game = SandboxGame(game_id)
    try:
        game.reset()
        for action in action_list(actions):
            _act(game, action)
            if bool(game.done):
                break
        frame = _frame_list(game.frame)
        return {
            "frame": frame,
            "level": int(game.levels),
            "win": int(game.win),
            "done": bool(game.done),
            "player": _largest_component_center(frame, {4, 9}),
        }
    finally:
        game.close()


def _reachable_graph_paths(
    graph: Mapping[Node, Sequence[tuple[Node, int]]],
    start: Node,
    *,
    max_depth: int,
) -> dict[Node, list[Action]]:
    paths: dict[Node, list[Action]] = {start: []}
    queue: deque[Node] = deque([start])
    while queue:
        node = queue.popleft()
        path = paths[node]
        if len(path) >= max_depth:
            continue
        for dest, action in graph.get(node, []):
            if dest in paths:
                continue
            paths[dest] = path + [[action]]
            queue.append(dest)
    return paths


def _transition_code(table: Mapping[str, Mapping[str, Any]]) -> str:
    table_literal = repr(json.loads(json.dumps(table, sort_keys=True)))
    return f"""TRANSITIONS = {table_literal}

def transition(state: dict, action: dict) -> dict:
    name = str(action.get("name", ""))
    if name.startswith("a"):
        key = name[1:]
    else:
        key = name
    row = TRANSITIONS.get(str(state.get("node_id")), {{}})
    edge = row.get(key)
    nxt = dict(state)
    nxt["steps"] = int(nxt.get("steps", 0)) + 1
    nxt["last_action"] = key
    nxt["misses"] = int(nxt.get("misses", 0))
    if edge is None:
        nxt["misses"] += 1
        return nxt
    nxt["node_id"] = edge["to"]
    nxt["level"] = edge["level"]
    nxt["done"] = edge["done"]
    nxt["terminal"] = edge["terminal"]
    return nxt
"""


def openworld_spec_for_verified_graph(world: Mapping[str, Any]) -> dict[str, Any]:
    """Return a runnable OpenWorld JSON spec for the verified finite table."""

    table: dict[str, dict[str, Any]] = {}
    for edge in world.get("edges", []):
        if not isinstance(edge, Mapping) or not edge.get("verified"):
            continue
        table.setdefault(str(edge["from"]), {})[str(edge["action"])] = {
            "to": str(edge["to"]),
            "level": int(edge["to_level"]),
            "done": bool(edge["done"]),
            "terminal": bool(edge["terminal"]),
        }
    preview_nodes = []
    for node in world.get("nodes", []):
        if not isinstance(node, Mapping):
            continue
        xy = node.get("xy", ["?", "?"])
        preview_nodes.append(
            {
                "id": str(node["id"]),
                "label": [str(node["id"]), f"xy {xy[0]},{xy[1]}", f"c {node.get('color', '?')}"],
                "kind": "state",
                "initial": str(node["id"]) == str(world["initial_node_id"]),
            }
        )
    preview_edges = [
        {"src": str(edge["from"]), "dst": str(edge["to"]), "action": f"a{edge['action']}"}
        for edge in world.get("edges", [])
        if isinstance(edge, Mapping) and edge.get("verified")
    ]
    return {
        "openworld_spec_version": "1.0",
        "name": f"arc-{world.get('game')}-verified-graph-stage-{world.get('stage', 0):02d}",
        "description": (
            "A source-free finite graph world. Nodes are visible component centers; "
            "dynamics include only transitions replay-verified in SandboxGame."
        ),
        "state_schema": {
            "node_id": "str",
            "level": "int",
            "steps": "int",
            "misses": "int",
            "done": "bool",
            "terminal": "bool",
            "last_action": "str",
        },
        "initial_state": {
            "node_id": str(world["initial_node_id"]),
            "level": int(world["level"]),
            "steps": 0,
            "misses": 0,
            "done": False,
            "terminal": False,
            "last_action": "",
        },
        "actions": [f"a{a}" for a in world.get("actions", [1, 2, 3, 4])],
        "rules": [
            "Only replay-verified graph transitions are executable.",
            "Missing transitions increment misses and leave the symbolic node unchanged.",
            "Terminal transitions record level-up or game-done observations.",
        ],
        "transition": {
            "kind": "code",
            "func_name": "transition",
            "code": _transition_code(table),
        },
        "preview": {
            "graph": {
                "kind": "verified_arc_graph",
                "nodes": preview_nodes,
                "edges": preview_edges,
            }
        },
        "metadata": {
            "source_free": True,
            "verification": world.get("verification", {}),
            "verified_graph": {
                "nodes": world.get("nodes", []),
                "edges": world.get("edges", []),
            },
            "artifact": "experiments.e146.verified_graph_world",
        },
    }


def validate_openworld_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Best-effort OpenWorld validation/render check for a generated spec."""

    try:
        from openworld import render_card, validate_spec

        findings = validate_spec(dict(spec))
        rendered = bool(render_card(dict(spec)))
        return {"spec_valid": findings == [], "findings": findings, "card_renders": rendered}
    except Exception as ex:
        return {"spec_valid": False, "findings": [str(ex)[:500]], "card_renders": False}


def build_verified_graph_world(
    scratch: Path,
    *,
    stage: int = 0,
    pitch: int | None = None,
    max_depth: int = 18,
    colors: set[int] | None = None,
) -> dict[str, Any]:
    """Build and verify a finite graph world from ``scratch/frontier.json``."""

    colors = colors or {0, 2, 8, 9, 10, 11, 12, 13, 14, 15}
    frontier = json.loads((scratch / "frontier.json").read_text())
    game_id = str(frontier["game"])
    prefix = action_list(frontier.get("actions", []))
    base = _replay_to_frame(scratch, game_id, prefix)
    start_level = int(base["level"])
    start_player = base["player"]
    if start_player is None:
        raise ValueError("frontier frame has no visible player component")

    frame = base["frame"]
    nodes = _center_nodes(frame, colors)
    if start_player not in nodes:
        nodes[start_player] = 9
    pitches = [pitch] if pitch is not None else [6, 3]

    best: dict[str, Any] | None = None
    for candidate_pitch in pitches:
        graph = _graph_for_pitch(nodes, pitch=int(candidate_pitch))
        graph_paths = _reachable_graph_paths(graph, start_player, max_depth=max_depth)
        verified_paths: dict[Node, list[Action]] = {}
        for node, suffix in graph_paths.items():
            observed = _replay_to_frame(scratch, game_id, prefix + suffix)
            if observed["done"] and int(observed["level"]) <= start_level:
                continue
            if observed["player"] == node or int(observed["level"]) > start_level:
                verified_paths[node] = suffix

        edges: list[dict[str, Any]] = []
        for node, suffix in sorted(verified_paths.items()):
            for expected_dest, action in graph.get(node, []):
                observed = _replay_to_frame(scratch, game_id, prefix + suffix + [[action]])
                level = int(observed["level"])
                done = bool(observed["done"])
                player = observed["player"]
                level_up = level > start_level
                verified = level_up or (not done and level == start_level and player == expected_dest)
                if not verified:
                    continue
                to_node = expected_dest if player is None else player
                if to_node not in nodes:
                    nodes[to_node] = -1
                edges.append(
                    {
                        "from": _node_id(node),
                        "to": _node_id(to_node),
                        "action": int(action),
                        "from_xy": list(node),
                        "to_xy": list(to_node),
                        "expected_to_xy": list(expected_dest),
                        "from_level": start_level,
                        "to_level": level,
                        "done": done,
                        "terminal": bool(level_up or done),
                        "verified": True,
                    }
                )

        world = {
            "world_id": f"{game_id}__verified-graph__stage-{stage:02d}__{_utc_id()}",
            "game": game_id,
            "stage": int(stage),
            "level": start_level,
            "win": int(base["win"]),
            "pitch": int(candidate_pitch),
            "frontier_action_count": len(prefix),
            "initial_node_id": _node_id(start_player),
            "initial_xy": list(start_player),
            "actions": [1, 2, 3, 4],
            "nodes": [
                {"id": _node_id(node), "xy": list(node), "color": int(color)}
                for node, color in sorted(nodes.items())
                if node in verified_paths or any(_node_id(node) in (e["from"], e["to"]) for e in edges)
            ],
            "edges": edges,
            "verification": {
                "mode": "fresh SandboxGame replay per node/edge",
                "source_free": True,
                "verified_node_count": len(verified_paths),
                "verified_edge_count": len(edges),
                "max_depth": int(max_depth),
            },
        }
        if best is None or len(world["edges"]) > len(best["edges"]):
            best = world

    if best is None:
        raise RuntimeError("no candidate graph world was built")
    spec = openworld_spec_for_verified_graph(best)
    best["openworld_spec"] = spec
    best["openworld_validation"] = validate_openworld_spec(spec)
    return best


def write_verified_graph_world(world: Mapping[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    world_path = out_dir / "verified_graph_world.json"
    spec_path = out_dir / "verified_graph_world.spec.json"
    world_payload = dict(world)
    spec = world_payload.pop("openworld_spec")
    world_path.write_text(json.dumps(world_payload, indent=2) + "\n")
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    return {"world": str(world_path), "spec": str(spec_path)}


def _plan_nodes(
    graph: Mapping[Node, Sequence[tuple[Node, int]]],
    start: Node,
    actions: Sequence[int],
) -> list[Node] | None:
    node = start
    nodes = [node]
    for action in actions:
        matches = [dest for dest, edge_action in graph.get(node, []) if int(edge_action) == int(action)]
        if not matches:
            return None
        node = matches[0]
        nodes.append(node)
    return nodes


def _turn_indices(actions: Sequence[int]) -> set[int]:
    out: set[int] = set()
    for idx in range(1, len(actions)):
        if int(actions[idx]) != int(actions[idx - 1]):
            out.add(idx)
            out.add(idx + 1)
            out.add(max(1, idx - 1))
    return {idx for idx in out if 1 <= idx <= len(actions)}


def verified_world_symbolic_candidate(
    scratch: Path,
    *,
    stage: int = 0,
    max_depth: int = 32,
    max_paths: int = 128,
    max_candidates: int = 2000,
) -> dict[str, Any] | None:
    """Use a verified graph world to prioritize timing-loop candidates.

    The verified world supplies safe reversible edges. The visible graph supplies
    routes to goal markers. We insert bounded reverse/forward loops on those
    safe edges, then accept only candidates that fresh-replay to a level-up.
    """

    frontier = json.loads((scratch / "frontier.json").read_text())
    game_id = str(frontier["game"])
    prefix = action_list(frontier.get("actions", []))
    world = build_verified_graph_world(scratch, stage=stage, max_depth=18)
    start_level = int(world["level"])
    pitch = int(world["pitch"])

    base = _replay_to_frame(scratch, game_id, prefix)
    frame = base["frame"]
    nodes = _center_nodes(frame, {0, 2, 8, 9, 10, 11, 12, 13, 14, 15})
    starts = [p for p, color in nodes.items() if color == 9]
    goals = [p for p, color in nodes.items() if color == 14]
    if not starts or not goals:
        return None

    start = starts[0]
    graph = _graph_for_pitch(nodes, pitch=pitch)
    verified_loop_nodes: set[Node] = set()
    verified_edge_actions: set[tuple[Node, int]] = set()
    for edge in world.get("edges", []):
        if not isinstance(edge, Mapping) or not edge.get("verified"):
            continue
        node = tuple(int(v) for v in edge["from_xy"])
        verified_edge_actions.add((node, int(edge["action"])))
    for node, action in list(verified_edge_actions):
        dests = [dest for dest, edge_action in graph.get(node, []) if int(edge_action) == action]
        for dest in dests:
            if (dest, _reverse_action(action)) in verified_edge_actions:
                verified_loop_nodes.add(dest)
                verified_loop_nodes.add(node)
    if not verified_loop_nodes:
        return None

    plan_records: list[tuple[tuple[int, int, int], tuple[int, ...], list[Node]]] = []
    for goal in goals:
        for plan in _simple_paths(graph, start, goal, max_depth=max_depth, max_paths=max_paths):
            actions = tuple(int(a[0]) for a in plan)
            path_nodes = _plan_nodes(graph, start, actions)
            if path_nodes is None:
                continue
            marker_visits = sum(1 for p in path_nodes[1:-1] if nodes.get(p) not in (0, 2, 9, 14))
            turns = sum(1 for a, b in zip(actions, actions[1:]) if a != b)
            plan_records.append(((-marker_visits, len(actions), turns), actions, path_nodes))

    checked: set[tuple[int, ...]] = set()
    candidates_checked = 0
    repeat_pairs = ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1), (2, 2))
    for _, actions, path_nodes in sorted(plan_records, key=lambda item: item[0]):
        loop_sites: list[tuple[int, int]] = []
        priority = _turn_indices(actions)
        for idx in range(1, len(actions) + 1):
            node = path_nodes[idx]
            prev_action = int(actions[idx - 1])
            if node not in verified_loop_nodes:
                continue
            if (node, _reverse_action(prev_action)) not in verified_edge_actions:
                continue
            score = 0 if idx in priority else 1
            loop_sites.append((score, idx))
        loop_sites = sorted(loop_sites, key=lambda item: (item[0], item[1]))

        variants: list[tuple[int, ...]] = [actions]
        for _, idx in loop_sites[:16]:
            prev_action = int(actions[idx - 1])
            loop = (_reverse_action(prev_action), prev_action)
            for repeats in range(1, 6):
                variants.append(actions[:idx] + loop * repeats + actions[idx:])
        for left_i, (_, idx_a) in enumerate(loop_sites[:12]):
            for _, idx_b in loop_sites[left_i + 1:left_i + 9]:
                for repeats_a, repeats_b in repeat_pairs:
                    inserts = sorted(
                        (
                            (idx_a, (_reverse_action(actions[idx_a - 1]), int(actions[idx_a - 1])), repeats_a),
                            (idx_b, (_reverse_action(actions[idx_b - 1]), int(actions[idx_b - 1])), repeats_b),
                        ),
                        key=lambda item: item[0],
                        reverse=True,
                    )
                    candidate = actions
                    for idx, loop, repeats in inserts:
                        candidate = candidate[:idx] + loop * repeats + candidate[idx:]
                    variants.append(candidate)

        for suffix in variants:
            if suffix in checked:
                continue
            checked.add(suffix)
            candidates_checked += 1
            verified = _fresh_verified_level(
                scratch,
                game_id,
                prefix + [[a] for a in suffix],
                must_exceed=start_level,
            )
            if verified is not None:
                level, win = verified
                return {
                    "game": game_id,
                    "actions": prefix + [[a] for a in suffix],
                    "levels": level,
                    "win": win,
                    "primitive": "verified_world_symbolic_loop_search",
                    "world_id": world["world_id"],
                    "pitch": pitch,
                    "base_steps": len(actions),
                    "searched_depth": len(suffix),
                    "candidates_checked": candidates_checked,
                    "verified_edge_count": int(world["verification"]["verified_edge_count"]),
                }
            if candidates_checked >= max_candidates:
                return None
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True, help="Scratch dir containing frontier.json and arc3_sandbox.py")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--pitch", type=int, choices=[3, 6], default=None)
    parser.add_argument("--max-depth", type=int, default=18)
    args = parser.parse_args(argv)

    world = build_verified_graph_world(
        args.scratch,
        stage=args.stage,
        pitch=args.pitch,
        max_depth=args.max_depth,
    )
    paths = write_verified_graph_world(world, args.out_dir)
    print(json.dumps({"paths": paths, "verification": world["verification"], "validation": world["openworld_validation"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
