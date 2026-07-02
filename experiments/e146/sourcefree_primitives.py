"""Deterministic source-free ARC solver primitives for E146.

These helpers operate only on public ``SandboxGame`` observations. They do not
import game source or use private state; they parse the rendered frame and
return executable action candidates for the controller to replay-verify.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Sequence

from experiments.e143.transition_miner import action_list


@dataclass(frozen=True)
class MacroScore:
    level_delta: int
    alive: int
    novelty: int
    object_progress: int
    activation_delta: int
    reversible_control: int
    cost: int

    def rank(self) -> tuple[int, int, int, int, int, int, int]:
        return (
            self.level_delta,
            self.alive,
            self.novelty,
            self.object_progress,
            self.activation_delta,
            self.reversible_control,
            -self.cost,
        )

    def compose(self, other: "MacroScore") -> "MacroScore":
        return MacroScore(
            level_delta=self.level_delta + other.level_delta,
            alive=min(self.alive, other.alive),
            novelty=self.novelty + other.novelty,
            object_progress=self.object_progress + other.object_progress,
            activation_delta=self.activation_delta + other.activation_delta,
            reversible_control=self.reversible_control + other.reversible_control,
            cost=self.cost + other.cost,
        )


ZERO_SCORE = MacroScore(-9999, 0, -9999, -9999, -9999, -9999, 10**9)
ONE_SCORE = MacroScore(0, 1, 0, 0, 0, 0, 0)


def _act(game: Any, action: Sequence[int]) -> None:
    if int(action[0]) == 6:
        game.step(6, int(action[1]), int(action[2]))
    else:
        game.step(int(action[0]))


def _frame_list(frame: Any) -> list[list[int]]:
    return frame.tolist() if hasattr(frame, "tolist") else frame


def _color_components_summary(frame: list[list[int]], colors: set[int]) -> tuple[tuple[int, int, int, int], ...]:
    rows: list[tuple[int, int, int, int]] = []
    for color in sorted(colors):
        for cells in _component_cells(frame, color):
            xs = [p[0] for p in cells]
            ys = [p[1] for p in cells]
            rows.append((color, round(sum(xs) / len(xs)), round(sum(ys) / len(ys)), len(cells)))
    return tuple(sorted(rows))


def _frame_digest(frame: list[list[int]]) -> str:
    h = hashlib.sha1()
    for row in frame:
        h.update(bytes(int(v) & 0xFF for v in row))
        h.update(b"\n")
    return h.hexdigest()


def _dominant_color(frame: list[list[int]]) -> int:
    counts: dict[int, int] = {}
    for row in frame:
        for value in row:
            color = int(value)
            counts[color] = counts.get(color, 0) + 1
    return max(counts, key=counts.get) if counts else 0


def _world_modality(
    frame: list[list[int]],
    *,
    available_actions: Sequence[int] | None = None,
) -> str:
    """Route a frame to a compact state representation family."""

    actions = {int(a) for a in (available_actions or [])}
    components = _component_records(frame)
    small = sum(1 for component in components if int(component["n"]) <= 24 and int(component["c"]) != 0)
    player = _player_center(frame)
    if actions == {6}:
        return "click_layout"
    if player is not None and {1, 2, 3, 4} & actions:
        return "navigation"
    if len(components) > 90 or small > 48:
        return "dense_grid"
    return "sparse_objects"


def _component_lens(
    frame: list[list[int]],
    *,
    exact: bool,
    limit: int,
) -> tuple[tuple[int, int, int, int], ...]:
    rows = []
    for component in _component_records(frame):
        color = int(component["c"])
        size = int(component["n"])
        if color == 0:
            continue
        x = int(component["x"])
        y = int(component["y"])
        if exact:
            rows.append((color, min(size, 256), x, y))
        elif color in (4, 8, 9, 10, 11, 12, 13, 14, 15) or size <= 24:
            rows.append((color, min(size, 64), x // 2, y // 2))
    return tuple(sorted(rows)[:limit])


def _row_col_lens(objects: Sequence[tuple[int, int, int, int]]) -> dict[str, tuple[tuple[int, ...], ...]]:
    rows: dict[int, list[tuple[int, int]]] = {}
    cols: dict[int, list[tuple[int, int]]] = {}
    for color, _size, x, y in objects:
        rows.setdefault(int(y), []).append((int(x), int(color)))
        cols.setdefault(int(x), []).append((int(y), int(color)))
    row_sig = tuple((y, tuple(color for _x, color in sorted(values))) for y, values in sorted(rows.items()))
    col_sig = tuple((x, tuple(color for _y, color in sorted(values))) for x, values in sorted(cols.items()))
    return {"rows": row_sig[:16], "cols": col_sig[:16]}


def _region_lens(frame: list[list[int]], *, grid: int = 8) -> tuple[tuple[int, ...], ...]:
    bg = _dominant_color(frame)
    h = len(frame)
    w = len(frame[0]) if h else 0
    out: list[tuple[int, ...]] = []
    for gy in range(grid):
        row: list[int] = []
        y0 = int(gy * h / grid)
        y1 = int((gy + 1) * h / grid)
        for gx in range(grid):
            x0 = int(gx * w / grid)
            x1 = int((gx + 1) * w / grid)
            counts: dict[int, int] = {}
            for yy in range(y0, max(y0 + 1, y1)):
                for xx in range(x0, max(x0 + 1, x1)):
                    if yy >= h or xx >= w:
                        continue
                    color = int(frame[yy][xx])
                    if color == bg:
                        continue
                    counts[color] = counts.get(color, 0) + 1
            row.append(max(counts, key=counts.get) if counts else bg)
        out.append(tuple(row))
    return tuple(out)


def _composite_world_key(
    frame: list[list[int]],
    *,
    modality: str | None = None,
    available_actions: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Small source-free world key composed from routed public-frame lenses."""

    components = _component_records(frame)
    palette: dict[int, int] = {}
    for row in frame:
        for value in row:
            color = int(value)
            palette[color] = palette.get(color, 0) + 1
    player = _player_center(frame)
    goal = _rare_goal_center(frame, player)
    distance = -1 if player is None or goal is None else abs(player[0] - goal[0]) + abs(player[1] - goal[1])
    mode = modality or _world_modality(frame, available_actions=available_actions)
    occupancy: dict[tuple[int, int, int], int] = {}
    for component in components:
        color = int(component["c"])
        size = int(component["n"])
        x = int(component["x"])
        y = int(component["y"])
        if color == 0:
            continue
        bin_key = (color, min(7, x // 8), min(7, y // 8))
        occupancy[bin_key] = occupancy.get(bin_key, 0) + 1
    palette_lens = tuple(sorted((color, min(count // 8, 255)) for color, count in palette.items() if color != 0))
    occupancy_lens = tuple(sorted((color, bx, by, min(count, 9)) for (color, bx, by), count in occupancy.items()))
    key: dict[str, Any] = {
        "mode": mode,
        "player": player,
        "goal": goal,
        "distance": distance,
        "active": _small_activation_count(frame),
        "palette": palette_lens,
    }
    if mode == "click_layout":
        exact_objects = _component_lens(frame, exact=True, limit=80)
        key["objects"] = exact_objects
        key["layout"] = _row_col_lens(exact_objects)
        key["regions"] = _region_lens(frame, grid=8)
    elif mode == "navigation":
        key["objects"] = _component_lens(frame, exact=False, limit=48)
        key["occupancy"] = occupancy_lens[:64]
        key["regions"] = _region_lens(frame, grid=8)
    elif mode == "dense_grid":
        key["regions"] = _region_lens(frame, grid=8)
        key["occupancy"] = occupancy_lens[:48]
    else:
        key["objects"] = _component_lens(frame, exact=True, limit=64)
        key["occupancy"] = occupancy_lens[:64]
    return key


def _salient_colors_present(frame: list[list[int]], colors: set[int]) -> set[int]:
    return {int(v) for row in frame for v in row if int(v) in colors}


def _largest_component_center(frame: list[list[int]], colors: set[int]) -> tuple[int, int] | None:
    best: tuple[int, int, int] | None = None
    for color in colors:
        for cells in _component_cells(frame, color):
            xs = [p[0] for p in cells]
            ys = [p[1] for p in cells]
            center = (round(sum(xs) / len(xs)), round(sum(ys) / len(ys)))
            item = (len(cells), center[0], center[1])
            if best is None or item[0] > best[0]:
                best = item
    if best is None:
        return None
    return best[1], best[2]


def _closest_component_center(
    frame: list[list[int]],
    colors: set[int],
    origin: tuple[int, int] | None,
) -> tuple[int, int] | None:
    centers: list[tuple[int, int]] = []
    for color in colors:
        for cells in _component_cells(frame, color):
            xs = [p[0] for p in cells]
            ys = [p[1] for p in cells]
            centers.append((round(sum(xs) / len(xs)), round(sum(ys) / len(ys))))
    if not centers:
        return None
    if origin is None:
        return centers[0]
    return min(centers, key=lambda p: abs(p[0] - origin[0]) + abs(p[1] - origin[1]))


def _has_component(frame: list[list[int]], color: int, *, min_size: int = 1) -> bool:
    return any(len(cells) >= min_size for cells in _component_cells(frame, color))


def _replay_actions(game: Any, actions: Sequence[Sequence[int]]) -> tuple[int, int, bool]:
    game.reset()
    for action in actions:
        _act(game, action)
        if bool(game.done):
            break
    return int(game.levels), int(game.win), bool(game.done)


def _component_cells(frame: list[list[int]], color: int) -> list[list[tuple[int, int]]]:
    h = len(frame)
    w = len(frame[0]) if h else 0
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if (x, y) in seen or int(frame[y][x]) != color:
                continue
            seen.add((x, y))
            q: deque[tuple[int, int]] = deque([(x, y)])
            cells: list[tuple[int, int]] = []
            while q:
                cx, cy = q.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if (
                        0 <= nx < w
                        and 0 <= ny < h
                        and (nx, ny) not in seen
                        and int(frame[ny][nx]) == color
                    ):
                        seen.add((nx, ny))
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    cells = list(cells)
    xs = [p[0] for p in cells]
    ys = [p[1] for p in cells]
    return min(xs), min(ys), max(xs), max(ys)


def _marker_top_lefts(frame: list[list[int]], color: int) -> list[tuple[int, int]]:
    markers: list[tuple[int, int, int]] = []
    for cells in _component_cells(frame, color):
        min_x, min_y, max_x, max_y = _bbox(cells)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        if 2 <= width <= 4 and 2 <= height <= 4 and len(cells) >= 4:
            markers.append((len(cells), min_x, min_y))
    markers.sort(key=lambda m: (-m[0], m[2], m[1]))
    return [(m[1], m[2]) for m in markers]


def _marker_top_left(frame: list[list[int]], color: int) -> tuple[int, int] | None:
    markers = _marker_top_lefts(frame, color)
    return markers[0] if markers else None


def _goal_marker_candidates(frame: list[list[int]], start_px: tuple[int, int]) -> list[tuple[int, int, int]]:
    priority = [14, 8, 10, 11, 12, 13, 15, 7, 3, 1]
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int, int]] = []
    for color in priority + [c for c in range(16) if c not in priority]:
        if color in (0, 2, 4, 5, 6, 9):
            continue
        for marker in _marker_top_lefts(frame, color):
            if marker == start_px or marker in seen:
                continue
            seen.add(marker)
            out.append((color, marker[0], marker[1]))
    return out


def _candidate_origins(start: tuple[int, int], goal: tuple[int, int], pitch: int, size: int) -> list[tuple[int, int]]:
    origins: list[tuple[int, int]] = []
    for ox in range(0, pitch):
        if (start[0] - ox) % pitch != 0 or (goal[0] - ox) % pitch != 0:
            continue
        for oy in range(0, pitch):
            if (start[1] - oy) % pitch == 0 and (goal[1] - oy) % pitch == 0:
                origins.append((ox, oy))
    # Prefer origins that keep both markers in a compact non-negative lattice.
    return sorted(
        origins,
        key=lambda o: (
            abs((start[0] - o[0]) // pitch) + abs((start[1] - o[1]) // pitch),
            abs(o[0] - start[0]) + abs(o[1] - start[1]),
            size,
        ),
    )


def _corridor_graph(
    frame: list[list[int]],
    *,
    origin: tuple[int, int],
    pitch: int,
    size: int,
    connector_color: int,
) -> dict[tuple[int, int], list[tuple[tuple[int, int], int]]]:
    h = len(frame)
    w = len(frame[0]) if h else 0
    ox, oy = origin
    max_x = (w - ox - size) // pitch
    max_y = (h - oy - size) // pitch
    if max_x < 0 or max_y < 0:
        return {}

    graph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {}
    for gy in range(max_y + 1):
        for gx in range(max_x + 1):
            x = ox + gx * pitch
            y = oy + gy * pitch
            graph[(gx, gy)] = []
            if gx < max_x:
                seg = [frame[sy][sx] for sy in range(y, min(y + size, h)) for sx in range(x + size, min(x + pitch, w))]
                if sum(int(v) == connector_color for v in seg) >= max(1, size):
                    graph[(gx, gy)].append(((gx + 1, gy), 4))
            if gx > 0:
                seg = [frame[sy][sx] for sy in range(y, min(y + size, h)) for sx in range(max(0, x - (pitch - size)), x)]
                if sum(int(v) == connector_color for v in seg) >= max(1, size):
                    graph[(gx, gy)].append(((gx - 1, gy), 3))
            if gy < max_y:
                seg = [frame[sy][sx] for sy in range(y + size, min(y + pitch, h)) for sx in range(x, min(x + size, w))]
                if sum(int(v) == connector_color for v in seg) >= max(1, size):
                    graph[(gx, gy)].append(((gx, gy + 1), 2))
            if gy > 0:
                seg = [frame[sy][sx] for sy in range(max(0, y - (pitch - size)), y) for sx in range(x, min(x + size, w))]
                if sum(int(v) == connector_color for v in seg) >= max(1, size):
                    graph[(gx, gy)].append(((gx, gy - 1), 1))
    return graph


def _bfs_actions(
    graph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    blocked: set[tuple[int, int]] | None = None,
) -> list[list[int]] | None:
    blocked = blocked or set()
    q: deque[tuple[tuple[int, int], list[list[int]]]] = deque([(start, [])])
    seen = {start} | (blocked - {goal})
    while q:
        node, path = q.popleft()
        if node == goal:
            return path
        for nxt, action in graph.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, path + [[action]]))
    return None


def _simple_paths(
    graph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]],
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    max_depth: int = 36,
    max_paths: int = 64,
) -> list[list[list[int]]]:
    out: list[list[list[int]]] = []
    q: deque[tuple[tuple[int, int], list[list[int]], set[tuple[int, int]]]] = deque([(start, [], {start})])
    while q and len(out) < max_paths:
        node, path, seen = q.popleft()
        if node == goal:
            out.append(path)
            continue
        if len(path) >= max_depth:
            continue
        for nxt, action in graph.get(node, []):
            if nxt not in seen:
                q.append((nxt, path + [[action]], seen | {nxt}))
    return sorted(out, key=len)


def _fresh_verified_level(
    scratch: Path,
    game_id: str,
    actions: Sequence[Sequence[int]],
    *,
    must_exceed: int,
) -> tuple[int, int] | None:
    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    game = SandboxGame(game_id)
    try:
        level, win, _ = _replay_actions(game, actions)
        if level > must_exceed:
            return level, win
        return None
    finally:
        game.close()


def lattice_corridor_candidate(scratch: Path, *, max_steps: int = 96) -> dict[str, Any] | None:
    """Return a deeper solved candidate for visible 3x3-node corridor mazes.

    This targets the pattern seen in ``tu93``: a 3x3 start marker, a 3x3 goal
    marker, and color-2 corridor segments between pitched grid nodes.
    """

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    game = SandboxGame(frontier["game"])
    try:
        game.reset()
        for action in prefix:
            _act(game, action)
            if bool(game.done):
                return None
        start_level = int(game.levels)
        frame = _frame_list(game.frame)
        start_px = _marker_top_left(frame, 9)
        if start_px is None:
            return None

        for goal_color, goal_x, goal_y in _goal_marker_candidates(frame, start_px):
            goal_px = (goal_x, goal_y)
            for pitch, size in ((6, 3), (5, 3), (4, 2), (8, 3)):
                for origin in _candidate_origins(start_px, goal_px, pitch, size):
                    start = ((start_px[0] - origin[0]) // pitch, (start_px[1] - origin[1]) // pitch)
                    goal = ((goal_px[0] - origin[0]) // pitch, (goal_px[1] - origin[1]) // pitch)
                    graph = _corridor_graph(frame, origin=origin, pitch=pitch, size=size, connector_color=2)
                    plans = _simple_paths(graph, start, goal, max_depth=max_steps, max_paths=96)
                    for plan in plans:
                        game.reset()
                        for action in prefix:
                            _act(game, action)
                        used: list[list[int]] = []
                        for action in plan:
                            _act(game, action)
                            used.append(action)
                            if bool(game.done) and int(game.levels) <= start_level:
                                break
                            if int(game.levels) > start_level:
                                all_actions = prefix + used
                                verified = _fresh_verified_level(
                                    scratch,
                                    frontier["game"],
                                    all_actions,
                                    must_exceed=start_level,
                                )
                                if verified is None:
                                    break
                                level, win = verified
                                return {
                                    "game": frontier["game"],
                                    "actions": all_actions,
                                    "levels": level,
                                    "win": win,
                                    "primitive": "lattice_corridor_paths",
                                    "origin": list(origin),
                                    "pitch": pitch,
                                    "size": size,
                                    "start": list(start),
                                    "goal": list(goal),
                                    "goal_color": goal_color,
                                    "planned_steps": len(plan),
                                    "executed_steps": len(used),
                                }
                        if used == plan:
                            all_actions = prefix + used
                            verified = _fresh_verified_level(
                                scratch,
                                frontier["game"],
                                all_actions,
                                must_exceed=start_level,
                            )
                            if verified is not None:
                                level, win = verified
                                return {
                                    "game": frontier["game"],
                                    "actions": all_actions,
                                    "levels": level,
                                    "win": win,
                                    "primitive": "lattice_corridor_paths",
                                    "origin": list(origin),
                                    "pitch": pitch,
                                    "size": size,
                                    "start": list(start),
                                    "goal": list(goal),
                                    "goal_color": goal_color,
                                    "planned_steps": len(plan),
                                    "executed_steps": len(used),
                                }
    finally:
        game.close()
    return None


def _center_nodes(frame: list[list[int]], colors: set[int]) -> dict[tuple[int, int], int]:
    nodes: dict[tuple[int, int], int] = {}
    for color in colors:
        for cells in _component_cells(frame, color):
            if not (8 <= len(cells) <= 9):
                continue
            xs = [p[0] for p in cells]
            ys = [p[1] for p in cells]
            nodes[(round(sum(xs) / len(xs)), round(sum(ys) / len(ys)))] = color
    return nodes


def center_corridor_candidate(scratch: Path, *, max_steps: int = 48) -> dict[str, Any] | None:
    """Try simple paths on a graph of 3x3 component centers and color-2 connectors."""

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    game = SandboxGame(frontier["game"])
    try:
        game.reset()
        for action in prefix:
            _act(game, action)
            if bool(game.done):
                return None
        start_level = int(game.levels)
        frame = _frame_list(game.frame)
        nodes = _center_nodes(frame, {0, 2, 8, 9, 10, 11, 12, 13, 14})
        connectors = {p for p, color in nodes.items() if color == 2}
        starts = [p for p, color in nodes.items() if color == 9]
        goals = [p for p, color in nodes.items() if color == 14]
        if not starts or not goals:
            return None
        xs = sorted({p[0] for p in nodes})
        ys = sorted({p[1] for p in nodes})
        gaps = [
            b - a
            for vals in (xs, ys)
            for a, b in zip(vals, vals[1:])
            if b - a > 0
        ]
        pitch = min(gaps) if gaps else 6

        graphs: list[tuple[str, dict[tuple[int, int], list[tuple[tuple[int, int], int]]]]] = []
        endpoint_nodes = {p for p, color in nodes.items() if color != 2}
        sparse_graph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {p: [] for p in endpoint_nodes}
        for x, y in endpoint_nodes:
            for dx, dy, action in ((6, 0, 4), (-6, 0, 3), (0, 6, 2), (0, -6, 1)):
                dest = (x + dx, y + dy)
                mid = (x + dx // 2, y + dy // 2)
                if dest in endpoint_nodes and mid in connectors:
                    sparse_graph[(x, y)].append((dest, action))
        graphs.append(("sparse", sparse_graph))

        if pitch <= 3:
            walkable = set(nodes)
            dense_graph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {p: [] for p in walkable}
            for x, y in walkable:
                for dx, dy, action in ((pitch, 0, 4), (-pitch, 0, 3), (0, pitch, 2), (0, -pitch, 1)):
                    dest = (x + dx, y + dy)
                    if dest in walkable:
                        dense_graph[(x, y)].append((dest, action))
            graphs.append(("dense", dense_graph))

        for graph_kind, graph in graphs:
            for start in starts:
                if start not in graph:
                    continue
                for goal in goals:
                    if goal not in graph:
                        continue
                    for plan in _simple_paths(graph, start, goal, max_depth=max_steps, max_paths=128):
                        all_actions = prefix + plan
                        verified = _fresh_verified_level(
                            scratch,
                            frontier["game"],
                            all_actions,
                            must_exceed=start_level,
                        )
                        if verified is None:
                            continue
                        level, win = verified
                        return {
                            "game": frontier["game"],
                            "actions": all_actions,
                            "levels": level,
                            "win": win,
                            "primitive": "center_corridor_paths",
                            "graph": graph_kind,
                            "pitch": pitch,
                            "start": list(start),
                            "goal": list(goal),
                            "planned_steps": len(plan),
                            "executed_steps": len(plan),
                        }
    finally:
        game.close()
    return None


def bounded_preserving_search_candidate(
    scratch: Path,
    *,
    max_depth: int = 32,
    max_expansions: int = 8000,
    actions: Sequence[int] = (1, 2, 3, 4),
) -> dict[str, Any] | None:
    """Explore short action sequences from the frontier under verifier-side pruning.

    This is intentionally domain-light. It keeps only branches that preserve the
    current completed-level count and retain the active color-4/color-9 mover,
    while immediately accepting any branch that raises ``levels``. It is useful
    after an LLM or primitive discovers a partial mechanics model but not the
    next exact route.
    """

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    def new_game() -> Any:
        return SandboxGame(frontier["game"])

    game = new_game()
    try:
        def replay(suffix: Sequence[int]) -> tuple[int, int, int, bool, str]:
            game.reset()
            for action in prefix:
                _act(game, action)
                if bool(game.done):
                    break
            start_level = int(game.levels)
            win = int(game.win)
            for action in suffix:
                _act(game, [action])
                if bool(game.done):
                    break
            frame = _frame_list(game.frame)
            # Full rendered-frame hash is robust across moving hazards. Depth is
            # kept separately in the BFS key because some games have tick-based
            # hazard phase even when visible positions recur.
            signature = str(hash(tuple(tuple(int(v) for v in row) for row in frame)))
            return start_level, int(game.levels), win, bool(game.done), signature

        start_level, _, win, _, start_sig = replay(())

        queue: deque[tuple[int, ...]] = deque([()])
        seen: set[tuple[int, int, str]] = {(0, start_level, start_sig)}
        expansions = 0
        while queue and expansions < max_expansions:
            suffix = queue.popleft()
            if len(suffix) >= max_depth:
                continue
            for action in actions:
                nxt = suffix + (int(action),)
                expansions += 1
                _, level, win, done, sig = replay(nxt)
                if level > start_level:
                    all_actions = prefix + [[a] for a in nxt]
                    verified = _fresh_verified_level(
                        scratch,
                        frontier["game"],
                        all_actions,
                        must_exceed=start_level,
                    )
                    if verified is None:
                        continue
                    verify_level, verify_win = verified
                    return {
                        "game": frontier["game"],
                        "actions": all_actions,
                        "levels": verify_level,
                        "win": verify_win,
                        "primitive": "bounded_preserving_action_search",
                        "searched_depth": len(nxt),
                        "expansions": expansions,
                    }
                if done or level < start_level:
                    game.close()
                    game = new_game()
                    continue
                key = (len(nxt), level, sig)
                if key in seen:
                    continue
                seen.add(key)
                queue.append(nxt)
        return None
    finally:
        game.close()


def fresh_replay_survivor_search_candidate(
    scratch: Path,
    *,
    max_depth: int = 34,
    max_expansions: int = 2400,
    actions: Sequence[int] = (1, 2, 3, 4),
) -> dict[str, Any] | None:
    """Search from the frontier with a fresh sandbox instance per branch.

    Dynamic ARC levels can have hidden cadence state that is easy to corrupt by
    repeated ``reset`` calls in one worker. This primitive pays more verifier
    cost to avoid that class of false state: every branch is replayed from a new
    public ``SandboxGame`` object, and only a fresh level increase is accepted.
    """

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    game_id = frontier["game"]

    def eval_suffix(suffix: Sequence[int]) -> tuple[int, int, int, bool, list[list[int]], str]:
        game = SandboxGame(game_id)
        try:
            game.reset()
            for action in prefix:
                _act(game, action)
                if bool(game.done):
                    break
            base_level = int(game.levels)
            for action in suffix:
                _act(game, [action])
                if bool(game.done):
                    break
            frame = _frame_list(game.frame)
            return base_level, int(game.levels), int(game.win), bool(game.done), frame, _frame_digest(frame)
        finally:
            game.close()

    start_level, current_level, win, done, start_frame, start_sig = eval_suffix(())
    if done or current_level != start_level:
        return None

    dynamic_colors = {12, 13, 15}
    if not _salient_colors_present(start_frame, dynamic_colors):
        return None

    start_player = _largest_component_center(start_frame, {4, 9})
    goal = _closest_component_center(start_frame, {14}, start_player)

    def score_suffix(suffix: tuple[int, ...], frame: list[list[int]]) -> tuple[int, int, int]:
        player = _largest_component_center(frame, {4, 9})
        dist = 0
        if player is not None and goal is not None:
            dist = abs(player[0] - goal[0]) + abs(player[1] - goal[1])
        # Length remains the dominant term; distance only orders same-depth
        # survivors. The final tie-breaker rotates action bias by depth so the
        # search does not lock into one directional convention.
        return (len(suffix), dist, sum((i + 1) * a for i, a in enumerate(suffix)) % 17)

    queue: list[tuple[tuple[int, int, int], tuple[int, ...], list[list[int]]]] = [
        (score_suffix((), start_frame), (), start_frame)
    ]
    seen: set[tuple[int, int, str]] = {(0, start_level, start_sig)}
    expansions = 0
    best_depth = 0

    while queue and expansions < max_expansions:
        queue.sort(key=lambda item: item[0])
        _, suffix, _ = queue.pop(0)
        if len(suffix) >= max_depth:
            continue
        depth_actions = list(actions)
        rotate = len(suffix) % len(depth_actions)
        depth_actions = depth_actions[rotate:] + depth_actions[:rotate]
        for action in depth_actions:
            nxt = suffix + (int(action),)
            expansions += 1
            _, level, win, done, frame, sig = eval_suffix(nxt)
            best_depth = max(best_depth, len(nxt))
            if level > start_level:
                all_actions = prefix + [[a] for a in nxt]
                verified = _fresh_verified_level(
                    scratch,
                    game_id,
                    all_actions,
                    must_exceed=start_level,
                )
                if verified is None:
                    continue
                verify_level, verify_win = verified
                return {
                    "game": game_id,
                    "actions": all_actions,
                    "levels": verify_level,
                    "win": verify_win,
                    "primitive": "fresh_replay_survivor_search",
                    "searched_depth": len(nxt),
                    "expansions": expansions,
                    "start_level": start_level,
                }
            if done or level < start_level:
                continue
            key = (len(nxt), level, sig)
            if key in seen:
                continue
            seen.add(key)
            queue.append((score_suffix(nxt, frame), nxt, frame))
    return None


def temporal_corridor_phase_candidate(
    scratch: Path,
    *,
    max_depth: int = 34,
    max_goal_checks: int = 192,
    max_states_per_phase: int = 12,
    max_queue: int = 12000,
) -> dict[str, Any] | None:
    """Search visible corridor routes while allowing phase-setting detours.

    This handles levels where the visible maze is static but moving markers must
    be phased by deliberate backtracking. The search state is source-free:
    rendered corridor node, path length modulo a few plausible hazard periods,
    and the action sequence that got there. Candidate goal arrivals are then
    accepted only by a fresh sandbox replay.
    """

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    game_id = frontier["game"]
    game = SandboxGame(game_id)
    try:
        game.reset()
        for action in prefix:
            _act(game, action)
            if bool(game.done):
                return None
        start_level = int(game.levels)
        frame = _frame_list(game.frame)
    finally:
        game.close()

    if not _salient_colors_present(frame, {12, 13, 15}):
        return None

    nodes = _center_nodes(frame, {0, 2, 8, 9, 10, 11, 12, 13, 14, 15})
    starts = [p for p, color in nodes.items() if color == 9]
    goals = [p for p, color in nodes.items() if color == 14]
    if not starts or not goals:
        return None

    periods = (4, 6, 8, 10, 12)
    graphs: list[tuple[int, dict[tuple[int, int], list[tuple[tuple[int, int], int]]]]] = []
    for pitch in (6, 3):
        graph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {p: [] for p in nodes}
        for x, y in nodes:
            for dx, dy, action in ((pitch, 0, 4), (-pitch, 0, 3), (0, pitch, 2), (0, -pitch, 1)):
                dest = (x + dx, y + dy)
                if dest in nodes:
                    graph[(x, y)].append((dest, action))
        if any(graph.values()):
            graphs.append((pitch, graph))

    def dist_to_goal(node: tuple[int, int]) -> int:
        return min(abs(node[0] - goal[0]) + abs(node[1] - goal[1]) for goal in goals)

    for pitch, graph in graphs:
        for start in sorted(starts, key=dist_to_goal):
            queue: list[tuple[tuple[int, int, int], tuple[int, int], tuple[int, ...]]] = [
                ((0, dist_to_goal(start), 0), start, ())
            ]
            seen_counts: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {
                (start, tuple(0 for _ in periods)): 1
            }
            goal_checks = 0
            while queue and goal_checks < max_goal_checks and len(queue) < max_queue:
                queue.sort(key=lambda item: item[0])
                _, node, suffix = queue.pop(0)
                if len(suffix) >= max_depth:
                    continue
                for nxt_node, action in sorted(graph.get(node, []), key=lambda item: (dist_to_goal(item[0]), item[1])):
                    nxt = suffix + (action,)
                    phase = tuple(len(nxt) % p for p in periods)
                    key = (nxt_node, phase)
                    count = seen_counts.get(key, 0)
                    if count >= max_states_per_phase:
                        continue
                    seen_counts[key] = count + 1
                    if nxt_node in goals:
                        goal_checks += 1
                        all_actions = prefix + [[a] for a in nxt]
                        verified = _fresh_verified_level(
                            scratch,
                            game_id,
                            all_actions,
                            must_exceed=start_level,
                        )
                        if verified is not None:
                            level, win = verified
                            return {
                                "game": game_id,
                                "actions": all_actions,
                                "levels": level,
                                "win": win,
                                "primitive": "temporal_corridor_phase_search",
                                "pitch": pitch,
                                "start": list(start),
                                "goal": list(nxt_node),
                                "searched_depth": len(nxt),
                                "goal_checks": goal_checks,
                            }
                    score = (len(nxt), dist_to_goal(nxt_node), sum((i + 1) * a for i, a in enumerate(nxt)) % 23)
                    queue.append((score, nxt_node, nxt))
    return None


def _reverse_action(action: int) -> int:
    return {1: 2, 2: 1, 3: 4, 4: 3}[int(action)]


def simple_path_detour_candidate(
    scratch: Path,
    *,
    max_path_depth: int = 22,
    max_paths: int = 96,
    max_goal_checks: int = 1800,
) -> dict[str, Any] | None:
    """Verify simple visible routes with bounded reversible timing detours."""

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    game_id = frontier["game"]
    game = SandboxGame(game_id)
    try:
        game.reset()
        for action in prefix:
            _act(game, action)
            if bool(game.done):
                return None
        start_level = int(game.levels)
        frame = _frame_list(game.frame)
    finally:
        game.close()

    if not _salient_colors_present(frame, {12, 13, 15}):
        return None

    nodes = _center_nodes(frame, {0, 2, 8, 9, 10, 11, 12, 13, 14, 15})
    starts = [p for p, color in nodes.items() if color == 9]
    goals = [p for p, color in nodes.items() if color == 14]
    if not starts or not goals:
        return None

    checked: set[tuple[int, ...]] = set()
    goal_checks = 0
    probe = SandboxGame(game_id)
    try:
        def try_variants(
            variants: Sequence[tuple[int, ...]],
            *,
            pitch: int,
            start: tuple[int, int],
            goal: tuple[int, int],
            base: tuple[int, ...],
        ) -> dict[str, Any] | bool | None:
            nonlocal goal_checks, probe
            for suffix in variants:
                if suffix in checked:
                    continue
                checked.add(suffix)
                goal_checks += 1
                all_actions = prefix + [[a] for a in suffix]
                if goal_checks % 16 == 1 and goal_checks > 1:
                    probe.close()
                    probe = SandboxGame(game_id)
                probe_level, _, _ = _replay_actions(probe, all_actions)
                if probe_level > start_level:
                    verified = _fresh_verified_level(
                        scratch,
                        game_id,
                        all_actions,
                        must_exceed=start_level,
                    )
                    if verified is not None:
                        level, win = verified
                        return {
                            "game": game_id,
                            "actions": all_actions,
                            "levels": level,
                            "win": win,
                            "primitive": "simple_path_detour_search",
                            "pitch": pitch,
                            "start": list(start),
                            "goal": list(goal),
                            "base_steps": len(base),
                            "searched_depth": len(suffix),
                            "goal_checks": goal_checks,
                        }
                if goal_checks >= max_goal_checks:
                    return False
            return None

        for pitch in (6, 3):
            graph: dict[tuple[int, int], list[tuple[tuple[int, int], int]]] = {p: [] for p in nodes}
            for x, y in nodes:
                for dx, dy, action in ((pitch, 0, 4), (-pitch, 0, 3), (0, pitch, 2), (0, -pitch, 1)):
                    dest = (x + dx, y + dy)
                    if dest in nodes:
                        graph[(x, y)].append((dest, action))
            if not any(graph.values()):
                continue

            for start in starts:
                for goal in goals:
                    raw_plans = _simple_paths(graph, start, goal, max_depth=max_path_depth, max_paths=max_paths)
                    plan_records: list[tuple[tuple[int, int, int], list[list[int]], list[tuple[int, int]]]] = []
                    for plan in raw_plans:
                        base = tuple(int(a[0]) for a in plan)
                        path_nodes = [start]
                        node = start
                        valid_path = True
                        for action in base:
                            matches = [dest for dest, edge_action in graph.get(node, []) if edge_action == action]
                            if not matches:
                                valid_path = False
                                break
                            node = matches[0]
                            path_nodes.append(node)
                        if not valid_path:
                            continue
                        marker_visits = sum(1 for p in path_nodes[1:-1] if nodes.get(p) not in (0, 2, 9, 14))
                        turns = sum(1 for a, b in zip(base, base[1:]) if a != b)
                        plan_records.append(((-marker_visits, len(base), turns), plan, path_nodes))

                    for _, plan, path_nodes in sorted(plan_records, key=lambda item: item[0]):
                        base = tuple(int(a[0]) for a in plan)
                        variants: list[tuple[int, ...]] = [base]

                        loop_options: list[tuple[int, int, int]] = []
                        for idx, node in enumerate(path_nodes[:-1]):
                            next_action = base[idx] if idx < len(base) else None
                            edges = []
                            for dest, out_action in graph.get(node, []):
                                if any(
                                    back == _reverse_action(out_action) and back_dest == node
                                    for back_dest, back in graph.get(dest, [])
                                ):
                                    edges.append(
                                        (0 if out_action == next_action else 1, int(out_action), _reverse_action(out_action))
                                    )
                            for _, out_action, back_action in sorted(edges):
                                loop_options.append((idx, out_action, back_action))

                        # Repeated loops at one node cover timing waits without
                        # changing the eventual route.
                        for idx, out_action, back_action in loop_options:
                            for repeats in range(1, 5):
                                variants.append(
                                    base[:idx] + tuple([out_action, back_action] * repeats) + base[idx:]
                                )

                        # Two separated one-loop waits cover simple multi-gate
                        # phasing while keeping candidate count bounded.
                        for left_i, (idx_a, out_a, back_a) in enumerate(loop_options[:32]):
                            for idx_b, out_b, back_b in loop_options[left_i:left_i + 16]:
                                first, second = sorted(
                                    ((idx_a, out_a, back_a), (idx_b, out_b, back_b)),
                                    key=lambda item: item[0],
                                    reverse=True,
                                )
                                candidate = base
                                for idx, out_action, back_action in (first, second):
                                    candidate = candidate[:idx] + (out_action, back_action) + candidate[idx:]
                                variants.append(candidate)

                        hit = try_variants(variants, pitch=pitch, start=start, goal=goal, base=base)
                        if isinstance(hit, dict):
                            return hit
                        if hit is False:
                            return None

                        # Some dynamic corridors need one short detour plus a
                        # longer wait loop at another node. Keep this bounded
                        # and biased toward loop options close to the base path.
                        extended_variants: list[tuple[int, ...]] = []
                        repeat_pairs = ((1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1), (2, 2))
                        for left_i, (idx_a, out_a, back_a) in enumerate(loop_options[:24]):
                            for idx_b, out_b, back_b in loop_options[left_i + 1:left_i + 13]:
                                for repeats_a, repeats_b in repeat_pairs:
                                    first, second = sorted(
                                        (
                                            (idx_a, out_a, back_a, repeats_a),
                                            (idx_b, out_b, back_b, repeats_b),
                                        ),
                                        key=lambda item: item[0],
                                        reverse=True,
                                    )
                                    candidate = base
                                    for idx, out_action, back_action, repeats in (first, second):
                                        candidate = (
                                            candidate[:idx]
                                            + tuple([out_action, back_action] * repeats)
                                            + candidate[idx:]
                                        )
                                    extended_variants.append(candidate)
                        hit = try_variants(extended_variants, pitch=pitch, start=start, goal=goal, base=base)
                        if isinstance(hit, dict):
                            return hit
                        if hit is False:
                            return None
    finally:
        probe.close()
    return None


def sandbox_frontier_explore_candidate(
    scratch: Path,
    *,
    budget: int = 2500,
    max_steps: int = 80,
    max_clicks: int = 24,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Source-free graph-frontier exploration through ``SandboxGame`` only.

    This is the cold-start rung: no prior traces and no game source. It builds a
    small public action set from directional actions plus click centers inferred
    from rendered connected components, then biases rollouts toward untested
    ``(frame, action)`` pairs. Any level-up is fresh-replay verified before it
    becomes a candidate.
    """

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    game_id = frontier["game"]
    rng = random.Random(seed)
    tested: dict[str, set[tuple[int, ...]]] = {}
    click_cache: dict[str, list[list[int]]] = {}
    best: dict[str, Any] | None = None
    expansions = 0
    start_time = time.time()

    def frame_key(frame: Any) -> str:
        if frame is None:
            return "none"
        return _frame_digest(_frame_list(frame))

    def action_key(action: Sequence[int]) -> tuple[int, ...]:
        return tuple(int(x) for x in action)

    def macro_key(macro: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
        return tuple(action_key(action) for action in macro)

    def base_macros(frame: Any) -> list[list[list[int]]]:
        return _cold_macros(frame, max_clicks=max_clicks, cache=click_cache)

    game = SandboxGame(game_id)
    try:
        game.reset()
        for action in prefix:
            _act(game, action)
            if bool(game.done):
                return None
        start_level = int(game.levels)
        win = int(game.win)
        initial_macros = base_macros(game.frame)

        while expansions < budget:
            game.reset()
            for action in prefix:
                _act(game, action)
                if bool(game.done):
                    break
            if bool(game.done):
                return None

            seq: list[list[int]] = []
            frame = game.frame
            macros = list(initial_macros)
            for _depth in range(max_steps):
                state = frame_key(frame)
                tried = tested.setdefault(state, set())
                untried = [macro for macro in macros if macro_key(macro) not in tried]
                if untried:
                    macro = rng.choice(untried[:32])
                else:
                    clicks = [macro for macro in macros if len(macro) == 1 and int(macro[0][0]) == 6]
                    macro = rng.choice(clicks if clicks and rng.random() < 0.45 else macros)
                tried.add(macro_key(macro))
                for action in macro:
                    try:
                        _act(game, action)
                    except Exception:
                        break
                    expansions += 1
                    seq.append(list(action))
                    if int(game.levels) > start_level:
                        all_actions = prefix + seq
                        verified = _fresh_verified_level(
                            scratch,
                            game_id,
                            all_actions,
                            must_exceed=start_level,
                        )
                        if verified is None:
                            break
                        level, verify_win = verified
                        return {
                            "game": game_id,
                            "actions": all_actions,
                            "levels": level,
                            "win": verify_win,
                            "primitive": "sandbox_frontier_explore",
                            "search_prior": "fibonacci_zeckendorf_macros",
                            "searched_steps": len(seq),
                            "expansions": expansions,
                            "states": len(tested),
                            "elapsed_s": round(time.time() - start_time, 3),
                        }
                    if bool(game.done) or int(game.levels) < start_level or expansions >= budget:
                        break
                if bool(game.done) or int(game.levels) < start_level or expansions >= budget:
                    break
                frame = game.frame
                macros = base_macros(frame)

        return best
    finally:
        game.close()


def first_level_macro_tournament_candidate(
    scratch: Path,
    *,
    max_candidates: int = 900,
    max_clicks: int = 18,
) -> dict[str, Any] | None:
    """Tournament independent first-level macro families before random search."""

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    game_id = frontier["game"]
    click_cache: dict[str, list[list[int]]] = {}
    start_time = time.time()

    def replay_candidate(game: Any, suffix: Sequence[Sequence[int]]) -> tuple[int, int, bool]:
        game.reset()
        for action in prefix:
            _act(game, action)
            if bool(game.done):
                break
        start_level = int(game.levels)
        for action in suffix:
            _act(game, action)
            if bool(game.done) or int(game.levels) > start_level:
                break
        return int(game.levels), int(game.win), bool(game.done)

    def directional_runs() -> list[list[list[int]]]:
        return _fibonacci_direction_macros()

    def saturation_runs(game: Any) -> list[list[list[int]]]:
        macros: list[list[list[int]]] = []
        for action in (1, 2, 3, 4):
            game.reset()
            for p in prefix:
                _act(game, p)
                if bool(game.done):
                    break
            if bool(game.done):
                continue
            seq: list[list[int]] = []
            last = _frame_digest(_frame_list(game.frame))
            for _ in range(24):
                _act(game, [action])
                seq.append([action])
                now = "none" if game.frame is None else _frame_digest(_frame_list(game.frame))
                if bool(game.done) or now == last:
                    break
                last = now
            if seq:
                macros.append(seq)
        return macros

    def click_families(frame: Any) -> list[list[list[int]]]:
        clicks = _frame_click_actions(frame, max_clicks=max_clicks, cache=click_cache)
        macros: list[list[list[int]]] = [[click] for click in clicks]
        for idx, first in enumerate(clicks[:12]):
            for second in clicks[idx + 2 : idx + 10 : 2]:
                macros.append([first, second])
        for click in clicks[:8]:
            for direction in (1, 2, 3, 4):
                for length in (2, 3, 5, 8):
                    run = [[direction] for _ in range(length)]
                    macros.append([click] + run)
                    macros.append(run + [click])
                for total in (6, 9, 11):
                    run = [[direction] for part in _zeckendorf_parts(total) for _ in range(part)]
                    macros.append([click] + run + [click])
        return macros

    def interact_families() -> list[list[list[int]]]:
        macros: list[list[list[int]]] = []
        for action in (5, 7):
            for length in (1, 2, 3, 5, 8):
                macros.append([[action] for _ in range(length)])
        for action in (5, 7):
            for direction in (1, 2, 3, 4):
                for length in (2, 3, 5):
                    run = [[direction] for _ in range(length)]
                    macros.append([[action]] + run)
                    macros.append(run + [[action]])
        return macros

    game = SandboxGame(game_id)
    attempts = 0
    best: dict[str, Any] | None = None
    try:
        game.reset()
        for action in prefix:
            _act(game, action)
            if bool(game.done):
                return None
        start_level = int(game.levels)
        start_frame = game.frame
        if start_level > 0:
            return None
        families: list[tuple[str, list[list[list[int]]]]] = [
            ("directional_zeckendorf", directional_runs()),
            ("saturation_runs", saturation_runs(game)),
            ("click_zeckendorf", click_families(start_frame)),
            ("interact_loops", interact_families()),
        ]
        seen: set[tuple[tuple[int, ...], ...]] = set()
        for family, macros in families:
            for suffix in macros:
                key = tuple(tuple(int(v) for v in action) for action in suffix)
                if key in seen:
                    continue
                seen.add(key)
                attempts += 1
                if attempts > max_candidates:
                    break
                try:
                    level, win, done = replay_candidate(game, suffix)
                except Exception:
                    continue
                if level <= start_level:
                    continue
                all_actions = prefix + [list(action) for action in suffix]
                verified = _fresh_verified_level(
                    scratch,
                    game_id,
                    all_actions,
                    must_exceed=start_level,
                )
                if verified is None:
                    continue
                verify_level, verify_win = verified
                candidate = {
                    "game": game_id,
                    "actions": all_actions,
                    "levels": verify_level,
                    "win": verify_win,
                    "primitive": "first_level_macro_tournament",
                    "macro_family": family,
                    "searched_steps": len(suffix),
                    "attempts": attempts,
                    "elapsed_s": round(time.time() - start_time, 3),
                }
                if best is None or len(candidate["actions"]) < len(best["actions"]):
                    best = candidate
            if attempts > max_candidates:
                break
        return best
    finally:
        game.close()


def code_world_discriminator_candidate(
    scratch: Path,
    *,
    budget: int = 220,
    beam_width: int = 6,
    max_depth: int = 4,
    max_clicks: int = 12,
) -> dict[str, Any] | None:
    """Verifier-gated code-world discriminator for cold macro search.

    The learned model is intentionally small and local: it only sees public
    rendered frames, stores observed object-key transitions, and uses those
    transitions to rank future macro branches. ``SandboxGame`` remains the only
    source of truth for accepting a candidate.
    """

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    game_id = frontier["game"]
    click_cache: dict[str, list[list[int]]] = {}
    transition_counts: dict[tuple[str, tuple[tuple[int, ...], ...]], dict[str, int]] = {}
    transition_examples: dict[tuple[tuple[int, ...], ...], list[tuple[str, str, int, bool]]] = {}
    macro_stats: dict[tuple[tuple[int, ...], ...], dict[str, int]] = {}
    macro_catalog: dict[tuple[tuple[int, ...], ...], list[list[int]]] = {}
    start_time = time.time()

    def macro_key(macro: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(v) for v in action) for action in macro)

    def object_key(frame: Any) -> str:
        if frame is None:
            return "none"
        rows = _frame_list(frame)
        return json.dumps(_composite_world_key(rows), sort_keys=True, separators=(",", ":"))

    def parse_key(key: str) -> dict[str, Any] | None:
        try:
            data = json.loads(key)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        return data

    def key_distance(data: Mapping[str, Any]) -> int:
        value = data.get("distance", -1)
        try:
            return int(value)
        except Exception:
            return 9999

    def replace_player_key(
        key: str,
        player: tuple[int, int] | None,
        *,
        active_delta: int = 0,
        level: int = 0,
    ) -> str | None:
        data = parse_key(key)
        if data is None:
            return None
        goal_raw = data.get("goal")
        goal: tuple[int, int] | None = None
        if isinstance(goal_raw, (list, tuple)) and len(goal_raw) == 2:
            try:
                goal = (int(goal_raw[0]), int(goal_raw[1]))
            except Exception:
                goal = None
        if player is not None:
            data["player"] = [int(player[0]), int(player[1])]
            data["distance"] = -1 if goal is None else abs(player[0] - goal[0]) + abs(player[1] - goal[1])
        try:
            data["active"] = int(data.get("active", 0)) + int(active_delta)
        except Exception:
            data["active"] = int(active_delta)
        if level > 0:
            data["predicted_level_delta"] = int(level)
        return json.dumps(data, sort_keys=True, separators=(",", ":"))

    probe_game = SandboxGame(game_id)

    def replay_suffix(suffix: Sequence[Sequence[int]]) -> tuple[int, int, bool, list[list[int]] | None, str]:
        probe_game.reset()
        for action in prefix:
            _act(probe_game, action)
            if bool(probe_game.done):
                break
        for action in suffix:
            _act(probe_game, action)
            if bool(probe_game.done):
                break
        frame = None if probe_game.frame is None else _frame_list(probe_game.frame)
        return int(probe_game.levels), int(probe_game.win), bool(probe_game.done), frame, object_key(frame)

    def macro_library(frame: Any) -> list[list[list[int]]]:
        macros = _cold_macros(frame, max_clicks=max_clicks, cache=click_cache)
        # A few short compositions give the discriminator something to model
        # without exploding the raw action branching factor.
        clicks = _frame_click_actions(frame, max_clicks=min(10, max_clicks), cache=click_cache)
        for first in clicks[:6]:
            for second in clicks[1:7]:
                if first != second:
                    macros.append([first, second])
        seen: set[tuple[tuple[int, ...], ...]] = set()
        unique: list[list[list[int]]] = []
        for macro in macros:
            key = macro_key(macro)
            if key in seen or len(macro) > 13:
                continue
            seen.add(key)
            macro_catalog.setdefault(key, [list(action) for action in macro])
            unique.append(macro)
        return unique

    def update_model(src_key: str, macro: Sequence[Sequence[int]], dst_key: str, *, level_delta: int, done: bool) -> None:
        mk = macro_key(macro)
        macro_catalog.setdefault(mk, [list(action) for action in macro])
        row = transition_counts.setdefault((src_key, mk), {})
        row[dst_key] = row.get(dst_key, 0) + 1
        transition_examples.setdefault(mk, []).append((src_key, dst_key, int(level_delta), bool(done)))
        stats = macro_stats.setdefault(mk, {"uses": 0, "novel": 0, "level": 0, "dead": 0, "noop": 0})
        stats["uses"] += 1
        stats["level"] += max(0, int(level_delta))
        stats["dead"] += 1 if done and level_delta <= 0 else 0
        stats["noop"] += 1 if dst_key == src_key else 0

    def model_confidence(src_key: str, macro: Sequence[Sequence[int]]) -> tuple[int, int]:
        row = transition_counts.get((src_key, macro_key(macro)), {})
        if not row:
            return (0, 0)
        counts = sorted(row.values(), reverse=True)
        return (counts[0], sum(counts))

    def branch_score(
        before_frame: list[list[int]],
        after_frame: list[list[int]] | None,
        *,
        before_key: str,
        after_key: str,
        macro: Sequence[Sequence[int]],
        level_delta: int,
        done: bool,
        visited: set[str],
        path_len: int,
    ) -> tuple[int, int, int, int, int, int, int, int]:
        if after_frame is None:
            return (-9999, -9999, -9999, -9999, -9999, -9999, -9999, -path_len)
        stats = macro_stats.get(macro_key(macro), {"uses": 0, "level": 0, "dead": 0, "noop": 0})
        confidence_hits, confidence_total = model_confidence(before_key, macro)
        before_dist = _distance_to_goal(before_frame)
        after_dist = _distance_to_goal(after_frame)
        progress = before_dist - after_dist
        activation = _small_activation_count(after_frame) - _small_activation_count(before_frame)
        novelty = 1 if after_key not in visited else 0
        survival = 0 if done and level_delta <= 0 else 1
        consistency = confidence_hits - max(0, confidence_total - confidence_hits)
        prior = int(stats["level"]) * 6 - int(stats["dead"]) * 4 - int(stats["noop"])
        return (
            int(level_delta) * 100,
            survival,
            novelty,
            int(progress),
            int(activation),
            int(consistency),
            int(prior),
            -path_len,
        )

    def induced_effects() -> dict[tuple[tuple[int, ...], ...], tuple[int, int, int, int]]:
        effects: dict[tuple[tuple[int, ...], ...], tuple[int, int, int, int]] = {}
        for mk, examples in transition_examples.items():
            deltas: dict[tuple[int, int], int] = {}
            active_deltas: dict[int, int] = {}
            level_hits = 0
            live = 0
            for src_key, dst_key, level_delta, done in examples:
                if done and level_delta <= 0:
                    continue
                src = parse_key(src_key)
                dst = parse_key(dst_key)
                if src is None or dst is None:
                    continue
                try:
                    active_delta = int(dst.get("active", 0)) - int(src.get("active", 0))
                    active_deltas[active_delta] = active_deltas.get(active_delta, 0) + 1
                except Exception:
                    pass
                sp = src.get("player")
                dp = dst.get("player")
                if (
                    isinstance(sp, (list, tuple))
                    and isinstance(dp, (list, tuple))
                    and len(sp) == 2
                    and len(dp) == 2
                ):
                    dx = int(dp[0]) - int(sp[0])
                    dy = int(dp[1]) - int(sp[1])
                    deltas[(dx, dy)] = deltas.get((dx, dy), 0) + 1
                level_hits += max(0, int(level_delta))
                live += 1
            if live <= 0:
                continue
            if deltas:
                (dx, dy), count = max(deltas.items(), key=lambda item: item[1])
                if count * 2 < live:
                    dx, dy = 0, 0
            else:
                dx, dy = 0, 0
            active_delta, active_count = (0, 0)
            if active_deltas:
                active_delta, active_count = max(active_deltas.items(), key=lambda item: item[1])
            if active_count and active_count * 2 < live:
                active_delta = 0
            if (dx, dy) == (0, 0) and active_delta == 0 and level_hits <= 0:
                continue
            effects[mk] = (dx, dy, int(active_delta), min(1, level_hits))
        return effects

    def effect_model_plans(start_key: str, *, max_plans: int = 4, depth: int = 8, beam: int = 12) -> list[list[list[int]]]:
        effects = induced_effects()
        if not effects:
            return []
        start_data = parse_key(start_key)
        if start_data is None:
            return []
        start_dist = key_distance(start_data)
        nodes: list[tuple[tuple[int, int, int, int], str, list[list[int]]]] = [((0, -start_dist, 0, 0), start_key, [])]
        best: list[tuple[tuple[int, int, int, int], list[list[int]]]] = []
        seen: set[tuple[str, int]] = {(start_key, 0)}
        for _ in range(depth):
            candidates: list[tuple[tuple[int, int, int, int], str, list[list[int]]]] = []
            for _score, key, path in nodes:
                data = parse_key(key)
                if data is None:
                    continue
                player_raw = data.get("player")
                has_player = isinstance(player_raw, (list, tuple)) and len(player_raw) == 2
                px, py = (int(player_raw[0]), int(player_raw[1])) if has_player else (0, 0)
                before_dist = key_distance(data)
                before_active = int(data.get("active", 0) or 0)
                for mk, (dx, dy, active_delta, level_hint) in effects.items():
                    macro = macro_catalog.get(mk)
                    if not macro:
                        continue
                    next_player = (px + dx, py + dy) if has_player else None
                    next_key = replace_player_key(key, next_player, active_delta=active_delta, level=level_hint)
                    if next_key is None:
                        continue
                    after = parse_key(next_key)
                    if after is None:
                        continue
                    after_dist = key_distance(after)
                    after_active = int(after.get("active", 0) or 0)
                    progress = before_dist - after_dist
                    next_path = path + [list(action) for action in macro]
                    score = (
                        int(level_hint) * 100,
                        -after_dist,
                        max(int(progress), int(after_active - before_active)),
                        -len(next_path),
                    )
                    if (
                        level_hint > 0
                        or after_dist <= 1
                        or (start_dist > 0 and after_dist < start_dist)
                        or after_active > before_active
                    ):
                        best.append((score, next_path))
                    visit_key = (next_key, len(next_path))
                    if visit_key in seen:
                        continue
                    seen.add(visit_key)
                    candidates.append((score, next_key, next_path))
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0], reverse=True)
            nodes = candidates[:beam]
        best.sort(key=lambda item: item[0], reverse=True)
        plans: list[list[list[int]]] = []
        plan_seen: set[tuple[tuple[int, ...], ...]] = set()
        for _score, plan in best:
            key = tuple(tuple(int(v) for v in action) for action in plan)
            if key in plan_seen:
                continue
            plan_seen.add(key)
            plans.append(plan)
            if len(plans) >= max_plans:
                break
        return plans

    evaluations = 0
    effect_plan_checks = 0
    verified_effect_plan_keys: set[tuple[tuple[int, ...], ...]] = set()
    best: dict[str, Any] | None = None
    try:
        start_level, _, start_done, start_frame, start_key = replay_suffix([])
        if start_done or start_frame is None:
            return None

        beam: list[tuple[tuple[int, ...], list[list[int]], list[list[int]], str, set[str]]] = [
            ((0, 1, 0, 0, 0, 0, 0, 0), [], start_frame, start_key, {start_key})
        ]

        for depth in range(max_depth):
            branches: list[tuple[tuple[int, ...], list[list[int]], list[list[int]], str, set[str]]] = []
            for _score, suffix, frame, key, visited in beam:
                macros = macro_library(frame)
                macros.sort(
                    key=lambda macro: (
                        model_confidence(key, macro)[0],
                        macro_stats.get(macro_key(macro), {}).get("level", 0),
                        -macro_stats.get(macro_key(macro), {}).get("dead", 0),
                        -len(macro),
                    ),
                    reverse=True,
                )
                for macro in macros[:28]:
                    if evaluations >= budget:
                        break
                    candidate_suffix = suffix + [list(action) for action in macro]
                    try:
                        level, win, done, after_frame, after_key = replay_suffix(candidate_suffix)
                    except Exception:
                        evaluations += 1
                        continue
                    evaluations += 1
                    level_delta = level - start_level
                    update_model(key, macro, after_key, level_delta=level_delta, done=done)
                    if level_delta > 0:
                        all_actions = prefix + candidate_suffix
                        verified = _fresh_verified_level(
                            scratch,
                            game_id,
                            all_actions,
                            must_exceed=start_level,
                        )
                        if verified is None:
                            continue
                        verify_level, verify_win = verified
                        candidate = {
                            "game": game_id,
                            "actions": all_actions,
                            "levels": verify_level,
                            "win": verify_win,
                            "primitive": "code_world_discriminator",
                            "search_prior": "object_key_world_model",
                            "searched_steps": len(candidate_suffix),
                            "depth": depth + 1,
                            "evaluations": evaluations,
                            "model_transitions": len(transition_counts),
                            "elapsed_s": round(time.time() - start_time, 3),
                        }
                        if best is None or len(candidate["actions"]) < len(best["actions"]):
                            best = candidate
                        continue
                    if done or after_frame is None or level < start_level:
                        continue
                    score = branch_score(
                        frame,
                        after_frame,
                        before_key=key,
                        after_key=after_key,
                        macro=macro,
                        level_delta=level_delta,
                        done=done,
                        visited=visited,
                        path_len=len(candidate_suffix),
                    )
                    if score[1] <= 0:
                        continue
                    branches.append((score, candidate_suffix, after_frame, after_key, visited | {after_key}))
                if evaluations >= budget:
                    break
            if best is not None:
                return best
            for plan in effect_model_plans(start_key):
                plan_key = tuple(tuple(int(v) for v in action) for action in plan)
                if plan_key in verified_effect_plan_keys:
                    continue
                verified_effect_plan_keys.add(plan_key)
                effect_plan_checks += 1
                all_actions = prefix + plan
                verified = _fresh_verified_level(
                    scratch,
                    game_id,
                    all_actions,
                    must_exceed=start_level,
                )
                if verified is None:
                    continue
                verify_level, verify_win = verified
                best = {
                    "game": game_id,
                    "actions": all_actions,
                    "levels": verify_level,
                    "win": verify_win,
                    "primitive": "code_world_discriminator",
                    "search_prior": "induced_object_effect_model",
                    "searched_steps": len(plan),
                    "depth": depth + 1,
                        "evaluations": evaluations,
                        "effect_plan_checks": effect_plan_checks,
                        "model_transitions": len(transition_counts),
                    "induced_effects": len(induced_effects()),
                    "elapsed_s": round(time.time() - start_time, 3),
                }
                return best
            if not branches:
                break
            branches.sort(key=lambda item: item[0], reverse=True)
            next_beam: list[tuple[tuple[int, ...], list[list[int]], list[list[int]], str, set[str]]] = []
            seen_round: set[tuple[int, str]] = set()
            for item in branches:
                round_key = (len(item[1]), item[3])
                if round_key in seen_round:
                    continue
                seen_round.add(round_key)
                next_beam.append(item)
                if len(next_beam) >= beam_width:
                    break
            beam = next_beam
        return best
    finally:
        stats_path = scratch / "code_world_discriminator_stats.json"
        try:
            stats_path.write_text(
                json.dumps(
                    {
                        "primitive": "code_world_discriminator",
                        "game": game_id,
                        "evaluations": evaluations,
                        "transition_keys": len(transition_counts),
                        "induced_effects": len(induced_effects()),
                        "effect_plan_checks": effect_plan_checks,
                        "macro_keys": len(macro_stats),
                        "found_candidate": best is not None,
                        "budget": budget,
                        "beam_width": beam_width,
                        "max_depth": max_depth,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        except Exception:
            pass
        probe_game.close()


def go_explore_archive_candidate(
    scratch: Path,
    *,
    budget: int = 350,
    max_suffix_len: int = 120,
    max_clicks: int = 18,
    archive_limit: int = 96,
    seed: int = 17,
    max_fresh_checks: int = 2,
) -> dict[str, Any] | None:
    """Go-Explore style archive search over public sandbox observations.

    The frontier explorer samples rollouts from the same start repeatedly. This
    primitive stores replayable suffixes to novel public states, then branches
    from those states. It is still source-free: every cell is a rendered-frame
    digest/object summary and every accepted route is fresh replay verified.
    """

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    game_id = frontier["game"]
    rng = random.Random(seed)
    click_cache: dict[str, list[list[int]]] = {}
    start_time = time.time()
    expansions = 0
    level_hits = 0
    rejected_hits = 0
    modality = "sparse_objects"

    def replay(game: Any, suffix: Sequence[Sequence[int]]) -> tuple[int, int, bool, list[list[int]] | None]:
        game.reset()
        for action in prefix:
            _act(game, action)
            if bool(game.done):
                break
        for action in suffix:
            _act(game, action)
            if bool(game.done):
                break
        frame = None if game.frame is None else _frame_list(game.frame)
        return int(game.levels), int(game.win), bool(game.done), frame

    def base_state_key(frame: list[list[int]] | None) -> str:
        if frame is None:
            return "none"
        return json.dumps(_composite_world_key(frame, modality=modality), sort_keys=True, separators=(",", ":"))

    def state_key(
        frame: list[list[int]] | None,
        suffix: Sequence[Sequence[int]],
        *,
        include_machine: bool = False,
    ) -> str:
        if frame is None:
            return "none"
        return base_state_key(frame)

    def cell_metrics(frame: list[list[int]]) -> tuple[int, int]:
        return _distance_to_goal(frame), _small_activation_count(frame)

    def cell_score(cell: Mapping[str, Any]) -> tuple[int, int, int, int]:
        return (
            -int(cell["visits"]),
            -int(cell["distance"]),
            int(cell["active"]),
            -len(cell["suffix"]),
        )

    def macro_pool(frame: list[list[int]], suffix: Sequence[Sequence[int]]) -> list[list[list[int]]]:
        macros = _cold_macros(frame, max_clicks=max_clicks, cache=click_cache)
        # Local continuation macros are cheap and help when the avatar/object is
        # already near an interesting state.
        for action in (1, 2, 3, 4, 5, 7):
            for length in (1, 2, 3):
                macros.append([[action] for _ in range(length)])
        clicks = _frame_click_actions(frame, max_clicks=min(8, max_clicks), cache=click_cache)
        if suffix:
            last = list(suffix[-1])
            if last and int(last[0]) == 6:
                for direction in (1, 2, 3, 4):
                    macros.append([last] + [[direction] for _ in range(2)])
        for first in clicks[:5]:
            for second in clicks[:5]:
                if first != second:
                    macros.append([first, second])
        seen: set[tuple[tuple[int, ...], ...]] = set()
        unique: list[list[list[int]]] = []
        for macro in macros:
            key = tuple(tuple(int(v) for v in action) for action in macro)
            if key in seen:
                continue
            seen.add(key)
            unique.append(macro)
        return unique

    game = SandboxGame(game_id)
    try:
        start_level, _, done, start_frame = replay(game, [])
        if done or start_frame is None:
            return None
        modality = _world_modality(start_frame, available_actions=getattr(game, "avail", []))
        start_base_key = base_state_key(start_frame)
        start_key = state_key(start_frame, [], include_machine=False)
        start_dist, start_active = cell_metrics(start_frame)
        archive: dict[str, dict[str, Any]] = {
            start_key: {
                "base_key": start_base_key,
                "suffix": [],
                "frame": start_frame,
                "visits": 0,
                "level": start_level,
                "distance": start_dist,
                "active": start_active,
            }
        }
        frontier_keys: list[str] = [start_key]
        seen_suffixes: set[tuple[tuple[int, ...], ...]] = {()}

        while expansions < budget and frontier_keys:
            frontier_keys.sort(
                key=lambda key: cell_score(archive[key]),
                reverse=True,
            )
            # Sample among the top cells to avoid deterministic tunnel vision.
            top = frontier_keys[: min(18, len(frontier_keys))]
            key = rng.choice(top)
            cell = archive[key]
            cell["visits"] = int(cell["visits"]) + 1
            suffix = [list(action) for action in cell["suffix"]]
            frame = cell["frame"]
            macros = macro_pool(frame, suffix)
            rng.shuffle(macros)
            for macro in macros[:24]:
                if expansions >= budget:
                    break
                candidate_suffix = suffix + [list(action) for action in macro]
                if len(candidate_suffix) > max_suffix_len:
                    continue
                suffix_key = tuple(tuple(int(v) for v in action) for action in candidate_suffix)
                if suffix_key in seen_suffixes:
                    continue
                seen_suffixes.add(suffix_key)
                try:
                    level, win, done, next_frame = replay(game, candidate_suffix)
                except Exception:
                    expansions += 1
                    continue
                expansions += 1
                if level > start_level:
                    level_hits += 1
                    if level_hits > max_fresh_checks:
                        continue
                    all_actions = prefix + candidate_suffix
                    verified = _fresh_verified_level(
                        scratch,
                        game_id,
                        all_actions,
                        must_exceed=start_level,
                    )
                    if verified is None:
                        rejected_hits += 1
                        continue
                    verify_level, verify_win = verified
                    return {
                        "game": game_id,
                        "actions": all_actions,
                        "levels": verify_level,
                        "win": verify_win,
                        "primitive": "go_explore_archive",
                        "search_prior": "public_state_archive",
                        "searched_steps": len(candidate_suffix),
                        "expansions": expansions,
                        "archive_size": len(archive),
                        "level_hits": level_hits,
                        "rejected_hits": rejected_hits,
                        "elapsed_s": round(time.time() - start_time, 3),
                    }
                if done or next_frame is None or level < start_level:
                    continue
                next_base_key = base_state_key(next_frame)
                next_key = state_key(next_frame, candidate_suffix, include_machine=False)
                previous = archive.get(next_key)
                if previous is None or len(candidate_suffix) < len(previous["suffix"]):
                    dist, active = cell_metrics(next_frame)
                    archive[next_key] = {
                        "base_key": next_base_key,
                        "suffix": candidate_suffix,
                        "frame": next_frame,
                        "visits": 0,
                        "level": level,
                        "distance": dist,
                        "active": active,
                    }
                    if next_key not in frontier_keys:
                        frontier_keys.append(next_key)
                    if len(archive) > archive_limit:
                        frontier_keys.sort(
                            key=lambda item: cell_score(archive[item]),
                            reverse=True,
                        )
                        for old_key in frontier_keys[archive_limit:]:
                            archive.pop(old_key, None)
                        frontier_keys = frontier_keys[:archive_limit]
        stats_path = scratch / "go_explore_archive_stats.json"
        stats_path.write_text(
            json.dumps(
                {
                    "primitive": "go_explore_archive",
                    "game": game_id,
                    "expansions": expansions,
                    "archive_size": len(archive),
                    "found_candidate": False,
                    "level_hits": level_hits,
                    "rejected_hits": rejected_hits,
                    "budget": budget,
                    "max_suffix_len": max_suffix_len,
                    "max_fresh_checks": max_fresh_checks,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return None
    finally:
        game.close()


def hidden_click_state_candidate(
    scratch: Path,
    *,
    max_expansions: int = 2200,
    max_depth: int = 12,
    max_clicks: int = 28,
    beam_width: int = 96,
) -> dict[str, Any] | None:
    """Click-only hidden-state search with bounded click-history state.

    Some click games intentionally keep the visible frame unchanged while a
    selection/register changes. A visible-state archive collapses those paths.
    This primitive treats short click history as a latent machine state, but
    only for click-only frontiers and with strict beam/expansion bounds.
    """

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    game_id = frontier["game"]
    start_time = time.time()
    expansions = 0
    level_hits = 0
    rejected_hits = 0

    probe_game = SandboxGame(game_id)

    def replay(suffix: Sequence[Sequence[int]]) -> tuple[int, int, bool, list[list[int]] | None, list[int]]:
        probe_game.reset()
        for action in prefix:
            _act(probe_game, action)
            if bool(probe_game.done):
                break
        for action in suffix:
            _act(probe_game, action)
            if bool(probe_game.done):
                break
        frame = None if probe_game.frame is None else _frame_list(probe_game.frame)
        return (
            int(probe_game.levels),
            int(probe_game.win),
            bool(probe_game.done),
            frame,
            [int(a) for a in getattr(probe_game, "avail", [])],
        )

    try:
        start_level, _win, done, start_frame, avail = replay([])
    except Exception:
        probe_game.close()
        return None
    if done or start_frame is None or set(avail) != {6}:
        probe_game.close()
        return None

    components = _component_records(start_frame)
    click_rows: list[tuple[int, int, int, int, list[int]]] = []
    for component in components:
        color = int(component["c"])
        size = int(component["n"])
        x = int(component["x"])
        y = int(component["y"])
        if size <= 0 or size > 32 or y >= 63:
            continue
        # Hidden click puzzles often use tiny top-row markers as registers or
        # clues; include them here even though the generic salience clicker
        # ranks them low.
        top_priority = 0 if y <= 3 and size <= 8 else 1
        color_priority = 0 if color in (8, 9, 10, 11, 12, 13, 14, 15) else 1
        click_rows.append((top_priority, color_priority, size, y, [6, x, y]))
    click_rows.sort()
    clicks = [row[-1] for row in click_rows]
    for action in _frame_click_actions(start_frame, max_clicks=max_clicks):
        if action not in clicks:
            clicks.append(action)
    clicks = clicks[:max_clicks]
    if not clicks:
        return None
    click_to_index = {tuple(action): idx for idx, action in enumerate(clicks)}

    def click_rank(action: Sequence[int]) -> tuple[int, int, int, int]:
        x = int(action[1])
        y = int(action[2])
        match = None
        for component in components:
            if int(component["x"]) == x and int(component["y"]) == y:
                match = component
                break
        if match is None:
            return (9, y, x, 999)
        color = int(match["c"])
        size = int(match["n"])
        # Prefer small colored widgets before broad background/rails.
        color_priority = 0 if color in (8, 9, 10, 11, 12, 13, 14, 15) else 1
        return (color_priority, size, y, x)

    clicks = sorted(clicks, key=click_rank)[:max_clicks]
    click_to_index = {tuple(action): idx for idx, action in enumerate(clicks)}

    def latent_key(frame: list[list[int]] | None, suffix: Sequence[Sequence[int]]) -> tuple[Any, ...]:
        visible = "none" if frame is None else base_key(frame)
        history = tuple(click_to_index.get(tuple(action), -1) for action in suffix if int(action[0]) == 6)
        # Order matters for registers; keep the recent suffix and a coarse set
        # so combinations are not collapsed to the same visible state.
        return (visible, history[-8:], tuple(sorted(set(history))))

    def base_key(frame: list[list[int]]) -> str:
        return json.dumps(
            _composite_world_key(frame, modality="click_layout", available_actions=[6]),
            sort_keys=True,
            separators=(",", ":"),
        )

    def candidate_actions(suffix: Sequence[Sequence[int]], frame: list[list[int]] | None) -> list[list[int]]:
        used = [click_to_index.get(tuple(action), -1) for action in suffix if int(action[0]) == 6]
        used_set = {idx for idx in used if idx >= 0}
        actions: list[list[int]] = []
        # New targets first; then a few repeats because toggle/select games
        # often require confirming the same widget.
        for idx, click in enumerate(clicks):
            if idx not in used_set:
                actions.append(list(click))
        for idx in used[-3:]:
            if 0 <= idx < len(clicks):
                actions.append(list(clicks[idx]))
        # If the hidden selection may have unlocked directional controls, test
        # short runs sparingly without making them persistent branches.
        if suffix:
            for action in (1, 2, 3, 4):
                actions.append([action])
        seen: set[tuple[int, ...]] = set()
        out: list[list[int]] = []
        for action in actions:
            key = tuple(int(v) for v in action)
            if key in seen:
                continue
            seen.add(key)
            out.append(action)
        return out[:20]

    start_key = latent_key(start_frame, [])
    queue: list[tuple[tuple[int, int, int], list[list[int]], list[list[int]] | None]] = [((0, 0, 0), [], start_frame)]
    seen: set[tuple[Any, ...]] = {start_key}

    while queue and expansions < max_expansions:
        queue.sort(key=lambda item: item[0])
        _score, suffix, frame = queue.pop(0)
        if len(suffix) >= max_depth:
            continue
        for action in candidate_actions(suffix, frame):
            if expansions >= max_expansions:
                break
            candidate_suffix = suffix + [list(action)]
            try:
                level, win, done, next_frame, _avail = replay(candidate_suffix)
            except Exception:
                expansions += 1
                continue
            expansions += 1
            if level > start_level:
                level_hits += 1
                all_actions = prefix + candidate_suffix
                verified = _fresh_verified_level(scratch, game_id, all_actions, must_exceed=start_level)
                if verified is None:
                    rejected_hits += 1
                    continue
                verify_level, verify_win = verified
                probe_game.close()
                return {
                    "game": game_id,
                    "actions": all_actions,
                    "levels": verify_level,
                    "win": verify_win,
                    "primitive": "hidden_click_state",
                    "search_prior": "bounded_click_history_machine_state",
                    "searched_steps": len(candidate_suffix),
                    "expansions": expansions,
                    "level_hits": level_hits,
                    "rejected_hits": rejected_hits,
                    "elapsed_s": round(time.time() - start_time, 3),
                }
            if done or next_frame is None or level < start_level:
                continue
            key = latent_key(next_frame, candidate_suffix)
            if key in seen:
                continue
            seen.add(key)
            active = _small_activation_count(next_frame)
            score = (len(candidate_suffix), -active, sum(click_to_index.get(tuple(a), 0) for a in candidate_suffix) % 17)
            queue.append((score, candidate_suffix, next_frame))
            if len(queue) > beam_width:
                queue.sort(key=lambda item: item[0])
                queue = queue[:beam_width]

    stats_path = scratch / "hidden_click_state_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "primitive": "hidden_click_state",
                "game": game_id,
                "expansions": expansions,
                "states": len(seen),
                "clicks": len(clicks),
                "found_candidate": False,
                "level_hits": level_hits,
                "rejected_hits": rejected_hits,
                "max_depth": max_depth,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    probe_game.close()
    return None


def react_world_tool_candidate(
    scratch: Path,
    *,
    max_sequences: int = 192,
    max_actions: int = 96,
) -> dict[str, Any] | None:
    """Synthesize small world-representation tools and verify their actions.

    This is the deterministic shell of a ReAct rung: observe a composite public
    world, choose compact tool shapes over that representation, run those tools
    to produce action sequences, and accept only fresh sandbox-verified level-ups.
    """

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    game_id = frontier["game"]
    start_time = time.time()
    tested = 0
    rejected_hits = 0

    def replay_candidate(game: Any, suffix: Sequence[Sequence[int]]) -> tuple[int, int, bool]:
        game.reset()
        for action in prefix:
            _act(game, action)
            if bool(game.done):
                return int(game.levels), int(game.win), bool(game.done)
        for action in suffix:
            _act(game, action)
            if bool(game.done):
                break
        return int(game.levels), int(game.win), bool(game.done)

    def normalize_sequences(
        rows: Sequence[tuple[str, Sequence[Sequence[int]], str]],
    ) -> list[tuple[str, list[list[int]], str]]:
        out: list[tuple[str, list[list[int]], str]] = []
        seen: set[tuple[tuple[int, ...], ...]] = set()
        for name, actions, rationale in rows:
            seq: list[list[int]] = []
            for action in actions:
                item = [int(v) for v in action]
                if not item:
                    continue
                if item[0] == 6 and len(item) != 3:
                    continue
                if item[0] != 6 and len(item) != 1:
                    continue
                seq.append(item)
                if len(seq) >= max_actions:
                    break
            if not seq:
                continue
            seq_key = tuple(tuple(action) for action in seq)
            if seq_key in seen:
                continue
            seen.add(seq_key)
            out.append((name, seq, rationale))
            if len(out) >= max_sequences:
                break
        return out

    game = SandboxGame(game_id)
    try:
        game.reset()
        for action in prefix:
            _act(game, action)
            if bool(game.done):
                return None
        start_level = int(game.levels)
        start_frame = _frame_list(game.frame)
        modality = _world_modality(start_frame, available_actions=getattr(game, "avail", []))
        world = _composite_world_key(
            start_frame,
            modality=modality,
            available_actions=getattr(game, "avail", []),
        )
        if modality == "click_layout":
            raw_sequences = _react_click_tool_sequences(start_frame, max_actions=max_actions)
        else:
            raw_sequences = _react_navigation_tool_sequences(start_frame, max_actions=max_actions)
        sequences = normalize_sequences(raw_sequences)

        for tool_name, suffix, rationale in sequences:
            tested += 1
            try:
                level, _win, _done = replay_candidate(game, suffix)
            except Exception:
                continue
            if level <= start_level:
                continue
            all_actions = prefix + suffix
            verified = _fresh_verified_level(
                scratch,
                game_id,
                all_actions,
                must_exceed=start_level,
            )
            if verified is None:
                rejected_hits += 1
                continue
            verify_level, verify_win = verified
            return {
                "game": game_id,
                "actions": all_actions,
                "levels": verify_level,
                "win": verify_win,
                "primitive": "react_world_tool",
                "search_prior": "composite_world_tool_synthesis",
                "tool": tool_name,
                "rationale": rationale,
                "modality": modality,
                "world_mode": world.get("mode"),
                "searched_steps": len(suffix),
                "tool_sequences_tested": tested,
                "rejected_hits": rejected_hits,
                "elapsed_s": round(time.time() - start_time, 3),
            }

        (scratch / "react_world_tool_stats.json").write_text(
            json.dumps(
                {
                    "primitive": "react_world_tool",
                    "game": game_id,
                    "modality": modality,
                    "world_mode": world.get("mode"),
                    "tool_sequences": len(sequences),
                    "tool_sequences_tested": tested,
                    "found_candidate": False,
                    "rejected_hits": rejected_hits,
                    "elapsed_s": round(time.time() - start_time, 3),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return None
    finally:
        game.close()


def semiring_macro_search_candidate(
    scratch: Path,
    *,
    beam_width: int = 24,
    macro_depth: int = 5,
    max_clicks: int = 16,
    max_evaluations: int = 600,
) -> dict[str, Any] | None:
    """Beam search over Fibonacci/click macros with semiring-composed scores."""

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    prefix = action_list(frontier.get("actions", []))
    game_id = frontier["game"]
    click_cache: dict[str, list[list[int]]] = {}
    static_macros: list[list[list[int]]] = []
    seen_global: set[str] = set()
    evaluations = 0
    start_time = time.time()

    def replay_suffix(game: Any, suffix: Sequence[Sequence[int]]) -> tuple[int, int, bool, list[list[int]] | None]:
        game.reset()
        for action in prefix:
            _act(game, action)
            if bool(game.done):
                break
        for action in suffix:
            _act(game, action)
            if bool(game.done):
                break
        frame = None if game.frame is None else _frame_list(game.frame)
        return int(game.levels), int(game.win), bool(game.done), frame

    def score_transition(
        before_frame: list[list[int]],
        after_frame: list[list[int]] | None,
        *,
        level_before: int,
        level_after: int,
        done: bool,
        macro: Sequence[Sequence[int]],
        suffix_len: int,
    ) -> MacroScore:
        if after_frame is None:
            return ZERO_SCORE
        before_dist = _distance_to_goal(before_frame)
        after_dist = _distance_to_goal(after_frame)
        progress = before_dist - after_dist
        before_active = _small_activation_count(before_frame)
        after_active = _small_activation_count(after_frame)
        digest = _frame_digest(after_frame)
        return MacroScore(
            level_delta=max(0, level_after - level_before),
            alive=0 if done and level_after <= level_before else 1,
            novelty=1 if digest not in seen_global else 0,
            object_progress=int(progress),
            activation_delta=int(after_active - before_active),
            reversible_control=_macro_reversible_control(macro),
            cost=suffix_len,
        )

    game = SandboxGame(game_id)
    try:
        game.reset()
        for action in prefix:
            _act(game, action)
            if bool(game.done):
                return None
        start_level = int(game.levels)
        start_frame = _frame_list(game.frame)
        static_macros = _cold_macros(start_frame, max_clicks=max_clicks, cache=click_cache)
        seen_global.add(_frame_digest(start_frame))
        beam: list[tuple[MacroScore, list[list[int]], list[list[int]]]] = [(ONE_SCORE, [], start_frame)]

        for _round in range(macro_depth):
            candidates: list[tuple[MacroScore, list[list[int]], list[list[int]]]] = []
            for score, suffix, frame in beam:
                for macro in static_macros:
                    if evaluations >= max_evaluations:
                        break
                    candidate_suffix = suffix + [list(action) for action in macro]
                    try:
                        level, win, done, after_frame = replay_suffix(game, candidate_suffix)
                    except Exception:
                        evaluations += 1
                        continue
                    evaluations += 1
                    if level > start_level:
                        all_actions = prefix + candidate_suffix
                        verified = _fresh_verified_level(
                            scratch,
                            game_id,
                            all_actions,
                            must_exceed=start_level,
                        )
                        if verified is None:
                            continue
                        verify_level, verify_win = verified
                        return {
                            "game": game_id,
                            "actions": all_actions,
                            "levels": verify_level,
                            "win": verify_win,
                            "primitive": "semiring_macro_search",
                            "search_prior": "fibonacci_macro_semiring",
                            "searched_steps": len(candidate_suffix),
                            "macro_depth": _round + 1,
                            "evaluations": evaluations,
                            "elapsed_s": round(time.time() - start_time, 3),
                        }
                    if done or after_frame is None or level < start_level:
                        continue
                    transition_score = score_transition(
                        frame,
                        after_frame,
                        level_before=start_level,
                        level_after=level,
                        done=done,
                        macro=macro,
                        suffix_len=len(candidate_suffix),
                    )
                    if transition_score.alive <= 0:
                        continue
                    composed = score.compose(transition_score)
                    candidates.append((composed, candidate_suffix, after_frame))
                    seen_global.add(_frame_digest(after_frame))
                if evaluations >= max_evaluations:
                    break
            if not candidates:
                break
            candidates.sort(key=lambda item: item[0].rank(), reverse=True)
            next_beam: list[tuple[MacroScore, list[list[int]], list[list[int]]]] = []
            seen_round: set[tuple[int, str]] = set()
            for item in candidates:
                key = (len(item[1]), _frame_digest(item[2]))
                if key in seen_round:
                    continue
                seen_round.add(key)
                next_beam.append(item)
                if len(next_beam) >= beam_width:
                    break
            beam = next_beam
        return None
    finally:
        game.close()


def _unique_action_sequence(actions: Sequence[Sequence[int]], *, limit: int) -> list[list[int]]:
    out: list[list[int]] = []
    for action in actions:
        item = [int(v) for v in action]
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _react_click_tool_sequences(
    frame: list[list[int]],
    *,
    max_actions: int,
) -> list[tuple[str, list[list[int]], str]]:
    """Generate ReAct-style click tools from the composite component world."""

    background = _dominant_color(frame)
    components = [
        c
        for c in _component_records(frame)
        if int(c["c"]) not in (0, background) and 0 < int(c["y"]) < 63 and 1 <= int(c["n"]) <= 160
    ]
    objects = [c for c in components if int(c["n"]) >= 2]
    if not objects:
        return []

    def click(component: Mapping[str, int]) -> list[int]:
        return [6, int(component["x"]), int(component["y"])]

    def add(
        rows: list[tuple[str, list[list[int]], str]],
        name: str,
        ordered: Sequence[Mapping[str, int]],
        rationale: str,
    ) -> None:
        rows.append((name, _unique_action_sequence([click(c) for c in ordered], limit=max_actions), rationale))

    rows: list[tuple[str, list[list[int]], str]] = []
    by_y: dict[int, list[Mapping[str, int]]] = {}
    by_x: dict[int, list[Mapping[str, int]]] = {}
    by_color: dict[int, list[Mapping[str, int]]] = {}
    for component in objects:
        by_y.setdefault(int(component["y"]), []).append(component)
        by_x.setdefault(int(component["x"]), []).append(component)
        by_color.setdefault(int(component["c"]), []).append(component)

    ys = sorted(by_y)
    xs = sorted(by_x)
    colors = sorted(by_color)
    row_lr = [c for y in ys for c in sorted(by_y[y], key=lambda item: (int(item["x"]), int(item["c"]), int(item["n"])))]
    row_rl = [c for y in ys for c in sorted(by_y[y], key=lambda item: (-int(item["x"]), int(item["c"]), int(item["n"])))]
    row_bottom_lr = [
        c for y in reversed(ys) for c in sorted(by_y[y], key=lambda item: (int(item["x"]), int(item["c"]), int(item["n"])))
    ]
    row_bottom_rl = [
        c for y in reversed(ys) for c in sorted(by_y[y], key=lambda item: (-int(item["x"]), int(item["c"]), int(item["n"])))
    ]
    col_tb = [c for x in xs for c in sorted(by_x[x], key=lambda item: (int(item["y"]), int(item["c"]), int(item["n"])))]
    col_bt = [c for x in xs for c in sorted(by_x[x], key=lambda item: (-int(item["y"]), int(item["c"]), int(item["n"])))]
    col_right_tb = [
        c for x in reversed(xs) for c in sorted(by_x[x], key=lambda item: (int(item["y"]), int(item["c"]), int(item["n"])))
    ]
    col_right_bt = [
        c for x in reversed(xs) for c in sorted(by_x[x], key=lambda item: (-int(item["y"]), int(item["c"]), int(item["n"])))
    ]
    add(rows, "tool_rows_top_left", row_lr, "scan component rows from top-left")
    add(rows, "tool_rows_top_right", row_rl, "scan component rows from top-right")
    add(rows, "tool_rows_bottom_left", row_bottom_lr, "scan component rows from bottom-left")
    add(rows, "tool_rows_bottom_right", row_bottom_rl, "scan component rows from bottom-right")
    add(rows, "tool_cols_left_top", col_tb, "scan component columns from upper-left")
    add(rows, "tool_cols_left_bottom", col_bt, "scan component columns from lower-left")
    add(rows, "tool_cols_right_top", col_right_tb, "scan component columns from upper-right")
    add(rows, "tool_cols_right_bottom", col_right_bt, "scan component columns from lower-right")

    for name, ordered_colors in (
        ("tool_color_ascending", colors),
        ("tool_color_descending", list(reversed(colors))),
        ("tool_color_rare_first", sorted(colors, key=lambda color: (len(by_color[color]), color))),
        ("tool_color_common_first", sorted(colors, key=lambda color: (-len(by_color[color]), color))),
    ):
        ordered: list[Mapping[str, int]] = []
        for color in ordered_colors:
            ordered.extend(sorted(by_color[color], key=lambda item: (int(item["y"]), int(item["x"]), int(item["n"]))))
        add(rows, name, ordered, "group clicks by component color")

    outer = [
        c
        for c in objects
        if int(c["x"]) in (xs[0], xs[-1]) or int(c["y"]) in (ys[0], ys[-1])
    ]
    cx = sum(int(c["x"]) for c in objects) / max(1, len(objects))
    cy = sum(int(c["y"]) for c in objects) / max(1, len(objects))
    outer_cw = sorted(
        outer,
        key=lambda item: (
            0
            if int(item["y"]) == ys[0]
            else 1
            if int(item["x"]) == xs[-1]
            else 2
            if int(item["y"]) == ys[-1]
            else 3,
            int(item["x"]) if int(item["y"]) == ys[0] else int(item["y"]),
        ),
    )
    radial = sorted(objects, key=lambda item: (abs(int(item["x"]) - cx) + abs(int(item["y"]) - cy), int(item["y"]), int(item["x"])))
    add(rows, "tool_perimeter_clockwise", outer_cw, "click the visible component perimeter")
    add(rows, "tool_perimeter_counterclockwise", list(reversed(outer_cw)), "click the visible component perimeter in reverse")
    add(rows, "tool_radial_center_out", radial, "click components from layout center outward")
    add(rows, "tool_radial_outside_in", list(reversed(radial)), "click components from layout outside inward")

    top_markers = sorted([c for c in objects if int(c["y"]) <= min(12, ys[0] + 4)], key=lambda item: int(item["x"]))
    lower_objects = [c for c in objects if c not in top_markers]
    if top_markers and lower_objects:
        projected: list[Mapping[str, int]] = []
        for marker in top_markers:
            projected.extend(
                sorted(
                    lower_objects,
                    key=lambda item: (abs(int(item["x"]) - int(marker["x"])), int(item["y"]), int(item["x"])),
                )[:4]
            )
        add(rows, "tool_marker_projection_left", projected, "project top markers onto nearest lower components")
        add(rows, "tool_marker_projection_right", list(reversed(projected)), "project top markers onto nearest lower components in reverse")

    for color in colors:
        group = sorted(by_color[color], key=lambda item: (int(item["y"]), int(item["x"])))
        if len(group) >= 2:
            add(rows, f"tool_color_pair_{color}", [group[0], group[-1]], "test matching endpoints for a repeated color")

    return rows


def _react_navigation_tool_sequences(
    frame: list[list[int]],
    *,
    max_actions: int,
) -> list[tuple[str, list[list[int]], str]]:
    """Generate compact movement tools from perceived player/goal geometry."""

    player = _player_center(frame)
    goal = _rare_goal_center(frame, player)
    sequences: list[tuple[str, list[list[int]], str]] = []
    if player is not None and goal is not None:
        dx = goal[0] - player[0]
        dy = goal[1] - player[1]
        horizontal = [[4] if dx > 0 else [3] for _ in range(min(abs(dx), max_actions))]
        vertical = [[2] if dy > 0 else [1] for _ in range(min(abs(dy), max_actions))]
        sequences.append(("tool_nav_xy", (horizontal + vertical)[:max_actions], "move toward perceived goal on x then y"))
        sequences.append(("tool_nav_yx", (vertical + horizontal)[:max_actions], "move toward perceived goal on y then x"))
    for action in (1, 2, 3, 4):
        for length in (1, 2, 3, 5, 8, 13, 21):
            if length <= max_actions:
                sequences.append((f"tool_nav_run_{action}_{length}", [[action] for _ in range(length)], "try a Fibonacci-length movement run"))
    return sequences


def _component_records(frame: list[list[int]]) -> list[dict[str, int]]:
    records: list[dict[str, int]] = []
    colors = sorted({int(v) for row in frame for v in row})
    for color in colors:
        for cells in _component_cells(frame, color):
            xs = [p[0] for p in cells]
            ys = [p[1] for p in cells]
            records.append(
                {
                    "c": color,
                    "n": len(cells),
                    "x": round(sum(xs) / len(xs)),
                    "y": round(sum(ys) / len(ys)),
                }
            )
    records.sort(key=lambda c: (c["n"], c["c"], c["y"], c["x"]))
    return records


def _rare_goal_center(frame: list[list[int]], origin: tuple[int, int] | None) -> tuple[int, int] | None:
    priority = (14, 8, 10, 11, 12, 13, 15)
    for color in priority:
        center = _closest_component_center(frame, {color}, origin)
        if center is not None:
            return center
    return None


def _player_center(frame: list[list[int]]) -> tuple[int, int] | None:
    return _largest_component_center(frame, {4, 9})


def _distance_to_goal(frame: list[list[int]]) -> int:
    player = _player_center(frame)
    goal = _rare_goal_center(frame, player)
    if player is None or goal is None:
        return 0
    return abs(player[0] - goal[0]) + abs(player[1] - goal[1])


def _small_activation_count(frame: list[list[int]]) -> int:
    return sum(
        1
        for component in _component_records(frame)
        if 1 <= int(component["n"]) <= 16 and int(component["c"]) not in (0, 2, 4, 5, 6)
    )


def _macro_reversible_control(macro: Sequence[Sequence[int]]) -> int:
    values = [int(action[0]) for action in macro if len(action) == 1]
    pairs = {(1, 2), (2, 1), (3, 4), (4, 3)}
    return sum(1 for a, b in zip(values, values[1:]) if (a, b) in pairs)


def _frame_click_actions(
    frame: Any,
    *,
    max_clicks: int = 24,
    cache: dict[str, list[list[int]]] | None = None,
) -> list[list[int]]:
    if frame is None:
        return []
    frame_rows = _frame_list(frame)
    cache_key = _frame_digest(frame_rows)
    if cache is not None and cache_key in cache:
        return [list(a) for a in cache[cache_key]]
    rows: list[tuple[int, int, int, int, list[int]]] = []
    for component in _component_records(frame_rows):
        color = int(component["c"])
        size = int(component["n"])
        x = int(component["x"])
        y = int(component["y"])
        if y <= 0 or y >= 63 or size <= 0 or size > 144:
            continue
        priority = 0 if color in (8, 9, 10, 11, 12, 13, 14, 15) else 1
        rows.append((priority, size, color, y, [6, x, y]))
    rows.sort()
    seen: set[tuple[int, ...]] = set()
    out: list[list[int]] = []
    for *_prefix, action in rows:
        key = tuple(int(x) for x in action)
        if key in seen:
            continue
        seen.add(key)
        out.append(action)
        if len(out) >= max_clicks:
            break
    if cache is not None:
        cache[cache_key] = [list(a) for a in out]
    return out


def _zeckendorf_parts(n: int, fibs: Sequence[int] = (13, 8, 5, 3, 2, 1)) -> tuple[int, ...]:
    parts: list[int] = []
    remaining = int(n)
    last_idx = -10
    ordered = list(fibs)
    for idx, fib in enumerate(ordered):
        if fib <= remaining and abs(idx - last_idx) > 1:
            parts.append(int(fib))
            remaining -= int(fib)
            last_idx = idx
        if remaining <= 0:
            break
    return tuple(parts) if remaining == 0 else (int(n),)


def _fibonacci_direction_macros() -> list[list[list[int]]]:
    lengths = sorted({1, 2, 3, 5, 8, 13} | {4, 6, 7, 9, 10, 11, 12, 14, 16})
    macros: list[list[list[int]]] = []
    for action in (1, 2, 3, 4):
        for length in lengths:
            macros.append([[action] for _ in range(length)])
            parts = _zeckendorf_parts(length)
            if len(parts) > 1:
                macro: list[list[int]] = []
                for part in parts:
                    macro.extend([[action] for _ in range(part)])
                macros.append(macro)
    for first, second in ((1, 2), (2, 1), (3, 4), (4, 3)):
        for length in (2, 3, 5, 8):
            macro = []
            for idx in range(length):
                macro.append([first if idx % 2 == 0 else second])
            macros.append(macro)
        for total in (6, 9, 11):
            macro = []
            for part in _zeckendorf_parts(total):
                pair = (first, second) if len(macro) % 2 == 0 else (second, first)
                for idx in range(part):
                    macro.append([pair[idx % 2]])
            macros.append(macro)
    return macros


def _cold_macros(
    frame: Any,
    *,
    max_clicks: int,
    cache: dict[str, list[list[int]]] | None = None,
) -> list[list[list[int]]]:
    macros = _fibonacci_direction_macros()
    macros.extend([[[5]], [[7]]])
    clicks = _frame_click_actions(frame, max_clicks=max_clicks, cache=cache)
    macros.extend([[action] for action in clicks])
    for click in clicks[:4]:
        for direction in (1, 2, 3, 4):
            for length in (3, 5):
                run = [[direction] for _ in range(length)]
                macros.append([click] + run)
                macros.append(run + [click])
            for total in (6, 9):
                run = [[direction] for part in _zeckendorf_parts(total) for _ in range(part)]
                macros.append([click] + run + [click])
    seen: set[tuple[tuple[int, ...], ...]] = set()
    unique: list[list[list[int]]] = []
    for macro in macros:
        key = tuple(tuple(int(v) for v in action) for action in macro)
        if key in seen:
            continue
        seen.add(key)
        unique.append(macro)
    return unique


def sourcefree_primitive_candidates(
    scratch: Path,
    *,
    include_expensive: bool = False,
    include_cold_search: bool = False,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    center = _run_primitive_safely(center_corridor_candidate, scratch)
    if center is not None:
        candidates.append(center)
        return candidates
    lattice = _run_primitive_safely(lattice_corridor_candidate, scratch)
    if lattice is not None:
        candidates.append(lattice)
        return candidates
    if include_expensive:
        detour = _run_primitive_safely(simple_path_detour_candidate, scratch)
        if detour is not None:
            candidates.append(detour)
            return candidates
    if include_cold_search:
        first_level = _run_primitive_safely(first_level_macro_tournament_candidate, scratch)
        if first_level is not None:
            candidates.append(first_level)
            return candidates
        click_only = _frontier_is_click_only(scratch)
        if click_only:
            archive = _run_primitive_safely(go_explore_archive_candidate, scratch)
            if archive is not None:
                candidates.append(archive)
                return candidates
            react_tool = _run_primitive_safely(react_world_tool_candidate, scratch)
            if react_tool is not None:
                candidates.append(react_tool)
                return candidates
            if include_expensive:
                hidden_click = _run_primitive_safely(hidden_click_state_candidate, scratch)
                if hidden_click is not None:
                    candidates.append(hidden_click)
                    return candidates
            return candidates
        discriminator = _run_primitive_safely(code_world_discriminator_candidate, scratch)
        if discriminator is not None:
            candidates.append(discriminator)
            return candidates
        react_tool = _run_primitive_safely(react_world_tool_candidate, scratch)
        if react_tool is not None:
            candidates.append(react_tool)
            return candidates
        if not click_only:
            explore = _run_primitive_safely(sandbox_frontier_explore_candidate, scratch)
            if explore is not None:
                candidates.append(explore)
                return candidates
            return candidates
        explore = _run_primitive_safely(sandbox_frontier_explore_candidate, scratch)
        if explore is not None:
            candidates.append(explore)
            return candidates
    return candidates


def _frontier_is_click_only(scratch: Path) -> bool:
    try:
        sys.path.insert(0, str(scratch))
        from arc3_sandbox import SandboxGame  # type: ignore

        frontier = json.loads((scratch / "frontier.json").read_text())
        prefix = action_list(frontier.get("actions", []))
        game = SandboxGame(str(frontier["game"]))
        try:
            game.reset()
            for action in prefix:
                _act(game, action)
                if bool(game.done):
                    break
            return set(int(a) for a in getattr(game, "avail", [])) == {6}
        finally:
            game.close()
    except Exception:
        return False


def _run_primitive_safely(fn: Any, scratch: Path) -> dict[str, Any] | None:
    try:
        return fn(scratch)
    except Exception as exc:
        failures_path = scratch / "primitive_failures.jsonl"
        with failures_path.open("a") as f:
            f.write(json.dumps({"primitive": getattr(fn, "__name__", str(fn)), "error": str(exc)[:500]}) + "\n")
        return None
