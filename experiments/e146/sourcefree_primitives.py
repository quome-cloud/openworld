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
                    macro = rng.choice(untried)
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
                            "search_prior": "fibonacci_direction_macros",
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


def _fibonacci_direction_macros() -> list[list[list[int]]]:
    lengths = sorted({1, 2, 3, 5, 8, 13} | {4, 6, 7, 12, 14})
    macros: list[list[list[int]]] = []
    for action in (1, 2, 3, 4):
        for length in lengths:
            macros.append([[action] for _ in range(length)])
    return macros


def _cold_macros(
    frame: Any,
    *,
    max_clicks: int,
    cache: dict[str, list[list[int]]] | None = None,
) -> list[list[list[int]]]:
    macros = _fibonacci_direction_macros()
    macros.extend([[[5]], [[7]]])
    macros.extend([[action] for action in _frame_click_actions(frame, max_clicks=max_clicks, cache=cache)])
    return macros


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
        explore = _run_primitive_safely(sandbox_frontier_explore_candidate, scratch)
        if explore is not None:
            candidates.append(explore)
            return candidates
        semiring = _run_primitive_safely(semiring_macro_search_candidate, scratch)
        if semiring is not None:
            candidates.append(semiring)
            return candidates
    return candidates


def _run_primitive_safely(fn: Any, scratch: Path) -> dict[str, Any] | None:
    try:
        return fn(scratch)
    except Exception as exc:
        failures_path = scratch / "primitive_failures.jsonl"
        with failures_path.open("a") as f:
            f.write(json.dumps({"primitive": getattr(fn, "__name__", str(fn)), "error": str(exc)[:500]}) + "\n")
        return None
