"""Source-free transition graph mining for ARC probe runs.

E143 turns failed proposal traces into state-transition evidence. It records
frame deltas, identifies critical states, forks the public sandbox by replaying
to those states, probes small local macros, and writes a graph that an agent can
inspect as an OpenWorld-style world.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


Action = list[int]
Frame = list[list[int]]


@dataclass(frozen=True)
class ProbeMacro:
    macro_id: str
    actions: tuple[tuple[int, ...], ...]
    description: str


DEFAULT_MACROS: tuple[ProbeMacro, ...] = (
    ProbeMacro("wait", ((5,),), "single wait/action5 tick"),
    ProbeMacro("noop_alt_lr", ((3,), (4,)), "left/right parity cycle"),
    ProbeMacro("noop_alt_ud", ((1,), (2,)), "up/down parity cycle"),
    ProbeMacro("scan_cardinals", ((1,), (2,), (3,), (4,)), "short directional scan"),
    ProbeMacro("scan_actions", ((1,), (2,), (3,), (4,), (5,), (7,)), "all non-click actions once"),
)


def norm_action(raw: Any) -> tuple[int, ...] | None:
    if isinstance(raw, int):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        return None
    try:
        action = tuple(int(x) for x in raw)
    except Exception:
        return None
    if action[0] == 6:
        if len(action) == 3 and 0 <= action[1] <= 63 and 0 <= action[2] <= 63:
            return action
        return None
    if len(action) == 1 and action[0] in (1, 2, 3, 4, 5, 7):
        return action
    return None


def action_list(actions: Iterable[Sequence[int]]) -> list[Action]:
    out: list[Action] = []
    for raw in actions:
        action = norm_action(list(raw))
        if action is not None:
            out.append(list(action))
    return out


def frame_hash(frame: Frame | None) -> str:
    if frame is None:
        return "none"
    h = hashlib.sha1()
    for row in frame:
        h.update(bytes(int(x) & 0xFF for x in row))
    return h.hexdigest()[:16]


def palette_counts(frame: Frame | None) -> dict[str, int]:
    if frame is None:
        return {}
    counts: Counter[int] = Counter()
    for row in frame:
        counts.update(int(x) for x in row)
    return {str(k): counts[k] for k in sorted(counts)}


def _neighbors(x: int, y: int, w: int, h: int) -> Iterable[tuple[int, int]]:
    if x > 0:
        yield x - 1, y
    if x + 1 < w:
        yield x + 1, y
    if y > 0:
        yield x, y - 1
    if y + 1 < h:
        yield x, y + 1


def connected_components(frame: Frame | None, *, max_components: int = 96) -> list[dict[str, int]]:
    if frame is None:
        return []
    h = len(frame)
    w = len(frame[0]) if h else 0
    seen: set[tuple[int, int]] = set()
    comps: list[dict[str, int]] = []
    for y in range(h):
        for x in range(w):
            if (x, y) in seen:
                continue
            color = int(frame[y][x])
            seen.add((x, y))
            q: deque[tuple[int, int]] = deque([(x, y)])
            cells: list[tuple[int, int]] = []
            while q:
                cx, cy = q.popleft()
                cells.append((cx, cy))
                for nx, ny in _neighbors(cx, cy, w, h):
                    if (nx, ny) not in seen and int(frame[ny][nx]) == color:
                        seen.add((nx, ny))
                        q.append((nx, ny))
            xs = [p[0] for p in cells]
            ys = [p[1] for p in cells]
            comps.append(
                {
                    "c": color,
                    "n": len(cells),
                    "x": round(sum(xs) / len(xs)),
                    "y": round(sum(ys) / len(ys)),
                    "min_x": min(xs),
                    "min_y": min(ys),
                    "max_x": max(xs),
                    "max_y": max(ys),
                }
            )
    comps.sort(key=lambda c: (c["n"], c["c"], c["y"], c["x"]))
    return comps[:max_components]


def frame_summary(frame: Frame | None, *, levels: int = 0, win: int = 0, done: bool = False) -> dict[str, Any]:
    comps = connected_components(frame)
    small = [c for c in comps if c["n"] <= 16]
    return {
        "hash": frame_hash(frame),
        "levels": int(levels),
        "win": int(win),
        "done": bool(done),
        "palette": palette_counts(frame),
        "components": comps[:32],
        "small": small[:32],
    }


def component_key(component: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(component.get("c", -1)),
        int(component.get("n", -1)),
        int(component.get("x", -1)),
        int(component.get("y", -1)),
    )


def transition_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    bpal = before.get("palette") if isinstance(before.get("palette"), Mapping) else {}
    apal = after.get("palette") if isinstance(after.get("palette"), Mapping) else {}
    colors = sorted({str(k) for k in bpal} | {str(k) for k in apal}, key=lambda x: int(x) if x.isdigit() else x)
    palette_delta = {
        color: int(apal.get(color, 0)) - int(bpal.get(color, 0))
        for color in colors
        if int(apal.get(color, 0)) != int(bpal.get(color, 0))
    }
    before_comps = {component_key(c) for c in before.get("small", []) if isinstance(c, Mapping)}
    after_comps = {component_key(c) for c in after.get("small", []) if isinstance(c, Mapping)}
    changed_cells = sum(abs(v) for v in palette_delta.values()) // 2
    return {
        "level_delta": int(after.get("levels", 0)) - int(before.get("levels", 0)),
        "win_delta": int(after.get("win", 0)) - int(before.get("win", 0)),
        "done_delta": bool(after.get("done")) and not bool(before.get("done")),
        "hash_changed": before.get("hash") != after.get("hash"),
        "changed_cells_l1_half": changed_cells,
        "palette_delta": palette_delta,
        "small_appeared": [list(x) for x in sorted(after_comps - before_comps)],
        "small_disappeared": [list(x) for x in sorted(before_comps - after_comps)],
    }


def is_critical(delta: Mapping[str, Any], *, min_changed_cells: int = 1) -> bool:
    return bool(
        int(delta.get("level_delta", 0)) != 0
        or int(delta.get("win_delta", 0)) != 0
        or delta.get("done_delta")
        or int(delta.get("changed_cells_l1_half", 0)) >= min_changed_cells
        or delta.get("small_appeared")
        or delta.get("small_disappeared")
    )


def _act(game: Any, action: Sequence[int]) -> None:
    if int(action[0]) == 6:
        game.step(6, int(action[1]), int(action[2]))
    else:
        game.step(int(action[0]))


def _game_summary(game: Any) -> dict[str, Any]:
    return frame_summary(
        game.frame.tolist() if hasattr(game.frame, "tolist") else game.frame,
        levels=int(game.levels),
        win=int(game.win),
        done=bool(game.done),
    )


def record_trace(scratch: Path, proposal_path: Path | None = None) -> dict[str, Any]:
    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    frontier_actions = action_list(frontier.get("actions", []))
    actions = list(frontier_actions)
    proposal: Mapping[str, Any] = {}
    if proposal_path is not None:
        proposal = json.loads(proposal_path.read_text())
        actions.extend(action_list(proposal.get("probe_plan", [])))

    game = SandboxGame(frontier["game"])
    try:
        game.reset()
        states = [_game_summary(game)]
        edges: list[dict[str, Any]] = []
        for i, action in enumerate(actions):
            before = states[-1]
            _act(game, action)
            after = _game_summary(game)
            delta = transition_delta(before, after)
            edge = {
                "edge_id": f"trace-{i:04d}",
                "step": i,
                "action": list(action),
                "from": before["hash"],
                "to": after["hash"],
                "from_level": before["levels"],
                "to_level": after["levels"],
                "delta": delta,
                "critical": is_critical(delta),
            }
            edges.append(edge)
            states.append(after)
            if after["done"] and i + 1 >= len(frontier_actions):
                break
        return {
            "game": frontier["game"],
            "proposal_id": proposal.get("proposal_id"),
            "frontier_action_count": len(frontier_actions),
            "actions": actions[: len(edges)],
            "states": states,
            "edges": edges,
            "critical_steps": [e["step"] for e in edges if e["critical"]],
        }
    finally:
        game.close()


def _replay_to(game: Any, actions: Sequence[Sequence[int]], step: int) -> None:
    game.reset()
    for action in actions[:step]:
        _act(game, action)


def mine_probe_graph(
    scratch: Path,
    trace: Mapping[str, Any],
    *,
    macros: Sequence[ProbeMacro] = DEFAULT_MACROS,
    max_states: int = 12,
) -> dict[str, Any]:
    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    actions = action_list(trace.get("actions", []))
    frontier_action_count = int(trace.get("frontier_action_count", 0))
    critical_steps = list(trace.get("critical_steps", []))[-max_states:]
    game = SandboxGame(str(trace["game"]))
    nodes: dict[str, Mapping[str, Any]] = {}
    probe_edges: list[dict[str, Any]] = []
    try:
        for step in critical_steps:
            _replay_to(game, actions, int(step))
            before = _game_summary(game)
            nodes.setdefault(before["hash"], before)
            for macro in macros:
                _replay_to(game, actions, int(step))
                before = _game_summary(game)
                for action in macro.actions:
                    _act(game, action)
                    if game.done:
                        break
                after = _game_summary(game)
                nodes.setdefault(after["hash"], after)
                delta = transition_delta(before, after)
                probe_edges.append(
                    {
                        "edge_id": f"probe-{step:04d}-{macro.macro_id}",
                        "kind": "probe",
                        "step": int(step),
                        "macro_id": macro.macro_id,
                        "actions": [list(a) for a in macro.actions],
                        "from": before["hash"],
                        "to": after["hash"],
                        "from_level": before["levels"],
                        "to_level": after["levels"],
                        "delta": delta,
                        "critical": is_critical(delta),
                    }
                )
        return {"nodes": list(nodes.values()), "edges": probe_edges}
    finally:
        game.close()


def build_transition_graph(trace: Mapping[str, Any], probe_graph: Mapping[str, Any] | None = None) -> dict[str, Any]:
    nodes: dict[str, Mapping[str, Any]] = {}
    for state in trace.get("states", []):
        if isinstance(state, Mapping):
            nodes[str(state["hash"])] = state
    edges = list(trace.get("edges", []))
    if probe_graph:
        for state in probe_graph.get("nodes", []):
            if isinstance(state, Mapping):
                nodes[str(state["hash"])] = state
        edges.extend(list(probe_graph.get("edges", [])))
    return {
        "experiment": "E143",
        "game": trace.get("game"),
        "proposal_id": trace.get("proposal_id"),
        "nodes": list(nodes.values()),
        "edges": edges,
        "critical_edges": [e for e in edges if isinstance(e, Mapping) and e.get("critical")],
    }


def build_introspection_world(graph: Mapping[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    cards = []
    for edge in edges:
        if not isinstance(edge, Mapping) or not edge.get("critical"):
            continue
        delta = edge.get("delta", {})
        cards.append(
            {
                "id": edge.get("edge_id"),
                "title": f"{edge.get('kind', 'trace')} step {edge.get('step')}",
                "from": edge.get("from"),
                "to": edge.get("to"),
                "from_level": edge.get("from_level"),
                "to_level": edge.get("to_level"),
                "action": edge.get("action", edge.get("macro_id")),
                "salience": int(delta.get("level_delta", 0)) * 100
                + int(delta.get("changed_cells_l1_half", 0))
                + 5 * len(delta.get("small_appeared", []) or [])
                + 5 * len(delta.get("small_disappeared", []) or []),
                "observed_delta": delta,
            }
        )
    cards.sort(key=lambda x: (-int(x["salience"]), str(x["id"])))
    from_levels = [int(c["from_level"]) for c in cards if c.get("from_level") is not None]
    frontier_level = max(from_levels) if from_levels else None
    frontier_cards = [c for c in cards if c.get("from_level") == frontier_level]
    frontier_cards.sort(key=lambda x: (-int(x["salience"]), str(x["id"])))
    return {
        "world_id": f"transition-introspection-{graph.get('game')}",
        "description": "An OpenWorld-style inspection world whose objects are observed state transitions.",
        "rules": [
            "Nodes are source-free frame summaries.",
            "Edges are replayed actions or forked local probe macros.",
            "High-salience cards are the next states an agent should explain or compose.",
        ],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "frontier_level": frontier_level,
        "cards": cards[:64],
        "frontier_cards": frontier_cards[:64],
    }


def _clicks_from_delta(delta: Mapping[str, Any], *, max_clicks: int = 4) -> list[Action]:
    targets: list[tuple[int, int, int, int]] = []
    for key in ("small_appeared", "small_disappeared"):
        for raw in delta.get(key, []) or []:
            if not isinstance(raw, list) or len(raw) < 4:
                continue
            color, size, x, y = (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
            if color in (4, 12, 13, 14) and 0 <= x <= 63 and 0 <= y <= 63:
                targets.append((size, color, x, y))
    targets.sort()
    return [[6, x, y] for _, _, x, y in targets[:max_clicks]]


def _card_step(card: Mapping[str, Any]) -> int | None:
    cid = str(card.get("id", ""))
    parts = cid.split("-")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    title = str(card.get("title", ""))
    for part in reversed(title.split()):
        if part.isdigit():
            return int(part)
    return None


def card_local_motifs(card: Mapping[str, Any]) -> list[tuple[str, list[Action]]]:
    delta = card.get("observed_delta", {})
    action = norm_action(card.get("action"))
    observed = [list(action)] if action is not None else []
    clicks = _clicks_from_delta(delta)
    motifs = [
        ("repeat", observed * 3),
        ("clicks", observed + clicks + [[5]]),
        ("scan-clicks", observed + [[1], [2], [3], [4]] + clicks),
        ("parity-clicks", observed + [[3], [4], [1], [2]] + clicks),
    ]
    return [(name, actions) for name, actions in motifs if actions]


def card_seed_motifs(
    trace: Mapping[str, Any],
    card: Mapping[str, Any],
) -> list[tuple[str, list[Action]]]:
    actions = action_list(trace.get("actions", []))
    frontier_action_count = int(trace.get("frontier_action_count", 0))
    step = _card_step(card)
    if step is None or step < frontier_action_count or step >= len(actions):
        return []
    prefix = actions[frontier_action_count : step + 1]
    return [(name, prefix + motif) for name, motif in card_local_motifs(card)]


def generate_graph_proposals(
    trace: Mapping[str, Any],
    world: Mapping[str, Any],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Emit concrete probes from high-salience frontier transition cards."""

    actions = action_list(trace.get("actions", []))
    frontier_action_count = int(trace.get("frontier_action_count", 0))
    proposals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in world.get("frontier_cards", [])[:limit]:
        if not isinstance(card, Mapping):
            continue
        step = _card_step(card)
        if step is None:
            continue
        if step < 0 or step >= len(actions):
            continue
        delta = card.get("observed_delta", {})
        suffixes = card_seed_motifs(trace, card)
        for label, suffix in suffixes:
            pid = f"e143-{label}-{step:04d}"
            if pid in seen:
                continue
            seen.add(pid)
            proposals.append(
                {
                    "proposal_id": pid,
                    "schema_id": "transition-graph introspection",
                    "goal_schema_id": "compose observed late-level transition into level-up",
                    "hypothesis": (
                        "Replay to a high-salience frontier transition, then either repeat the "
                        "causal action or select components that appeared/disappeared in that transition."
                    ),
                    "role_bindings": {
                        "source_card": str(card.get("id")),
                        "from_level": str(card.get("from_level")),
                        "to_level": str(card.get("to_level")),
                        "observed_delta": json.dumps(delta, sort_keys=True)[:2000],
                    },
                    "probe_plan": actions[frontier_action_count : step + 1] + suffix,
                    "expected_deltas": [
                        "reproduces a real high-salience transition from the transition graph",
                        "continues or selects the observed changed components",
                        "successful composition should increase level or expose a new critical transition",
                    ],
                    "confidence": 0.07,
                }
            )
    return proposals


def delta_score(delta: Mapping[str, Any]) -> float:
    palette = delta.get("palette_delta", {}) if isinstance(delta.get("palette_delta"), Mapping) else {}
    c4_drop = max(0, -int(palette.get("4", 0)))
    rare_motion = sum(abs(int(v)) for k, v in palette.items() if str(k) in {"4", "12", "13", "14"})
    return (
        10000.0 * int(delta.get("level_delta", 0))
        + 5000.0 * int(delta.get("win_delta", 0))
        + 12.0 * c4_drop
        + 0.25 * rare_motion
        + 0.1 * int(delta.get("changed_cells_l1_half", 0))
        + 2.0 * len(delta.get("small_appeared", []) or [])
        + 2.0 * len(delta.get("small_disappeared", []) or [])
        - 250.0 * int(bool(delta.get("done_delta")))
    )


def evaluate_suffix(scratch: Path, suffix: Sequence[Sequence[int]]) -> dict[str, Any]:
    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    frontier = json.loads((scratch / "frontier.json").read_text())
    frontier_actions = action_list(frontier.get("actions", []))
    suffix_actions = action_list(suffix)
    game = SandboxGame(frontier["game"])
    try:
        game.reset()
        for action in frontier_actions:
            _act(game, action)
        start = _game_summary(game)
        used: list[Action] = []
        for action in suffix_actions:
            _act(game, action)
            used.append(action)
            if int(game.levels) > int(start["levels"]) or bool(game.done):
                break
        end = _game_summary(game)
        delta = transition_delta(start, end)
        return {
            "game": frontier["game"],
            "start": start,
            "end": end,
            "delta": delta,
            "score": delta_score(delta) - 0.02 * len(used),
            "actions": used,
            "level_up": int(end["levels"]) > int(start["levels"]),
            "done": bool(end["done"]),
        }
    finally:
        game.close()


def beam_search_transition_cards(
    scratch: Path,
    trace: Mapping[str, Any],
    world: Mapping[str, Any],
    *,
    beam_width: int = 6,
    depth: int = 2,
    card_limit: int = 8,
) -> dict[str, Any]:
    cards = [c for c in world.get("frontier_cards", []) if isinstance(c, Mapping)][:card_limit]
    local_motifs = []
    for card in cards:
        for name, motif in card_local_motifs(card):
            local_motifs.append((str(card.get("id")), name, motif))
    seeds: list[dict[str, Any]] = []
    for card in cards:
        for name, actions in card_seed_motifs(trace, card):
            seeds.append({"path": [f"{card.get('id')}:{name}"], "actions": actions})

    evaluated: list[dict[str, Any]] = []
    frontier = seeds
    seen: set[tuple[tuple[int, ...], ...]] = set()
    solved: dict[str, Any] | None = None
    for d in range(max(1, depth)):
        layer: list[dict[str, Any]] = []
        for candidate in frontier:
            key = tuple(tuple(a) for a in candidate["actions"])
            if key in seen:
                continue
            seen.add(key)
            result = evaluate_suffix(scratch, candidate["actions"])
            row = {
                "depth": d + 1,
                "path": candidate["path"],
                "actions": candidate["actions"],
                "steps": len(result["actions"]),
                "score": round(float(result["score"]), 6),
                "level_up": result["level_up"],
                "done": result["done"],
                "end": result["end"],
                "delta": result["delta"],
            }
            evaluated.append(row)
            layer.append(row)
            if result["level_up"]:
                solved = row
                break
        if solved or d + 1 >= depth:
            break
        layer.sort(key=lambda x: (-float(x["score"]), len(x["actions"])))
        next_frontier: list[dict[str, Any]] = []
        for parent in layer[:beam_width]:
            for card_id, name, motif in local_motifs:
                next_frontier.append(
                    {
                        "path": list(parent["path"]) + [f"{card_id}:{name}"],
                        "actions": list(parent["actions"]) + motif,
                    }
                )
        frontier = next_frontier[: beam_width * max(1, len(local_motifs))]

    evaluated.sort(key=lambda x: (-float(x["score"]), len(x["actions"])))
    return {
        "experiment": "E143-beam",
        "game": trace.get("game"),
        "beam_width": beam_width,
        "depth": depth,
        "card_limit": card_limit,
        "solved": solved,
        "evaluated": evaluated,
        "ranked": evaluated[: max(beam_width, 12)],
    }


def write_beam_artifacts(out_dir: Path, beam: Mapping[str, Any]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "beam_search.json"]
    paths[0].write_text(json.dumps(beam, indent=2) + "\n")
    for idx, row in enumerate(beam.get("ranked", [])[:12], start=1):
        if not isinstance(row, Mapping):
            continue
        pid = f"e143-beam-{idx:02d}"
        proposal = {
            "proposal_id": pid,
            "schema_id": "transition-card beam search",
            "goal_schema_id": "compose observed transition cards into level-up",
            "hypothesis": "This suffix survived E143 beam scoring against real source-free frame deltas.",
            "role_bindings": {
                "beam_path": json.dumps(row.get("path", [])),
                "beam_score": str(row.get("score")),
                "observed_delta": json.dumps(row.get("delta", {}), sort_keys=True)[:2000],
            },
            "probe_plan": row.get("actions", []),
            "expected_deltas": [
                "should reproduce a high-scoring composed transition path",
                "if not solved, its counterexample should be fed back as another transition trace",
            ],
            "confidence": 0.08,
        }
        path = out_dir / f"proposal_{pid}.json"
        path.write_text(json.dumps(proposal, indent=2) + "\n")
        paths.append(path)
    return paths


def _frontier_level_from_trace(trace: Mapping[str, Any]) -> int:
    states = trace.get("states", [])
    if isinstance(states, list) and states:
        last = states[-1]
        if isinstance(last, Mapping):
            try:
                return int(last.get("levels", 0))
            except Exception:
                return 0
    return 0


def signature_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Distance between source-free state summaries.

    Lower is better. The metric intentionally uses only public frame-derived
    summaries: level, palette counts, and small component signatures.
    """

    score = 0.0
    score += 10000.0 * abs(int(a.get("levels", 0)) - int(b.get("levels", 0)))
    apal = a.get("palette") if isinstance(a.get("palette"), Mapping) else {}
    bpal = b.get("palette") if isinstance(b.get("palette"), Mapping) else {}
    colors = {str(k) for k in apal} | {str(k) for k in bpal}
    score += 0.05 * sum(abs(int(apal.get(c, 0)) - int(bpal.get(c, 0))) for c in colors)

    acomp = {component_key(c) for c in a.get("small", []) if isinstance(c, Mapping)}
    bcomp = {component_key(c) for c in b.get("small", []) if isinstance(c, Mapping)}
    score += 5.0 * len(acomp.symmetric_difference(bcomp))

    # Exact frame hash match is strong evidence of state equivalence.
    if a.get("hash") == b.get("hash"):
        score -= 100.0
    return round(score, 6)


def _trace_frontier_summary(trace: Mapping[str, Any]) -> Mapping[str, Any] | None:
    states = trace.get("states")
    if isinstance(states, list) and states and isinstance(states[-1], Mapping):
        return states[-1]
    return None


def rank_behavioral_suffixes_by_signature(
    scratch: Path,
    trace: Mapping[str, Any],
    suffixes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Replay prior traces to suffix starts and rank by state compatibility."""

    target = _trace_frontier_summary(trace)
    if target is None:
        return [dict(s) for s in suffixes]

    sys.path.insert(0, str(scratch))
    from arc3_sandbox import SandboxGame  # type: ignore

    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for suffix in suffixes:
        by_source.setdefault(str(suffix.get("source_path")), []).append(suffix)

    ranked: list[dict[str, Any]] = []
    for source_path, rows in by_source.items():
        try:
            source = json.loads(Path(source_path).read_text())
        except Exception:
            continue
        if str(source.get("game")) != str(trace.get("game")):
            continue
        actions = action_list(source.get("actions", []))
        starts = sorted({int(r.get("start_index", 0)) for r in rows})
        game = SandboxGame(str(source.get("game")))
        try:
            game.reset()
            cursor = 0
            summaries: dict[int, Mapping[str, Any]] = {0: _game_summary(game)}
            for start in starts:
                if start < cursor:
                    game.reset()
                    cursor = 0
                while cursor < start and cursor < len(actions):
                    _act(game, actions[cursor])
                    cursor += 1
                summaries[start] = _game_summary(game)
            for row in rows:
                start = int(row.get("start_index", 0))
                summary = summaries.get(start)
                if summary is None:
                    continue
                enriched = dict(row)
                enriched["source_summary"] = summary
                enriched["signature_distance"] = signature_distance(target, summary)
                ranked.append(enriched)
        finally:
            game.close()

    ranked.sort(
        key=lambda x: (
            int(float(x.get("signature_distance", float("inf"))) // 25.0),
            -len(x.get("suffix", [])),
            float(x.get("signature_distance", float("inf"))),
        )
    )
    return ranked


def extract_behavioral_suffixes(
    trace: Mapping[str, Any],
    solution_paths: Sequence[str | Path],
    *,
    max_suffixes: int = 12,
) -> list[dict[str, Any]]:
    """Extract prior behavioral suffixes that start at the current frontier level.

    These are still source-free: they are action traces, not game code. They are
    useful when a previous run has reached a later level through a different
    prefix and we want to test whether the level-local policy transfers.
    """

    game = str(trace.get("game"))
    frontier_level = _frontier_level_from_trace(trace)
    frontier_action_count = int(trace.get("frontier_action_count", 0))
    out: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for raw_path in solution_paths:
        path = Path(raw_path)
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if str(data.get("game")) != game:
            continue
        try:
            final_level = int(data.get("levels", 0))
        except Exception:
            continue
        if final_level <= frontier_level:
            continue
        actions = action_list(data.get("actions", []))
        if not actions:
            continue
        # Prefer metadata-free cut points known from the recorded trace length:
        # emit every suffix long enough to plausibly contain the next level-up.
        # A verifier will execute and keep only transferring suffixes.
        starts = sorted(
            {
                max(0, len(actions) - suffix_len)
                for suffix_len in (220, 200, 180, 160, 150, 145, 140, 135, 130, 125, 120, 100)
            }
            | {
                start
                for start in (
                    frontier_action_count - 20,
                    frontier_action_count - 10,
                    frontier_action_count,
                    frontier_action_count + 10,
                    frontier_action_count + 20,
                )
                if 0 <= start < len(actions)
            }
        )
        for start in starts:
            suffix = actions[start:]
            key = tuple(tuple(a) for a in suffix)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "source_path": str(path),
                    "source_levels": final_level,
                    "frontier_level": frontier_level,
                    "start_index": start,
                    "suffix": suffix,
                }
            )
            if len(out) >= max_suffixes:
                return out
    return out


def write_transfer_proposals(out_dir: Path, trace: Mapping[str, Any], suffixes: Sequence[Mapping[str, Any]]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, suffix in enumerate(suffixes, start=1):
        pid = f"e143-transfer-{idx:02d}"
        proposal = {
            "proposal_id": pid,
            "schema_id": "behavioral-trace transfer",
            "goal_schema_id": "transfer prior source-free level-local suffix",
            "hypothesis": (
                "A previous behavioral run reached a later level. Test the suffix "
                "from that run on the current frontier state."
            ),
            "role_bindings": {
                "source_path": str(suffix.get("source_path")),
                "source_levels": str(suffix.get("source_levels")),
                "frontier_level": str(suffix.get("frontier_level")),
                "start_index": str(suffix.get("start_index")),
                "signature_distance": str(suffix.get("signature_distance", "")),
            },
            "probe_plan": suffix.get("suffix", []),
            "expected_deltas": ["level increases from the current frontier"],
            "confidence": 0.12,
        }
        path = out_dir / f"proposal_{pid}.json"
        path.write_text(json.dumps(proposal, indent=2) + "\n")
        paths.append(path)
    return paths


def write_graph_proposals(out_dir: Path, trace: Mapping[str, Any], world: Mapping[str, Any], *, limit: int = 8) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for proposal in generate_graph_proposals(trace, world, limit=limit):
        path = out_dir / f"proposal_{proposal['proposal_id']}.json"
        path.write_text(json.dumps(proposal, indent=2) + "\n")
        written.append(path)
    return written


def write_artifacts(out_dir: Path, trace: Mapping[str, Any], graph: Mapping[str, Any], world: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (out_dir / "transition_graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    (out_dir / "introspection_world.json").write_text(json.dumps(world, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record and mine source-free ARC transition graphs.")
    parser.add_argument("scratch")
    parser.add_argument("--proposal")
    parser.add_argument("--out-dir", default="e143_transition_miner")
    parser.add_argument("--max-states", type=int, default=12)
    parser.add_argument("--no-probes", action="store_true")
    parser.add_argument("--proposal-out-dir")
    parser.add_argument("--proposal-limit", type=int, default=8)
    parser.add_argument("--beam-out-dir")
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--beam-depth", type=int, default=2)
    parser.add_argument("--beam-card-limit", type=int, default=8)
    parser.add_argument("--transfer-trace-dir")
    parser.add_argument("--transfer-out-dir")
    parser.add_argument("--transfer-limit", type=int, default=12)
    parser.add_argument("--transfer-signature-threshold", type=float, default=1000.0)
    args = parser.parse_args(argv)

    scratch = Path(args.scratch).resolve()
    proposal = Path(args.proposal).resolve() if args.proposal else None
    trace = record_trace(scratch, proposal)
    probes = None if args.no_probes else mine_probe_graph(scratch, trace, max_states=args.max_states)
    graph = build_transition_graph(trace, probes)
    world = build_introspection_world(graph)
    write_artifacts(Path(args.out_dir).resolve(), trace, graph, world)
    proposal_paths: list[Path] = []
    if args.proposal_out_dir:
        proposal_paths = write_graph_proposals(
            Path(args.proposal_out_dir).resolve(),
            trace,
            world,
            limit=args.proposal_limit,
        )
    beam_paths: list[Path] = []
    beam: Mapping[str, Any] | None = None
    if args.beam_out_dir:
        beam = beam_search_transition_cards(
            scratch,
            trace,
            world,
            beam_width=args.beam_width,
            depth=args.beam_depth,
            card_limit=args.beam_card_limit,
        )
        beam_paths = write_beam_artifacts(Path(args.beam_out_dir).resolve(), beam)
    transfer_paths: list[Path] = []
    if args.transfer_trace_dir and args.transfer_out_dir:
        suffixes = extract_behavioral_suffixes(
            trace,
            sorted(Path(args.transfer_trace_dir).glob(f"{trace.get('game')}*.json")),
            max_suffixes=max(args.transfer_limit * 4, args.transfer_limit),
        )
        suffixes = rank_behavioral_suffixes_by_signature(scratch, trace, suffixes)
        if args.transfer_signature_threshold is not None:
            suffixes = [
                s for s in suffixes if float(s.get("signature_distance", float("inf"))) <= args.transfer_signature_threshold
            ]
        suffixes = suffixes[: args.transfer_limit]
        transfer_paths = write_transfer_proposals(Path(args.transfer_out_dir).resolve(), trace, suffixes)
    print(
        json.dumps(
            {
                "out_dir": str(Path(args.out_dir).resolve()),
                "game": graph.get("game"),
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
                "critical_edges": len(graph["critical_edges"]),
                "cards": len(world["cards"]),
                "proposals": len(proposal_paths),
                "beam_artifacts": len(beam_paths),
                "beam_solved": bool(beam and beam.get("solved")),
                "transfer_proposals": len(transfer_paths),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
