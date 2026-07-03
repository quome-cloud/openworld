"""Cross-level procedural schema induction for source-free ARC-AGI-3 runs.

The module is intentionally lightweight and source-free: it consumes observed
frames, actions, and level counters. It does not load the ARC engine package or
game source.
The output is a compact schema packet for an agent: prior solved levels become
demonstrations, and candidate cross-level procedures are ranked by how well they
explain those demonstrations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from experiments.e125.objstate import object_state
except ImportError:  # flat source-free workspace
    from objstate import object_state

try:
    from experiments.e134.composite import composite_key
except ImportError:  # flat source-free workspace
    try:
        from composite import composite_key
    except Exception:  # e134 is useful but optional for packet generation
        composite_key = None

try:
    from experiments.e137.goal_condition import induce_goal_conditions
except ImportError:  # flat source-free workspace
    from goal_condition import induce_goal_conditions


Action = list[int]


@dataclass(frozen=True)
class StepRecord:
    before: Any
    action: Action
    after: Any
    levels_before: int
    levels_after: int


def normalize_action(action: Any) -> Action:
    if isinstance(action, int):
        return [int(action)]
    if isinstance(action, tuple):
        action = list(action)
    if not isinstance(action, list) or not action:
        raise ValueError(f"bad action: {action!r}")
    return [int(x) for x in action]


def action_kind(action: Sequence[int]) -> str:
    a = int(action[0])
    if a == 6 and len(action) >= 3:
        return "click"
    if a in (1, 2, 3, 4):
        return "move"
    if a == 5:
        return "interact"
    if a == 7:
        return "undo"
    return f"action_{a}"


def action_signature(action: Sequence[int]) -> str:
    if int(action[0]) == 6 and len(action) >= 3:
        return "click"
    return str(int(action[0]))


def _json_safe_key(value: Any, limit: int = 600) -> str:
    try:
        text = json.dumps(value, sort_keys=True)
    except Exception:
        text = repr(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def frame_summary(frame: Any) -> dict[str, Any]:
    state = object_state(frame)
    objects = state.get("objects", [])
    hist = Counter(int(o["color"]) for o in objects)
    small = sorted(
        (
            {
                "color": int(o["color"]),
                "size": int(o["size"]),
                "y": int(o["y"]),
                "x": int(o["x"]),
            }
            for o in objects
            if int(o.get("size", 9999)) <= 24
        ),
        key=lambda o: (o["size"], hist[o["color"]], o["color"], o["y"], o["x"]),
    )[:16]
    summary = {
        "bg": int(state.get("bg", -1)),
        "n_objects": len(objects),
        "colors": sorted(hist),
        "color_counts": {str(k): int(v) for k, v in sorted(hist.items())},
        "small_salient_objects": small,
    }
    if composite_key is not None:
        try:
            summary["composite_key_preview"] = _json_safe_key(composite_key(frame), limit=300)
        except Exception:
            pass
    return summary


def segment_by_level(records: Sequence[StepRecord]) -> list[dict[str, Any]]:
    """Split a replay trajectory into one demo per level-up."""
    demos: list[dict[str, Any]] = []
    start = 0
    current_level = records[0].levels_before if records else 0
    for idx, rec in enumerate(records):
        if rec.levels_after > current_level:
            segment = records[start : idx + 1]
            demos.append(
                {
                    "level": int(rec.levels_after),
                    "start_index": start,
                    "end_index": idx,
                    "actions": [r.action for r in segment],
                    "action_signatures": [action_signature(r.action) for r in segment],
                    "action_kinds": [action_kind(r.action) for r in segment],
                    "start_summary": frame_summary(segment[0].before),
                    "pre_win_summary": frame_summary(rec.before),
                    "post_win_summary": frame_summary(rec.after),
                    "level_delta": int(rec.levels_after - rec.levels_before),
                    "last_action": rec.action,
                    "length": len(segment),
                }
            )
            start = idx + 1
            current_level = rec.levels_after
    return demos


def _suffix(items: Sequence[Any], n: int) -> tuple[Any, ...]:
    return tuple(items[-n:]) if len(items) >= n else tuple(items)


def induce_schemas(demos: Sequence[Mapping[str, Any]], max_suffix: int = 8) -> list[dict[str, Any]]:
    """Rank compact procedure candidates by cross-level support.

    The first implementation deliberately favors robust, inspectable patterns:
    action-kind tails, exact action tails, final-action type, and demo length
    trend. The agent receives the packet and can repair/generalize from there.
    """
    if not demos:
        return []

    candidates: list[dict[str, Any]] = []
    total = len(demos)

    for width in range(1, max_suffix + 1):
        kind_counts = Counter(_suffix(d.get("action_kinds", []), width) for d in demos)
        sig_counts = Counter(_suffix(d.get("action_signatures", []), width) for d in demos)
        for label, counts in (("action_kind_suffix", kind_counts), ("action_signature_suffix", sig_counts)):
            pattern, support = counts.most_common(1)[0]
            candidates.append(
                {
                    "type": label,
                    "pattern": list(pattern),
                    "width": width,
                    "support": support,
                    "support_frac": support / total,
                    "loo_success": _loo_pattern_success(demos, label, width),
                    "description": f"{label}[{width}] = {list(pattern)}",
                }
            )

    last_counts = Counter(action_kind(d["last_action"]) for d in demos)
    last_kind, support = last_counts.most_common(1)[0]
    candidates.append(
        {
            "type": "last_action_kind",
            "pattern": last_kind,
            "support": support,
            "support_frac": support / total,
            "loo_success": _loo_last_kind_success(demos),
            "description": f"final level-up action is usually {last_kind}",
        }
    )

    lengths = [int(d["length"]) for d in demos]
    candidates.append(
        {
            "type": "length_trend",
            "pattern": {
                "min": min(lengths),
                "median": sorted(lengths)[len(lengths) // 2],
                "max": max(lengths),
                "last": lengths[-1],
            },
            "support": total,
            "support_frac": 1.0,
            "loo_success": 1.0,
            "description": "observed per-level action-budget envelope",
        }
    )

    def score(c: Mapping[str, Any]) -> tuple[float, float, int]:
        return (
            float(c.get("loo_success", 0.0)),
            float(c.get("support_frac", 0.0)),
            -len(json.dumps(c.get("pattern", ""))),
        )

    candidates.sort(key=score, reverse=True)
    return candidates[:12]


def _loo_pattern_success(demos: Sequence[Mapping[str, Any]], label: str, width: int) -> float:
    if len(demos) <= 1:
        return 1.0
    ok = 0
    for i, held in enumerate(demos):
        train = [d for j, d in enumerate(demos) if j != i]
        key = "action_kinds" if label == "action_kind_suffix" else "action_signatures"
        pattern = Counter(_suffix(d.get(key, []), width) for d in train).most_common(1)[0][0]
        if _suffix(held.get(key, []), width) == pattern:
            ok += 1
    return ok / len(demos)


def _loo_last_kind_success(demos: Sequence[Mapping[str, Any]]) -> float:
    if len(demos) <= 1:
        return 1.0
    ok = 0
    for i, held in enumerate(demos):
        train = [d for j, d in enumerate(demos) if j != i]
        pred = Counter(action_kind(d["last_action"]) for d in train).most_common(1)[0][0]
        if action_kind(held["last_action"]) == pred:
            ok += 1
    return ok / len(demos)


def build_packet(
    game: str,
    records: Sequence[StepRecord],
    frontier_actions: Sequence[Sequence[int]],
    frontier_levels: int,
    win: int,
    target_games_rank: Sequence[str] = (),
) -> dict[str, Any]:
    demos = segment_by_level(records)
    schemas = induce_schemas(demos)
    goal_conditions = induce_goal_conditions(demos)
    return {
        "experiment": "E137 cross-level procedural schema induction",
        "game": game,
        "frontier_level": int(frontier_levels),
        "win": int(win),
        "remaining_levels": max(0, int(win) - int(frontier_levels)),
        "frontier_action_count": len(frontier_actions),
        "target_priority": list(target_games_rank),
        "n_solved_level_demos": len(demos),
        "solved_level_demos": demos,
        "candidate_schemas": schemas,
        "goal_condition_schemas": goal_conditions,
        "instructions": [
            "Read goal_condition_schemas FIRST: they say WHAT configuration each level-up achieves "
            "(the win condition), which action-shape schemas cannot.",
            "Validate/repair a candidate against solved-level demos, then instantiate it on the frontier frame.",
            "Use real-env probes only to bind uncertain roles; bank every deeper level.",
        ],
    }


def replay_records(env: Any, actions: Iterable[Sequence[int]]) -> list[StepRecord]:
    """Replay actions through a SandboxGame-like env and collect source-free records."""
    records: list[StepRecord] = []
    env.reset()
    for raw in actions:
        action = normalize_action(raw)
        before = env.frame.copy()
        lb = int(env.levels)
        if action[0] == 6:
            env.step(6, action[1], action[2])
        else:
            env.step(action[0])
        records.append(StepRecord(before, action, env.frame.copy(), lb, int(env.levels)))
    return records


def write_packet(packet: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(packet, indent=2, sort_keys=True))
