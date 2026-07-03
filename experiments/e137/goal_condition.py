"""Goal-condition schema induction (E137 extension).

The action-shape schemas in `schema_induction.py` capture HOW the win actions look
(click vs move, the action tail). They are blind to the GOAL CONDITION -- the recurring
state-relational change at each level-up -- which is where ARC-AGI-3 wins actually live.

This module mines that condition from the per-demo frame summaries
(`start_summary`, `pre_win_summary`, `post_win_summary`) the demos already carry:
  * colours consistently ADDED at the level-up (the win produces them),
  * colours consistently CONSUMED at the level-up,
  * colours present in EVERY pre-win state (the win configuration always involves them),
  * whether the level builds up or consumes objects (start -> pre-win object-count sign).

Each candidate is scored by cross-level support and leave-one-out (LOO) consistency,
exactly like the action-shape schemas, so the two stack in one ranked packet.
Stdlib only; source-free (operates on summaries, never game source).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def _colors(summary: Mapping[str, Any]) -> set:
    """Colours present, excluding the background (bg is trivially in every state)."""
    bg = int(summary.get("bg", -999))
    return set(int(c) for c in summary.get("colors", []) if int(c) != bg)


def _win_features(demo: Mapping[str, Any]) -> dict[str, Any]:
    start = demo.get("start_summary", {}) or {}
    pre = demo.get("pre_win_summary", {}) or {}
    post = demo.get("post_win_summary", {}) or {}
    pc, qc = _colors(pre), _colors(post)
    return {
        "added": qc - pc,                 # colours that appear across the level-up
        "removed": pc - qc,               # colours consumed at the level-up
        "pre_colors": pc,                 # the achieved win configuration's palette
        "nobj_build": int(pre.get("n_objects", 0)) - int(start.get("n_objects", 0)),
    }


def _loo_membership(feats: Sequence[Mapping[str, Any]], field: str, color: int) -> float:
    """LOO: predict 'color is in <field>' from the train majority; score held-out agreement."""
    if len(feats) <= 1:
        return 1.0
    ok = 0
    for i, held in enumerate(feats):
        train = [f for j, f in enumerate(feats) if j != i]
        pred = sum(1 for f in train if color in f[field]) * 2 >= len(train)
        if pred == (color in held[field]):
            ok += 1
    return ok / len(feats)


def induce_goal_conditions(
    demos: Sequence[Mapping[str, Any]], min_support_frac: float = 0.5
) -> list[dict[str, Any]]:
    """Rank recurring STATE-RELATIONAL win conditions from the demos' frame summaries."""
    if not demos:
        return []
    feats = [_win_features(d) for d in demos]
    total = len(feats)
    out: list[dict[str, Any]] = []

    for field, verb, desc in (
        ("added", "win_adds_color", "ADDS"),
        ("removed", "win_removes_color", "CONSUMES"),
    ):
        counts: Counter = Counter()
        for f in feats:
            counts.update(f[field])
        for color, sup in counts.items():
            if sup / total >= min_support_frac:
                out.append(
                    {
                        "type": verb,
                        "pattern": int(color),
                        "support": sup,
                        "support_frac": sup / total,
                        "loo_success": _loo_membership(feats, field, color),
                        "description": f"each level-up {desc} colour {color}",
                    }
                )

    # colours present in EVERY pre-win state -> the win configuration always involves them
    common_pre = set.intersection(*[f["pre_colors"] for f in feats]) if feats else set()
    if common_pre:
        out.append(
            {
                "type": "pre_win_color_invariant",
                "pattern": sorted(int(c) for c in common_pre),
                "support": total,
                "support_frac": 1.0,
                "loo_success": 1.0,
                "description": f"the win configuration always contains colours {sorted(common_pre)}",
            }
        )

    # does the level build up or consume objects (start -> pre-win)?
    signs = Counter(1 if f["nobj_build"] > 0 else (-1 if f["nobj_build"] < 0 else 0) for f in feats)
    sign, sup = signs.most_common(1)[0]
    if sign != 0 and sup / total >= min_support_frac:
        out.append(
            {
                "type": "object_count_direction",
                "pattern": "grows" if sign > 0 else "shrinks",
                "support": sup,
                "support_frac": sup / total,
                "loo_success": sup / total,
                "description": (
                    f"object count {'grows' if sign > 0 else 'shrinks'} from level-start to win"
                ),
            }
        )

    out.sort(key=lambda c: (float(c["loo_success"]), float(c["support_frac"])), reverse=True)
    return out
