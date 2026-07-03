"""Generate fresh ARC proposals from failed terminal counterexamples.

This is the missing counterpart to the E139 ranker. Ranking old proposals cannot
escape a bad abstraction; this module reads terminal counterexample summaries
and emits small suffix probes that target newly exposed facts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


Action = list[int]


def _cursor(summary: Mapping[str, Any]) -> tuple[int | None, int | None]:
    cur = summary.get("cursor")
    if isinstance(cur, Mapping) and isinstance(cur.get("zero"), Mapping):
        zero = cur["zero"]
        return _as_int(zero.get("x")), _as_int(zero.get("y"))
    return None, None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _small_color(summary: Mapping[str, Any], color: int) -> list[Mapping[str, Any]]:
    small = summary.get("small") or []
    return [x for x in small if isinstance(x, Mapping) and _as_int(x.get("c")) == color]


def _move_suffix(src: tuple[int, int], dst: tuple[int, int]) -> list[Action]:
    x, y = src
    tx, ty = dst
    out: list[Action] = []
    while y > ty:
        out.append([1])
        y -= 3
    while y < ty:
        out.append([2])
        y += 3
    while x > tx:
        out.append([3])
        x -= 3
    while x < tx:
        out.append([4])
        x += 3
    return out


def _proposal(pid: str, prefix: Sequence[Action], suffix: Sequence[Action], hypothesis: str, bindings: Mapping[str, str]) -> dict[str, Any]:
    return {
        "proposal_id": pid,
        "schema_id": "counterexample-terminal-state generator",
        "goal_schema_id": "terminal-state repair after failed level-up",
        "hypothesis": hypothesis,
        "role_bindings": dict(bindings),
        "probe_plan": [list(a) for a in prefix] + [list(a) for a in suffix],
        "expected_deltas": [
            "prefix reproduces the failed terminal state",
            "suffix targets a newly exposed terminal-state object or parity",
            "a correct suffix should increase levels immediately or within a few actions",
        ],
        "fallback_repairs": [
            "if object click is inert, try the adjacent component center from the terminal summary",
            "if move suffix changes meter parity without level-up, regenerate from the new terminal state",
        ],
        "confidence": 0.05,
    }


def generate_from_counterexample(counterexample: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = counterexample.get("final_summary")
    if not isinstance(summary, Mapping):
        return []
    prefix = counterexample.get("executed") or []
    if not isinstance(prefix, list):
        return []
    cx, cy = _cursor(summary)
    if cx is None or cy is None:
        return []
    c4 = (summary.get("palette") or {}).get("4")
    small4 = _small_color(summary, 4)
    base = str(counterexample.get("proposal_id") or "counterexample")
    bindings = {
        "failed_prefix": f"{base} reached cursor=({cx},{cy}) without level-up",
        "terminal_palette": f"terminal color-4 count is {c4}",
        "terminal_small4": json.dumps(small4, sort_keys=True),
    }
    out: list[dict[str, Any]] = []

    square = next((o for o in small4 if _as_int(o.get("x")) == 49 and _as_int(o.get("y")) == 22), None)
    if square:
        sx, sy = _as_int(square.get("x")), _as_int(square.get("y"))
        if sx is not None and sy is not None:
            to_square = _move_suffix((cx, cy), (sx, sy))
            out.append(
                _proposal(
                    "gen-terminal-click-square-after-meter",
                    prefix,
                    to_square + [[6, sx, sy]],
                    "At the c4=62 terminal state the missing operation is selection of the shrunken square, not further movement.",
                    bindings | {"target_square": f"small color-4 remnant at ({sx},{sy})"},
                )
            )
            out.append(
                _proposal(
                    "gen-terminal-square-parity-click",
                    prefix,
                    [[3], [4]] + to_square + [[6, sx, sy], [2]],
                    "One more left/right parity cycle may arm the c4=62 state; then click square and step down.",
                    bindings | {"target_square": f"small color-4 remnant at ({sx},{sy})"},
                )
            )
            out.append(
                _proposal(
                    "gen-terminal-square-perimeter",
                    prefix,
                    to_square + [[3], [4], [1], [2], [6, sx, sy]],
                    "The square may require perimeter contact before the click selector binds.",
                    bindings | {"target_square": f"small color-4 remnant at ({sx},{sy})"},
                )
            )

    for o in small4:
        ox, oy = _as_int(o.get("x")), _as_int(o.get("y"))
        size = _as_int(o.get("n"))
        if ox is None or oy is None:
            continue
        if ox == 49 and oy == 22:
            continue
        out.append(
            _proposal(
                f"gen-terminal-click-c4-remnant-{ox}-{oy}",
                prefix,
                [[6, ox, oy], [1], [2]],
                "The terminal state exposes a non-square color-4 remnant; it may be the real meter endpoint or latch.",
                bindings | {"terminal_remnant": f"small color-4 component at ({ox},{oy}) size {size}"},
            )
        )
        out.append(
            _proposal(
                f"gen-terminal-touch-c4-remnant-{ox}-{oy}",
                prefix,
                _move_suffix((cx, cy), (ox, min(60, oy)))[:24] + [[6, ox, oy]],
                "The bottom color-4 remnant may need cursor proximity before click selection.",
                bindings | {"terminal_remnant": f"small color-4 component at ({ox},{oy}) size {size}"},
            )
        )

    out.append(
        _proposal(
            "gen-terminal-action-scan",
            prefix,
            [[1], [2], [3], [4], [5], [7]],
            "The terminal c4=62 state may be armed; scan non-click actions once instead of targeting visible objects.",
            bindings,
        )
    )
    return out


def write_generated(counterexample_paths: Sequence[str | Path], out_dir: str | Path, *, prefix: str = "proposal_") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[str] = set()
    for path in counterexample_paths:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, Mapping):
            continue
        for proposal in generate_from_counterexample(data):
            pid = str(proposal["proposal_id"])
            if pid in seen:
                continue
            seen.add(pid)
            p = out / f"{prefix}{pid}.json"
            p.write_text(json.dumps(proposal, indent=2) + "\n")
            written.append(p)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate proposals from terminal counterexamples.")
    parser.add_argument("out_dir")
    parser.add_argument("counterexample_json", nargs="+")
    args = parser.parse_args(argv)

    written = write_generated(args.counterexample_json, args.out_dir)
    print(f"[e139-generate] wrote {len(written)} proposals to {args.out_dir}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

