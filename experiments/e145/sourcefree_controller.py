"""Source-free episodic-memory controller for ARC-AGI-3.

E145 is the fair version of E144. It keeps the useful idea--retrieve verified
behavior fragments by source-free state signature--but gates the memory bank by
run provenance. A trace is eligible only when it was produced by a source-free,
audit-clean, untainted, replay-verified run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.e144.full_run_controller import run_full_controller


def _truthy_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return bool(value)


def is_sourcefree_eligible_run(record: Mapping[str, Any]) -> bool:
    if not _truthy_bool(record.get("source_free")):
        return False
    if _truthy_bool(record.get("memory_tainted")):
        return False

    outcome = record.get("outcome")
    if not isinstance(outcome, Mapping):
        return False
    if not _truthy_bool(outcome.get("replay_verified")):
        return False
    if int(outcome.get("levels", 0) or 0) <= 0:
        return False

    audit = outcome.get("audit")
    if isinstance(audit, Mapping) and not _truthy_bool(audit.get("clean"), default=True):
        return False

    knowledge_audit = record.get("knowledge_audit")
    if isinstance(knowledge_audit, Mapping) and not _truthy_bool(knowledge_audit.get("clean"), default=True):
        return False

    return True


def load_run_records(trace_root: Path) -> list[dict[str, Any]]:
    runs_path = trace_root / "runs.jsonl"
    if runs_path.exists():
        records: list[dict[str, Any]] = []
        for line in runs_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    records = []
    for path in sorted((trace_root / "meta").glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except Exception:
            continue
    return records


def eligible_solution_paths(
    trace_root: Path,
    *,
    games: Iterable[str] | None = None,
) -> list[Path]:
    game_filter = {str(g) for g in games} if games is not None else None
    paths: list[Path] = []
    seen: set[Path] = set()
    for record in load_run_records(trace_root):
        game = str(record.get("game", ""))
        if game_filter is not None and game not in game_filter:
            continue
        if not is_sourcefree_eligible_run(record):
            continue
        solution_file = record.get("solution_file")
        if not isinstance(solution_file, str) or not solution_file:
            continue
        path = trace_root / solution_file
        if not path.exists() or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return sorted(paths)


def write_memory_manifest(out_dir: Path, solution_paths: Sequence[Path]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in solution_paths:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        rows.append(
            {
                "path": str(path),
                "game": data.get("game"),
                "levels": data.get("levels"),
                "action_count": len(data.get("actions", []) if isinstance(data.get("actions"), list) else []),
            }
        )
    manifest = out_dir / "sourcefree_memory_manifest.json"
    manifest.write_text(json.dumps({"count": len(rows), "traces": rows}, indent=2) + "\n")
    return manifest


def run_sourcefree_controller(
    game: str,
    trace_root: Path,
    scratch: Path,
    out_dir: Path,
    *,
    sandbox_path: Path = Path("experiments/arc3_sandbox.py"),
    max_stages: int = 12,
    transfer_limit: int = 24,
    signature_threshold: float = 1000.0,
) -> dict[str, Any]:
    solution_paths = eligible_solution_paths(trace_root, games=[game])
    manifest = write_memory_manifest(out_dir, solution_paths)
    result = run_full_controller(
        game,
        trace_root / "solutions",
        scratch,
        out_dir,
        sandbox_path=sandbox_path,
        max_stages=max_stages,
        transfer_limit=transfer_limit,
        signature_threshold=signature_threshold,
        solution_paths=solution_paths,
    )
    result["experiment"] = "E145"
    result["protocol"] = "source-free episodic memory"
    result["memory_manifest"] = str(manifest)
    result["eligible_memory_traces"] = len(solution_paths)
    result["solve_time_llm_calls"] = 0
    result["caveat"] = (
        "Uses only prior traces whose run metadata is source_free, audit-clean, "
        "untainted, and replay-verified."
    )
    (out_dir / "sourcefree_full_run_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E145 source-free episodic-memory controller.")
    parser.add_argument("game")
    parser.add_argument("--trace-root", default="experiments/results/arc3_traces")
    parser.add_argument("--scratch-dir", default="/tmp/e145_scratch")
    parser.add_argument("--out-dir", default="/tmp/e145_run")
    parser.add_argument("--sandbox-path", default="experiments/arc3_sandbox.py")
    parser.add_argument("--max-stages", type=int, default=12)
    parser.add_argument("--transfer-limit", type=int, default=24)
    parser.add_argument("--signature-threshold", type=float, default=1000.0)
    args = parser.parse_args()

    result = run_sourcefree_controller(
        args.game,
        Path(args.trace_root).resolve(),
        Path(args.scratch_dir).resolve(),
        Path(args.out_dir).resolve(),
        sandbox_path=Path(args.sandbox_path).resolve(),
        max_stages=args.max_stages,
        transfer_limit=args.transfer_limit,
        signature_threshold=args.signature_threshold,
    )
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "game": result["game"],
                "levels": result["levels"],
                "action_count": result["action_count"],
                "eligible_memory_traces": result["eligible_memory_traces"],
                "stopped_reason": result["stopped_reason"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

