"""Hybrid E138/E139 proposal ranker.

E138 is good at proposal quality: schema grounding, role bindings, compact
valid plans, and explicit repairs. E139 is good at allocation after failures:
many-world fit, proposal-family suppression, and exact counterexample memory.

This module combines them without inspecting game source.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from experiments.e138.judge_schema import load_json, load_proposals, score_proposal
    from experiments.e139.manyworld_semiring import (
        Candidate,
        _actions_from_mapping,
        _failure_counts,
        _flatten_text,
        load_candidate,
        make_hardworld_query_worlds,
        posterior_scores,
        proposal_family,
        rank_candidates,
    )
except ModuleNotFoundError:  # pragma: no cover - used from copied scratch dirs.
    from judge_schema import load_json, load_proposals, score_proposal  # type: ignore
    from manyworld_semiring import (  # type: ignore
        Candidate,
        _actions_from_mapping,
        _failure_counts,
        _flatten_text,
        load_candidate,
        make_hardworld_query_worlds,
        posterior_scores,
        proposal_family,
        rank_candidates,
    )


def _candidate_from_proposal(proposal: Mapping[str, Any], *, failed: bool = False) -> Candidate:
    summary = proposal.get("final_summary") if isinstance(proposal.get("final_summary"), Mapping) else None
    return Candidate(
        candidate_id=str(proposal.get("proposal_id") or proposal.get("id") or "proposal"),
        actions=_actions_from_mapping(proposal),
        text=_flatten_text(proposal),
        proposal=proposal,
        summary=summary,
        failed=failed,
    )


def _e139_scalar(row: Mapping[str, Any]) -> float:
    semiring = row.get("semiring") or [0, 0, 0, 0, 0]
    level_gain, neg_energy, novelty, neg_cost, neg_risk = [float(x) for x in semiring]
    score = (
        100.0 * level_gain
        + 1.6 * neg_energy
        + 0.9 * novelty
        + 0.018 * neg_cost
        + 0.65 * neg_risk
    )
    penalty = row.get("failure_penalty") or {}
    score -= 0.75 * float(penalty.get("total", 0.0))
    if row.get("failed"):
        score -= 100.0
    return score


def _quality_floor(e138_score: float, e139_row: Mapping[str, Any]) -> float:
    """Prevent pure diversification from elevating very weak leftovers."""
    n_actions = int(e139_row.get("n_actions") or 0)
    penalty = e139_row.get("failure_penalty") or {}
    exact = bool(penalty.get("exact_failure"))
    if exact:
        return -100.0
    if e138_score < 2.5:
        return -4.0
    if n_actions <= 3 and e138_score < 4.0:
        return -2.0
    return 0.0


def rank_hybrid(
    packet: Mapping[str, Any],
    proposals: Sequence[Mapping[str, Any]],
    *,
    counterexamples: Sequence[Candidate] = (),
    beta: float = 1.0,
    e138_weight: float = 1.0,
    e139_weight: float = 1.0,
) -> dict[str, Any]:
    worlds = make_hardworld_query_worlds()
    failed_family_counts, failed_world_counts, failed_world_by_id = _failure_counts(
        counterexamples,
        worlds,
        beta=beta,
    )
    failed_ids = {c.candidate_id for c in counterexamples}
    candidates = [
        _candidate_from_proposal(proposal, failed=str(proposal.get("proposal_id")) in failed_ids)
        for proposal in proposals
    ]
    e139_rows = rank_candidates(
        candidates,
        worlds=worlds,
        beta=beta,
        failed_family_counts=failed_family_counts,
        failed_world_counts=failed_world_counts,
    )
    by_id = {row["candidate_id"]: row for row in e139_rows}

    rows = []
    for proposal in proposals:
        e138 = score_proposal(packet, proposal)
        e139 = by_id[e138.proposal_id]
        _, posterior, best_energy = posterior_scores(
            _candidate_from_proposal(proposal, failed=e139.get("failed", False)),
            worlds,
            beta=beta,
        )
        hybrid = (
            e138_weight * e138.score
            + e139_weight * _e139_scalar(e139)
            + _quality_floor(e138.score, e139)
        )
        if not math.isfinite(hybrid):
            hybrid = -1e9
        rows.append(
            {
                "rank": 0,
                "proposal_id": e138.proposal_id,
                "hybrid_score": round(hybrid, 6),
                "e138_score": e138.score,
                "e138_reasons": e138.reasons,
                "e139_score": round(_e139_scalar(e139), 6),
                "e139": e139,
                "best_energy": round(best_energy, 6),
                "posterior": sorted(posterior, key=lambda row: row["posterior"], reverse=True),
                "proposal": proposal,
            }
        )
    rows.sort(key=lambda row: row["hybrid_score"], reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return {
        "experiment": "E139 hybrid E138-schema + many-world semiring ranker",
        "beta": beta,
        "weights": {"e138": e138_weight, "e139": e139_weight},
        "failure_model": {
            "families": failed_family_counts,
            "worlds": failed_world_counts,
            "failed_world_by_id": failed_world_by_id,
        },
        "game": packet.get("game"),
        "frontier_level": packet.get("frontier_level"),
        "win": packet.get("win"),
        "n_proposals": len(proposals),
        "winner": rows[0] if rows else None,
        "ranked": rows,
    }


def write_hybrid_ranking(
    packet_path: str | Path,
    proposal_paths: Sequence[str | Path],
    out_path: str | Path,
    *,
    counterexample_paths: Sequence[str | Path] = (),
    beta: float = 1.0,
    e138_weight: float = 1.0,
    e139_weight: float = 1.0,
) -> dict[str, Any]:
    packet = load_json(packet_path)
    proposals = load_proposals(proposal_paths)
    counterexamples = [load_candidate(path, failed=True) for path in counterexample_paths]
    ranking = rank_hybrid(
        packet,
        proposals,
        counterexamples=counterexamples,
        beta=beta,
        e138_weight=e138_weight,
        e139_weight=e139_weight,
    )
    Path(out_path).write_text(json.dumps(ranking, indent=2, sort_keys=True))
    return ranking


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank proposals with hybrid E138/E139 scoring.")
    parser.add_argument("schema_packet")
    parser.add_argument("out_json")
    parser.add_argument("proposal_json", nargs="+")
    parser.add_argument("--counterexample", action="append", default=[])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--e138-weight", type=float, default=1.0)
    parser.add_argument("--e139-weight", type=float, default=1.0)
    args = parser.parse_args(argv)

    counterexamples: list[str] = []
    for path in args.counterexample:
        matches = sorted(glob.glob(path))
        counterexamples.extend(matches or [path])

    ranking = write_hybrid_ranking(
        args.schema_packet,
        args.proposal_json,
        args.out_json,
        counterexample_paths=counterexamples,
        beta=args.beta,
        e138_weight=args.e138_weight,
        e139_weight=args.e139_weight,
    )
    winner = ranking.get("winner") or {}
    print(
        f"[e139-hybrid] ranked {ranking['n_proposals']} proposals; "
        f"winner={winner.get('proposal_id')} hybrid={winner.get('hybrid_score')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
