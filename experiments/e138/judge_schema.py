"""Judge/ranker utilities for E138 schema tournaments.

The module is deliberately source-free. It reads an E137 schema packet and
candidate proposal JSON files, scores them against the solved demos/schema
evidence, and writes a ranked tournament file. It does not inspect game source
and does not certify solutions; it only allocates execution budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


Action = list[int]


JUDGE_RUBRIC = """\
Rank the proposal most likely to advance the current ARC-AGI-3 source-free
frontier. Prefer proposals that:
1. cite a high-support E137 schema or goal-condition schema,
2. bind roles from observed frame/object evidence rather than guessing,
3. use a small valid probe/execution plan,
4. state expected level-up deltas and counterexample repairs,
5. avoid unsupported harness calls and generic random search.
Replay, audit, and banking remain the only certification path.
"""


@dataclass(frozen=True)
class ProposalScore:
    proposal_id: str
    score: float
    reasons: list[str]


def normalize_action(raw: Any) -> Action | None:
    if isinstance(raw, int):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        return None
    try:
        action = [int(x) for x in raw]
    except Exception:
        return None
    if action[0] == 6:
        if len(action) != 3:
            return None
        if not (0 <= action[1] <= 63 and 0 <= action[2] <= 63):
            return None
        return action
    if len(action) != 1:
        return None
    if action[0] in (1, 2, 3, 4, 5, 7):
        return action
    return None


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _schema_index(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for group in ("candidate_schemas", "goal_condition_schemas"):
        for i, schema in enumerate(packet.get(group, []) or []):
            sid = str(schema.get("id") or schema.get("description") or f"{group}:{i}")
            out[sid] = schema
            out[f"{group}:{i}"] = schema
            if schema.get("type") is not None:
                out[str(schema["type"])] = schema
    return out


def _best_schema_bonus(packet: Mapping[str, Any], proposal: Mapping[str, Any]) -> tuple[float, str | None]:
    index = _schema_index(packet)
    refs = []
    for key in ("schema_id", "schema_ref", "goal_schema_id", "goal_schema_ref"):
        if proposal.get(key):
            refs.append(str(proposal[key]))
    refs.extend(str(x) for x in proposal.get("schema_refs", []) or [])

    best = 0.0
    best_reason = None
    for ref in refs:
        schema = index.get(ref)
        if not schema:
            continue
        support = float(schema.get("support_frac", 0.0))
        loo = float(schema.get("loo_success", 0.0))
        bonus = 2.0 * support + 2.0 * loo
        if bonus > best:
            best = bonus
            best_reason = f"schema evidence {ref}: support={support:.2f}, loo={loo:.2f}"
    return best, best_reason


def score_proposal(packet: Mapping[str, Any], proposal: Mapping[str, Any]) -> ProposalScore:
    pid = str(proposal.get("proposal_id") or proposal.get("id") or "proposal")
    score = 0.0
    reasons: list[str] = []

    bonus, reason = _best_schema_bonus(packet, proposal)
    score += bonus
    if reason:
        reasons.append(reason)

    bindings = proposal.get("role_bindings") or {}
    if isinstance(bindings, Mapping) and bindings:
        n = min(len(bindings), 8)
        score += 0.35 * n
        reasons.append(f"{n} role bindings")

    plan = proposal.get("probe_plan") or proposal.get("execution_plan") or []
    valid_actions = [normalize_action(a) for a in plan]
    valid_actions = [a for a in valid_actions if a is not None]
    if valid_actions:
        score += 1.5
        reasons.append(f"{len(valid_actions)} valid planned actions")
        if len(valid_actions) <= 32:
            score += 0.75
            reasons.append("small budget plan")
    elif plan:
        score -= 2.0
        reasons.append("plan has no valid actions")

    if proposal.get("expected_deltas"):
        score += 0.75
        reasons.append("states expected deltas")
    if proposal.get("fallback_repairs") or proposal.get("counterexample_repairs"):
        score += 0.75
        reasons.append("has repair plan")
    if proposal.get("hypothesis"):
        score += 0.5
        reasons.append("has explicit hypothesis")

    text = json.dumps(proposal, sort_keys=True).lower()
    source_import = "import " + "arc" + "_agi"
    for bad in ("g.replay", "hard_reset", source_import, "random search", "brute force"):
        if bad in text:
            score -= 2.5
            reasons.append(f"penalty: {bad}")

    confidence = proposal.get("confidence")
    if isinstance(confidence, int | float):
        score += max(0.0, min(float(confidence), 1.0)) * 0.5

    return ProposalScore(pid, round(score, 4), reasons)


def rank_proposals(packet: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [
        {
            "rank": 0,
            "proposal_id": score.proposal_id,
            "score": score.score,
            "reasons": score.reasons,
            "proposal": proposal,
        }
        for proposal in proposals
        for score in [score_proposal(packet, proposal)]
    ]
    scored.sort(key=lambda row: row["score"], reverse=True)
    for i, row in enumerate(scored, start=1):
        row["rank"] = i
    return {
        "experiment": "E138 judge-guided schema tournament",
        "rubric": JUDGE_RUBRIC,
        "game": packet.get("game"),
        "frontier_level": packet.get("frontier_level"),
        "win": packet.get("win"),
        "n_proposals": len(proposals),
        "winner": scored[0] if scored else None,
        "ranked": scored,
    }


def load_proposals(paths: Sequence[str | Path]) -> list[Mapping[str, Any]]:
    proposals: list[Mapping[str, Any]] = []
    for path in paths:
        data = load_json(path)
        if isinstance(data, list):
            proposals.extend(x for x in data if isinstance(x, Mapping))
        elif isinstance(data, Mapping) and isinstance(data.get("proposals"), list):
            proposals.extend(x for x in data["proposals"] if isinstance(x, Mapping))
        elif isinstance(data, Mapping):
            proposals.append(data)
    return proposals


def write_ranking(packet_path: str | Path, proposal_paths: Sequence[str | Path], out_path: str | Path) -> dict[str, Any]:
    packet = load_json(packet_path)
    proposals = load_proposals(proposal_paths)
    ranking = rank_proposals(packet, proposals)
    Path(out_path).write_text(json.dumps(ranking, indent=2, sort_keys=True))
    return ranking


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Rank E138 schema proposals.")
    ap.add_argument("schema_packet")
    ap.add_argument("out_json")
    ap.add_argument("proposal_json", nargs="+")
    args = ap.parse_args(argv)

    ranking = write_ranking(args.schema_packet, args.proposal_json, args.out_json)
    winner = ranking.get("winner") or {}
    print(
        f"[e138] ranked {ranking['n_proposals']} proposals; "
        f"winner={winner.get('proposal_id')} score={winner.get('score')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
