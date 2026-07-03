"""Many-world query semiring ranker for ARC source-free probe allocation.

E139 is deliberately source-free: it consumes proposal JSON and optional probe
counterexamples/summaries, then ranks what to execute next. A "query world" is a
hidden-state hypothesis such as "the yellow square is a decoy", "the plus
fragments are selectable", or "the color-4 meter must be bracketed first".

The scoring has two parts:
  * a path-integral posterior over query worlds, using log-sum-exp over
    world-specific energies;
  * a conservative semiring tie-break that prefers level gain, then lower
    energy, novelty, lower cost, and lower risk.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


Action = list[int]


@dataclass(frozen=True)
class QueryWorld:
    world_id: str
    description: str
    terms: Mapping[str, float]
    prior: float = 1.0


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    actions: tuple[tuple[int, ...], ...]
    text: str
    proposal: Mapping[str, Any]
    summary: Mapping[str, Any] | None = None
    failed: bool = False


@dataclass(frozen=True, order=True)
class SemiringValue:
    level_gain: float
    neg_energy: float
    novelty: float
    neg_cost: float
    neg_risk: float

    def as_list(self) -> list[float]:
        return [
            round(self.level_gain, 6),
            round(self.neg_energy, 6),
            round(self.novelty, 6),
            round(self.neg_cost, 6),
            round(self.neg_risk, 6),
        ]


def proposal_family(candidate: Candidate) -> str:
    text = f"{candidate.candidate_id}\n{candidate.text}"
    if candidate.candidate_id.startswith("gen-terminal-") or _has_words(text, ("terminal-state generator", "terminal state exposes")):
        return "terminal_generated"
    if _has_words(text, ("meter-threshold", "threshold", "c4=62", "color-4 phase")):
        return "meter_threshold"
    if _has_words(text, ("meter-phase", "phase square", "oscillation")):
        return "meter_phase"
    if _has_words(text, ("click-plus-fragments", "fragment x=12", "plus fragment")):
        return "plus_fragment"
    if _has_words(text, ("dark-left", "dark object", "portal")):
        return "dark_object"
    if _has_words(text, ("large-plus", "plus target")):
        return "large_plus"
    if _has_words(text, ("direct-yellow-square", "direct square", "square-up", "square-down")):
        return "direct_square"
    if _has_words(text, ("orange-finish", "known-good prefix", "remaining_singleton")):
        return "orange_finish"
    if _has_words(text, ("orange-shepherd", "shepherd")):
        return "orange_shepherd"
    if _has_words(text, ("orange-orbit", "orbit")):
        return "orange_orbit"
    if _has_words(text, ("green-gate", "magenta-gate", "gate pair")):
        return "gate_pairing"
    if _has_words(text, ("visible-component", "sweep")):
        return "component_sweep"
    return "other"


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    m = max(values)
    if math.isinf(m):
        return m
    return m + math.log(sum(math.exp(v - m) for v in values))


def _norm_action(raw: Any) -> tuple[int, ...] | None:
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


def _actions_from_mapping(data: Mapping[str, Any]) -> tuple[tuple[int, ...], ...]:
    raw_plan = (
        data.get("probe_plan")
        or data.get("execution_plan")
        or data.get("executed")
        or data.get("actions")
        or []
    )
    out = []
    for raw in raw_plan:
        action = _norm_action(raw)
        if action is not None:
            out.append(action)
    return tuple(out)


def _flatten_text(data: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "proposal_id",
        "schema_id",
        "goal_schema_id",
        "hypothesis",
        "role_bindings",
        "expected_deltas",
        "fallback_repairs",
        "counterexample_repairs",
    ):
        if key in data:
            parts.append(json.dumps(data[key], sort_keys=True).lower())
    return "\n".join(parts)


def _cursor_xy(summary: Mapping[str, Any] | None) -> tuple[float | None, float | None]:
    if not isinstance(summary, Mapping):
        return None, None
    cur = summary.get("cursor")
    if isinstance(cur, Mapping):
        if isinstance(cur.get("zero"), Mapping):
            zero = cur["zero"]
            return _as_float(zero.get("x")), _as_float(zero.get("y"))
        return _as_float(cur.get("x")), _as_float(cur.get("y"))
    return None, None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _palette_count(summary: Mapping[str, Any] | None, color: int) -> float | None:
    if not isinstance(summary, Mapping) or not isinstance(summary.get("palette"), Mapping):
        return None
    palette = summary["palette"]
    return _as_float(palette.get(str(color), palette.get(color)))


def _small_components(summary: Mapping[str, Any] | None, color: int) -> list[Mapping[str, Any]]:
    if not isinstance(summary, Mapping) or not isinstance(summary.get("small"), list):
        return []
    return [x for x in summary["small"] if isinstance(x, Mapping) and int(x.get("c", -1)) == color]


def _distance_to(summary: Mapping[str, Any] | None, target: tuple[float, float]) -> float | None:
    x, y = _cursor_xy(summary)
    if x is None or y is None:
        return None
    return abs(x - target[0]) + abs(y - target[1])


def _contains_click(actions: Sequence[tuple[int, ...]], target: tuple[int, int], radius: int = 1) -> bool:
    for action in actions:
        if len(action) == 3 and action[0] == 6:
            if abs(action[1] - target[0]) <= radius and abs(action[2] - target[1]) <= radius:
                return True
    return False


def _count_alternations(actions: Sequence[tuple[int, ...]]) -> int:
    count = 0
    last = None
    for action in actions:
        if action in ((3,), (4,)):
            if last is not None and action != last:
                count += 1
            last = action
        else:
            last = None
    return count


def _has_words(text: str, words: Iterable[str]) -> bool:
    return any(word in text for word in words)


def feature_vector(candidate: Candidate) -> dict[str, float]:
    summary = candidate.summary
    actions = candidate.actions
    text = candidate.text
    levels = _as_float(summary.get("levels")) if isinstance(summary, Mapping) else None
    start_level = _as_float(candidate.proposal.get("start_level")) or 5.0
    level_gain = 0.0 if levels is None else max(0.0, levels - start_level)
    c4 = _palette_count(summary, 4)
    small4 = _small_components(summary, 4)

    features = {
        "level_gain": level_gain,
        "done": 1.0 if isinstance(summary, Mapping) and summary.get("done") else 0.0,
        "cost": float(len(actions)),
        "clicks": float(sum(1 for a in actions if len(a) == 3 and a[0] == 6)),
        "alt34": float(_count_alternations(actions)),
        "mentions_square": 1.0 if _has_words(text, ("square", "x=49", "49,y22", "49,y25")) else 0.0,
        "mentions_plus": 1.0 if _has_words(text, ("plus", "fragment", "x=12", "12,y")) else 0.0,
        "mentions_meter": 1.0 if _has_words(text, ("meter", "phase", "threshold", "register", "oscillation")) else 0.0,
        "mentions_dark": 1.0 if _has_words(text, ("dark", "portal", "left object")) else 0.0,
        "mentions_orange": 1.0 if _has_words(text, ("orange", "hazard", "cluster")) else 0.0,
        "mentions_magenta": 1.0 if _has_words(text, ("magenta", "singleton", "seed")) else 0.0,
        "mentions_green": 1.0 if _has_words(text, ("green", "gate")) else 0.0,
        "mentions_blue_hub": 1.0 if _has_words(text, ("blue", "hub", "disk")) else 0.0,
        "mentions_shepherd": 1.0 if _has_words(text, ("shepherd", "steer", "steering", "lower cluster")) else 0.0,
        "mentions_orbit": 1.0 if _has_words(text, ("orbit", "timer", "deplete", "meter reaches zero")) else 0.0,
        "mentions_sweep": 1.0 if _has_words(text, ("sweep", "visible component", "component clicks")) else 0.0,
        "mentions_bridge": 1.0 if _has_words(text, ("bridge", "arch", "symmetric")) else 0.0,
        "mentions_finish": 1.0 if _has_words(text, ("finish", "complete the level", "level-up", "level up")) else 0.0,
        "mentions_known_good": 1.0 if _has_words(text, ("known_good", "known-good", "known good", "safe_prefix")) else 0.0,
        "mentions_remaining": 1.0 if _has_words(text, ("remaining", "last visible", "bottom cleanup", "singletons are gone")) else 0.0,
        "clicks_square": 1.0 if _contains_click(actions, (49, 22), radius=4) else 0.0,
        "clicks_fragment": 1.0 if (
            _contains_click(actions, (12, 5), radius=1) or _contains_click(actions, (12, 9), radius=1)
        ) else 0.0,
        "clicks_bottom": float(sum(1 for a in actions if len(a) == 3 and a[0] == 6 and a[2] >= 40)),
        "clicks_hub": 1.0 if _contains_click(actions, (32, 19), radius=4) else 0.0,
        "c4": -1.0 if c4 is None else c4,
        "c4_near_62": 0.0 if c4 is None else max(0.0, 1.0 - abs(c4 - 62.0) / 62.0),
        "c4_near_94": 0.0 if c4 is None else max(0.0, 1.0 - abs(c4 - 94.0) / 94.0),
        "small4_count": float(len(small4)),
        "has_fragment_12_5": 1.0 if any(x.get("x") == 12 and x.get("y") == 5 for x in small4) else 0.0,
        "has_fragment_12_9": 1.0 if any(x.get("x") == 12 and x.get("y") == 9 for x in small4) else 0.0,
        "dist_square": 64.0 if _distance_to(summary, (49, 22)) is None else _distance_to(summary, (49, 22)) or 0.0,
        "dist_plus": 64.0 if _distance_to(summary, (10, 19)) is None else _distance_to(summary, (10, 19)) or 0.0,
        "failed": 1.0 if candidate.failed else 0.0,
    }
    features["orange_finish_signal"] = (
        features["mentions_orange"]
        * features["mentions_shepherd"]
        * features["mentions_finish"]
        * features["clicks_hub"]
    )
    features["cleanup_finish_signal"] = (
        features["orange_finish_signal"]
        * max(features["mentions_known_good"], features["mentions_remaining"])
    )
    features["orbit_signal"] = features["mentions_orange"] * features["mentions_orbit"] * features["clicks_hub"]
    features["gate_signal"] = features["mentions_magenta"] * features["mentions_green"] * features["clicks_hub"]
    features["sweep_signal"] = features["mentions_sweep"] * features["clicks_hub"]
    return features


def make_ka59_query_worlds() -> list[QueryWorld]:
    return [
        QueryWorld(
            "meter_then_square",
            "The yellow square is real only after a color-4 phase/meter precondition.",
            {
                "mentions_meter": -3.0,
                "mentions_square": -1.2,
                "alt34": -0.025,
                "c4_near_62": -4.0,
                "dist_square": 0.035,
                "cost": 0.004,
                "failed": 2.0,
            },
            prior=1.35,
        ),
        QueryWorld(
            "plus_fragments_selectable",
            "The large plus exposes tiny color-4 fragments that must be selected.",
            {
                "mentions_plus": -2.2,
                "clicks_fragment": -3.0,
                "has_fragment_12_5": -1.5,
                "has_fragment_12_9": -1.5,
                "dist_plus": 0.025,
                "cost": 0.006,
                "failed": 2.25,
            },
            prior=1.15,
        ),
        QueryWorld(
            "left_dark_object",
            "A left-side dark/portal-like object is the true post-plus object.",
            {
                "mentions_dark": -3.0,
                "mentions_plus": -0.7,
                "has_fragment_12_5": -0.8,
                "has_fragment_12_9": -0.8,
                "dist_plus": 0.02,
                "cost": 0.006,
                "failed": 2.5,
            },
            prior=0.8,
        ),
        QueryWorld(
            "direct_square_decoy",
            "Direct square contact is tempting but counterevidence makes it weak.",
            {
                "mentions_square": -1.0,
                "clicks_square": -0.8,
                "dist_square": 0.02,
                "mentions_meter": 1.1,
                "mentions_plus": 0.7,
                "failed": 3.0,
                "cost": 0.008,
            },
            prior=0.35,
        ),
    ]


def make_hardworld_query_worlds() -> list[QueryWorld]:
    worlds = make_ka59_query_worlds()
    worlds.extend(
        [
            QueryWorld(
                "orange_shepherd_finish",
                "Active orange clusters must be steered through remaining bottom singletons, then finished at the hub.",
                {
                    "orange_finish_signal": -5.0,
                    "cleanup_finish_signal": -4.0,
                    "mentions_orange": -0.7,
                    "mentions_shepherd": -0.8,
                    "mentions_finish": -0.6,
                    "mentions_magenta": -0.9,
                    "mentions_blue_hub": -0.8,
                    "clicks_bottom": -0.055,
                    "clicks_hub": -0.8,
                    "cost": 0.035,
                    "mentions_sweep": 2.0,
                    "mentions_green": 0.5,
                    "mentions_orbit": 1.2,
                    "failed": 2.0,
                },
                prior=1.25,
            ),
            QueryWorld(
                "orange_orbit_meter",
                "A safe orbit keeps hazards away while depleting a hidden meter before the hub finish.",
                {
                    "orbit_signal": -3.0,
                    "mentions_orange": -1.5,
                    "mentions_orbit": -1.2,
                    "mentions_meter": -1.2,
                    "clicks_bottom": -0.035,
                    "clicks_hub": -0.45,
                    "cost": 0.012,
                    "failed": 2.25,
                },
                prior=0.95,
            ),
            QueryWorld(
                "green_magenta_gate_pairing",
                "Magenta seeds must be paired through green gates before a hub click.",
                {
                    "gate_signal": -2.6,
                    "mentions_magenta": -1.5,
                    "mentions_green": -1.6,
                    "mentions_blue_hub": -0.6,
                    "clicks_bottom": -0.025,
                    "clicks_hub": -0.6,
                    "mentions_orange": 0.35,
                    "cost": 0.012,
                    "failed": 2.0,
                },
                prior=0.9,
            ),
            QueryWorld(
                "visible_component_sweep",
                "The board levels by sweeping visible salient components in a good order.",
                {
                    "sweep_signal": -3.0,
                    "mentions_sweep": -2.4,
                    "mentions_magenta": -0.8,
                    "mentions_green": -0.6,
                    "mentions_orange": -0.6,
                    "clicks_bottom": -0.02,
                    "clicks_hub": -0.5,
                    "cost": 0.02,
                    "failed": 2.0,
                },
                prior=0.75,
            ),
        ]
    )
    return worlds


def world_energy(world: QueryWorld, candidate: Candidate) -> float:
    features = feature_vector(candidate)
    energy = 4.0
    for key, weight in world.terms.items():
        energy += weight * features.get(key, 0.0)
    if world.world_id.startswith("orange_") and not features["mentions_orange"]:
        energy += 6.0
    if world.world_id == "green_magenta_gate_pairing" and not (
        features["mentions_green"] or features["mentions_magenta"]
    ):
        energy += 5.0
    if world.world_id == "visible_component_sweep" and not features["mentions_sweep"]:
        energy += 5.0
    if features["done"] and features["level_gain"] <= 0:
        energy += 8.0
    if features["level_gain"] > 0:
        energy -= 25.0 * features["level_gain"]
    return max(0.0, energy)


def posterior_scores(
    candidate: Candidate,
    worlds: Sequence[QueryWorld],
    *,
    beta: float = 1.0,
) -> tuple[float, list[dict[str, Any]], float]:
    terms: list[float] = []
    rows: list[dict[str, Any]] = []
    for world in worlds:
        energy = world_energy(world, candidate)
        log_prior = math.log(max(world.prior, 1e-12))
        log_weight = log_prior - beta * energy
        terms.append(log_weight)
        rows.append(
            {
                "world_id": world.world_id,
                "energy": round(energy, 6),
                "log_weight": round(log_weight, 6),
            }
        )
    log_z = _logsumexp(terms)
    for row, term in zip(rows, terms, strict=True):
        row["posterior"] = round(math.exp(term - log_z), 6) if not math.isinf(log_z) else 0.0
    best_energy = min((row["energy"] for row in rows), default=math.inf)
    return log_z, rows, best_energy


def semiring_value(candidate: Candidate, best_energy: float) -> SemiringValue:
    features = feature_vector(candidate)
    novelty = (
        features["mentions_meter"]
        + features["mentions_plus"]
        + features["mentions_dark"]
        + 2.0 * features["orange_finish_signal"]
        + 2.5 * features["cleanup_finish_signal"]
        + 1.35 * features["orbit_signal"]
        + 1.25 * features["gate_signal"]
        + 1.0 * features["sweep_signal"]
        + 0.25 * min(features["clicks"], 4.0)
    )
    risk = 2.5 * features["done"] + 3.0 * features["failed"] + 0.02 * features["cost"]
    return SemiringValue(
        level_gain=features["level_gain"],
        neg_energy=-best_energy,
        novelty=novelty,
        neg_cost=-features["cost"],
        neg_risk=-risk,
    )


def _failure_counts(
    counterexamples: Sequence[Candidate],
    worlds: Sequence[QueryWorld],
    *,
    beta: float,
) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    family_counts: dict[str, int] = {}
    world_counts: dict[str, int] = {}
    failed_world_by_id: dict[str, str] = {}
    for counterexample in counterexamples:
        family = proposal_family(counterexample)
        family_counts[family] = family_counts.get(family, 0) + 1
        _, posterior, _ = posterior_scores(counterexample, worlds, beta=beta)
        if posterior:
            world_id = min(posterior, key=lambda row: row["energy"])["world_id"]
            world_counts[world_id] = world_counts.get(world_id, 0) + 1
            failed_world_by_id[counterexample.candidate_id] = world_id
    return family_counts, world_counts, failed_world_by_id


def rank_candidates(
    candidates: Sequence[Candidate],
    worlds: Sequence[QueryWorld] | None = None,
    *,
    beta: float = 1.0,
    failed_family_counts: Mapping[str, int] | None = None,
    failed_world_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    worlds = list(worlds or make_hardworld_query_worlds())
    failed_family_counts = failed_family_counts or {}
    failed_world_counts = failed_world_counts or {}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        log_z, posterior, best_energy = posterior_scores(candidate, worlds, beta=beta)
        best_world = min(posterior, key=lambda row: row["energy"])["world_id"] if posterior else None
        family = proposal_family(candidate)
        family_failures = failed_family_counts.get(family, 0)
        world_failures = failed_world_counts.get(best_world or "", 0)
        if candidate.failed:
            family_failures = max(0, family_failures - 1)
            world_failures = max(0, world_failures - 1)
        family_penalty = 1.35 * family_failures
        world_penalty = 0.45 * world_failures
        exact_failure_penalty = 20.0 if candidate.failed else 0.0
        total_penalty = family_penalty + world_penalty + exact_failure_penalty
        value = semiring_value(candidate, best_energy + total_penalty)
        if total_penalty:
            value = SemiringValue(
                value.level_gain,
                value.neg_energy,
                value.novelty,
                value.neg_cost,
                value.neg_risk - total_penalty,
            )
        rows.append(
            {
                "rank": 0,
                "candidate_id": candidate.candidate_id,
                "path_integral_score": round(log_z, 6),
                "semiring": value.as_list(),
                "best_world": best_world,
                "family": family,
                "failure_penalty": {
                    "family_failures": family_failures,
                    "exact_failure": bool(candidate.failed),
                    "world_failures": world_failures,
                    "total": round(total_penalty, 6),
                },
                "worlds": sorted(posterior, key=lambda row: row["posterior"], reverse=True),
                "features": {k: round(v, 6) for k, v in feature_vector(candidate).items()},
                "failed": candidate.failed,
                "n_actions": len(candidate.actions),
            }
        )
    rows.sort(key=lambda row: (row["semiring"], row["path_integral_score"]), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def load_candidate(path: str | Path, *, failed: bool = False) -> Candidate:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} does not contain a JSON object")
    proposal = data
    summary = data.get("final_summary") if isinstance(data.get("final_summary"), Mapping) else None
    return Candidate(
        candidate_id=str(data.get("proposal_id") or Path(path).stem),
        actions=_actions_from_mapping(data),
        text=_flatten_text(data),
        proposal=proposal,
        summary=summary,
        failed=failed,
    )


def load_candidates(paths: Sequence[str | Path], failed_ids: set[str] | None = None) -> list[Candidate]:
    failed_ids = failed_ids or set()
    out = []
    for path in paths:
        candidate = load_candidate(path, failed=Path(path).name in failed_ids)
        if candidate.candidate_id in failed_ids:
            candidate = Candidate(
                candidate.candidate_id,
                candidate.actions,
                candidate.text,
                candidate.proposal,
                candidate.summary,
                True,
            )
        out.append(candidate)
    return out


def write_ranking(
    proposal_paths: Sequence[str | Path],
    out_path: str | Path,
    *,
    counterexample_paths: Sequence[str | Path] = (),
    beta: float = 1.0,
) -> dict[str, Any]:
    counterexamples = [load_candidate(path, failed=True) for path in counterexample_paths]
    failed_ids = {c.candidate_id for c in counterexamples}
    failed_ids.update(Path(path).name for path in counterexample_paths)
    candidates = load_candidates(proposal_paths, failed_ids=failed_ids)
    candidates.extend(c for c in counterexamples if c.candidate_id not in {x.candidate_id for x in candidates})
    worlds = make_hardworld_query_worlds()
    failed_family_counts, failed_world_counts, failed_world_by_id = _failure_counts(
        counterexamples,
        worlds,
        beta=beta,
    )
    ranked = rank_candidates(
        candidates,
        worlds=worlds,
        beta=beta,
        failed_family_counts=failed_family_counts,
        failed_world_counts=failed_world_counts,
    )
    packet = {
        "experiment": "E139 many-world query semiring ranker",
        "beta": beta,
        "worlds": [
            {"world_id": w.world_id, "description": w.description, "prior": w.prior, "terms": dict(w.terms)}
            for w in worlds
        ],
        "failure_model": {
            "families": failed_family_counts,
            "worlds": failed_world_counts,
            "failed_world_by_id": failed_world_by_id,
        },
        "n_candidates": len(candidates),
        "winner": ranked[0] if ranked else None,
        "ranked": ranked,
    }
    Path(out_path).write_text(json.dumps(packet, indent=2, sort_keys=True))
    return packet


def _expand(paths: Sequence[str]) -> list[str]:
    out: list[str] = []
    for path in paths:
        matches = sorted(glob.glob(path))
        out.extend(matches or [path])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rank ARC source-free probes with E139 many-world semiring scoring.")
    ap.add_argument("out_json")
    ap.add_argument("proposal_json", nargs="+")
    ap.add_argument("--counterexample", action="append", default=[])
    ap.add_argument("--beta", type=float, default=1.0)
    args = ap.parse_args(argv)

    packet = write_ranking(
        _expand(args.proposal_json),
        args.out_json,
        counterexample_paths=_expand(args.counterexample),
        beta=args.beta,
    )
    winner = packet.get("winner") or {}
    print(
        f"[e139] ranked {packet['n_candidates']} candidates; "
        f"winner={winner.get('candidate_id')} best_world={winner.get('best_world')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
