from experiments.e139.hybrid_rank import rank_hybrid
from experiments.e139.manyworld_semiring import Candidate


def packet():
    return {
        "game": "toy",
        "frontier_level": 5,
        "win": 6,
        "candidate_schemas": [
            {"type": "good_schema", "support_frac": 1.0, "loo_success": 1.0},
        ],
        "goal_condition_schemas": [
            {"type": "goal_schema", "support_frac": 1.0, "loo_success": 1.0},
        ],
    }


def counterexample(cid, text, actions):
    return Candidate(
        candidate_id=cid,
        actions=tuple(tuple(a) for a in actions),
        text=text.lower(),
        proposal={"proposal_id": cid, "executed": actions, "start_level": 5},
        summary=None,
        failed=True,
    )


def test_hybrid_keeps_schema_quality_above_raw_diversity():
    grounded = {
        "proposal_id": "grounded-plus",
        "schema_id": "good_schema",
        "goal_schema_id": "goal_schema",
        "hypothesis": "large plus fragment x=12,y9 click fragment",
        "role_bindings": {"fragment": "observed small component"},
        "probe_plan": [[6, 12, 9], [1]],
        "expected_deltas": ["level rises"],
        "fallback_repairs": ["try sibling fragment"],
    }
    weak = {
        "proposal_id": "weak-other",
        "hypothesis": "try something unrelated",
        "probe_plan": [[1]],
    }

    ranking = rank_hybrid(packet(), [weak, grounded])

    assert ranking["winner"]["proposal_id"] == "grounded-plus"
    assert ranking["winner"]["e138_score"] > ranking["ranked"][1]["e138_score"]


def test_hybrid_uses_counterexamples_to_suppress_failed_family():
    meter = {
        "proposal_id": "repair-e138-meter-threshold-extra-down",
        "schema_id": "good_schema",
        "goal_schema_id": "goal_schema",
        "hypothesis": "meter threshold register oscillation square x=49,y22",
        "role_bindings": {"meter": "observed"},
        "probe_plan": [[3], [4]] * 20,
        "expected_deltas": ["c4 reaches 62"],
        "fallback_repairs": ["try extra down"],
    }
    plus = {
        "proposal_id": "repair-e138-click-plus-fragments",
        "schema_id": "good_schema",
        "goal_schema_id": "goal_schema",
        "hypothesis": "large plus fragment x=12,y5 x=12,y9 click plus fragment",
        "role_bindings": {"fragment": "observed"},
        "probe_plan": [[3], [3], [6, 12, 9]],
        "expected_deltas": ["fragment binds"],
        "fallback_repairs": ["try other fragment"],
    }
    failures = [
        counterexample(f"meter-{i}", "meter threshold register square", [[3], [4]] * 10)
        for i in range(4)
    ]

    ranking = rank_hybrid(packet(), [meter, plus], counterexamples=failures)

    assert ranking["winner"]["proposal_id"] == "repair-e138-click-plus-fragments"
    assert ranking["ranked"][1]["e139"]["failure_penalty"]["family_failures"] == 4


def test_hybrid_never_reselects_exact_failed_proposal():
    failed = {
        "proposal_id": "repair-e138-click-plus-fragments",
        "schema_id": "good_schema",
        "goal_schema_id": "goal_schema",
        "hypothesis": "large plus fragment x=12,y9",
        "role_bindings": {"fragment": "observed"},
        "probe_plan": [[6, 12, 9]],
        "expected_deltas": ["level rises"],
        "fallback_repairs": ["try sibling"],
    }
    fallback = {
        "proposal_id": "direct-yellow-square",
        "schema_id": "good_schema",
        "goal_schema_id": "goal_schema",
        "hypothesis": "square fallback x=49,y22",
        "role_bindings": {"square": "observed"},
        "probe_plan": [[6, 49, 22]],
        "expected_deltas": ["level rises"],
        "fallback_repairs": ["try down"],
    }
    failures = [counterexample("repair-e138-click-plus-fragments", "large plus fragment x=12,y9", [[6, 12, 9]])]

    ranking = rank_hybrid(packet(), [failed, fallback], counterexamples=failures)

    assert ranking["winner"]["proposal_id"] == "direct-yellow-square"
    assert ranking["ranked"][1]["e139"]["failure_penalty"]["exact_failure"] is True
