from experiments.e138.judge_schema import normalize_action, rank_proposals, score_proposal


def _packet():
    return {
        "game": "toy",
        "frontier_level": 2,
        "win": 3,
        "candidate_schemas": [
            {
                "description": "final level-up action is usually click",
                "type": "last_action_kind",
                "support_frac": 1.0,
                "loo_success": 1.0,
            }
        ],
        "goal_condition_schemas": [
            {
                "description": "each level-up ADDS colour 7",
                "type": "win_adds_color",
                "support_frac": 0.75,
                "loo_success": 1.0,
            }
        ],
    }


def test_normalize_action_validates_arc_actions():
    assert normalize_action([6, 12, 34]) == [6, 12, 34]
    assert normalize_action(1) == [1]
    assert normalize_action([6, 99, 0]) is None
    assert normalize_action([9]) is None


def test_score_prefers_schema_grounded_small_plan():
    good = {
        "proposal_id": "grounded",
        "schema_id": "last_action_kind",
        "hypothesis": "click the object that completes the invariant",
        "role_bindings": {"target": "new color 7 object"},
        "probe_plan": [[6, 12, 34]],
        "expected_deltas": ["level rises by one"],
        "fallback_repairs": ["try sibling salient object"],
        "confidence": 0.6,
    }
    bad = {
        "proposal_id": "generic",
        "hypothesis": "random search",
        "probe_plan": [[6, 99, 0]],
    }
    assert score_proposal(_packet(), good).score > score_proposal(_packet(), bad).score


def test_rank_proposals_returns_winner():
    ranking = rank_proposals(
        _packet(),
        [
            {"proposal_id": "weak", "probe_plan": [[1]]},
            {"proposal_id": "strong", "schema_id": "win_adds_color", "probe_plan": [[6, 1, 2]]},
        ],
    )
    assert ranking["winner"]["proposal_id"] == "strong"
    assert ranking["ranked"][0]["rank"] == 1
