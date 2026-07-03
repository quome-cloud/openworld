from experiments.e139.manyworld_semiring import (
    Candidate,
    make_hardworld_query_worlds,
    make_ka59_query_worlds,
    rank_candidates,
)


def candidate(cid, text, actions, summary=None, failed=False):
    return Candidate(
        candidate_id=cid,
        actions=tuple(tuple(a) for a in actions),
        text=text.lower(),
        proposal={"proposal_id": cid, "start_level": 5},
        summary=summary,
        failed=failed,
    )


def test_meter_world_beats_direct_square_after_counterevidence():
    direct = candidate(
        "direct-square",
        "contact upper-right square x=49,y22",
        [[1], [1], [6, 49, 22]],
        {"levels": 5, "done": False, "cursor": {"zero": {"x": 49, "y": 22}}, "palette": {"4": 114}},
        failed=True,
    )
    meter = candidate(
        "meter-square",
        "meter phase threshold register oscillation then square x=49,y22",
        [[3], [4]] * 45 + [[1], [1], [2]],
        {"levels": 5, "done": False, "cursor": {"zero": {"x": 49, "y": 23}}, "palette": {"4": 62}},
    )

    ranked = rank_candidates([direct, meter], make_ka59_query_worlds())

    assert ranked[0]["candidate_id"] == "meter-square"
    assert ranked[0]["best_world"] == "meter_then_square"


def test_plus_fragment_world_uses_observed_small_components():
    plus = candidate(
        "plus-fragment",
        "large plus exposes fragment x=12,y5 x=12,y9 and click fragment",
        [[3]] * 16 + [[1]] * 3 + [[6, 12, 9]],
        {
            "levels": 5,
            "done": False,
            "cursor": {"zero": {"x": 10, "y": 19}},
            "palette": {"4": 94},
            "small": [{"c": 4, "n": 1, "x": 12, "y": 5}, {"c": 4, "n": 1, "x": 12, "y": 9}],
        },
    )
    generic = candidate("generic-square", "try the square again", [[6, 49, 22]])

    ranked = rank_candidates([generic, plus], make_ka59_query_worlds())

    assert ranked[0]["candidate_id"] == "plus-fragment"
    assert ranked[0]["best_world"] == "plus_fragments_selectable"


def test_level_gain_is_dominant_semiring_signal():
    solved = candidate(
        "solved",
        "ugly expensive branch",
        [[3], [4]] * 90,
        {"levels": 6, "done": False, "cursor": {"zero": {"x": 1, "y": 1}}, "palette": {"4": 20}},
    )
    elegant = candidate(
        "elegant",
        "meter square",
        [[3], [4], [1]],
        {"levels": 5, "done": False, "cursor": {"zero": {"x": 49, "y": 22}}, "palette": {"4": 62}},
    )

    ranked = rank_candidates([elegant, solved], make_ka59_query_worlds())

    assert ranked[0]["candidate_id"] == "solved"
    assert ranked[0]["semiring"][0] == 1.0


def test_hardworld_orange_finish_beats_naive_component_sweep():
    finish = candidate(
        "repair-orange-finish",
        "known-good prefix orange shepherd steering lower cluster remaining magenta singleton blue hub finish complete the level",
        [[6, 6, 40], [6, 44, 36], [6, 6, 52], [6, 48, 52], [6, 14, 54], [6, 32, 19]],
    )
    sweep = candidate(
        "visible-component-sweep",
        "visible component sweep magenta green orange components final hub",
        [[6, 3, 60], [6, 58, 59], [6, 14, 54], [6, 44, 53], [6, 6, 26], [6, 32, 19]],
    )

    ranked = rank_candidates([sweep, finish], make_hardworld_query_worlds())

    assert ranked[0]["candidate_id"] == "repair-orange-finish"
    assert ranked[0]["best_world"] == "orange_shepherd_finish"


def test_repeated_family_failures_force_world_diversification():
    meter_sibling = candidate(
        "repair-e138-meter-threshold-extra-down",
        "meter threshold register oscillation square x=49,y22",
        [[3], [4]] * 57 + [[1], [1], [2]],
    )
    plus = candidate(
        "repair-e138-click-plus-fragments",
        "large plus fragment x=12,y5 x=12,y9 click plus fragment",
        [[3]] * 16 + [[1]] * 3 + [[6, 12, 9]],
    )

    ranked = rank_candidates(
        [meter_sibling, plus],
        make_hardworld_query_worlds(),
        failed_family_counts={"meter_threshold": 4},
        failed_world_counts={"meter_then_square": 4},
    )

    assert ranked[0]["candidate_id"] == "repair-e138-click-plus-fragments"
    assert ranked[0]["best_world"] == "plus_fragments_selectable"
    assert ranked[1]["failure_penalty"]["family_failures"] == 4


def test_exact_failed_candidate_does_not_win_again():
    failed_meter = candidate(
        "repair-e138-meter-phase-square",
        "meter phase threshold register oscillation then square",
        [[3], [4]] * 50,
        failed=True,
    )
    weaker_new = candidate(
        "repair-click-square-down",
        "try square down finish",
        [[2], [6, 49, 22]],
    )

    ranked = rank_candidates([failed_meter, weaker_new], make_hardworld_query_worlds())

    assert ranked[0]["candidate_id"] == "repair-click-square-down"
    assert ranked[1]["failure_penalty"]["exact_failure"] is True
