from experiments.e137.goal_condition import induce_goal_conditions


def _demo(start_colors, pre_colors, post_colors, start_n=5, pre_n=8):
    return {
        "start_summary": {"colors": start_colors, "n_objects": start_n},
        "pre_win_summary": {"colors": pre_colors, "n_objects": pre_n},
        "post_win_summary": {"colors": post_colors, "n_objects": pre_n},
    }


def test_finds_color_consistently_added_at_win():
    # color 7 appears across the level-up in every demo -> a goal-condition schema
    demos = [
        _demo([0, 3], [0, 3], [0, 3, 7]),
        _demo([0, 4], [0, 4], [0, 4, 7]),
        _demo([0, 5], [0, 5], [0, 5, 7]),
    ]
    schemas = induce_goal_conditions(demos)
    top = [s for s in schemas if s["type"] == "win_adds_color" and s["pattern"] == 7]
    assert top and top[0]["loo_success"] == 1.0 and top[0]["support_frac"] == 1.0


def test_object_count_direction_and_invariant():
    demos = [
        _demo([0, 9], [0, 9], [0, 9], start_n=4, pre_n=10),
        _demo([0, 9], [0, 9], [0, 9], start_n=6, pre_n=12),
    ]
    schemas = induce_goal_conditions(demos)
    assert any(s["type"] == "object_count_direction" and s["pattern"] == "grows" for s in schemas)
    assert any(s["type"] == "pre_win_color_invariant" and 9 in s["pattern"] for s in schemas)
