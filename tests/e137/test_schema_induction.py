import numpy as np

from experiments.e137.schema_induction import (
    StepRecord,
    build_packet,
    induce_schemas,
    segment_by_level,
)


def _frame(x, color=5):
    f = np.zeros((8, 8), dtype=int)
    f[3, x] = color
    return f


def test_segments_level_up_demos_from_records():
    records = [
        StepRecord(_frame(1), [1], _frame(2), 0, 0),
        StepRecord(_frame(2), [5], _frame(2, 6), 0, 1),
        StepRecord(_frame(2, 6), [1], _frame(3, 6), 1, 1),
        StepRecord(_frame(3, 6), [5], _frame(3, 7), 1, 2),
    ]
    demos = segment_by_level(records)
    assert [d["level"] for d in demos] == [1, 2]
    assert demos[0]["actions"] == [[1], [5]]
    assert demos[1]["last_action"] == [5]


def test_induce_schemas_prefers_repeated_procedure_suffix():
    demos = [
        {"action_kinds": ["move", "interact"], "action_signatures": ["1", "5"], "last_action": [5], "length": 2},
        {"action_kinds": ["move", "interact"], "action_signatures": ["2", "5"], "last_action": [5], "length": 2},
        {"action_kinds": ["move", "interact"], "action_signatures": ["3", "5"], "last_action": [5], "length": 2},
    ]
    schemas = induce_schemas(demos)
    assert schemas[0]["loo_success"] == 1.0
    assert any(s["type"] == "action_kind_suffix" and s["pattern"] == ["move", "interact"] for s in schemas)


def test_build_packet_contains_ranked_schema_context():
    records = [
        StepRecord(_frame(1), [1], _frame(2), 0, 0),
        StepRecord(_frame(2), [5], _frame(2, 6), 0, 1),
    ]
    packet = build_packet("toy", records, [[1], [5]], frontier_levels=1, win=2)
    assert packet["game"] == "toy"
    assert packet["remaining_levels"] == 1
    assert packet["n_solved_level_demos"] == 1
    assert packet["candidate_schemas"]

