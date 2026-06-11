"""Tests for the agents-as-a-judge module."""

from openworld import Action, MockLLM, Trajectory, WorldState
from openworld.judge import Judge
from openworld.simulation import StepRecord


def test_choose_parses_json_choice():
    judge = Judge(MockLLM(['{"choice": 2, "reason": "handles the edge case"}']))
    assert judge.choose(["a", "b", "c"], context="pick best") == 2


def test_choose_falls_back_on_garbage_and_out_of_range():
    judge = Judge(MockLLM(["I like them all!"]))
    assert judge.choose(["a", "b"], default=1) == 1
    judge = Judge(MockLLM(['{"choice": 9}']))
    assert judge.choose(["a", "b"]) == 0


def test_choose_extracts_bare_integer():
    judge = Judge(MockLLM(["Option 1 is clearly best."]))
    assert judge.choose(["a", "b", "c"]) == 1


def test_choose_single_option_skips_llm():
    judge = Judge(MockLLM([]))  # would raise if called
    assert judge.choose(["only"]) == 0


def _tiny_trajectory():
    initial = WorldState({"x": 0})
    final = WorldState({"x": 3})
    return Trajectory(
        initial_state=initial,
        steps=[
            StepRecord(step=0, agent="a", action=Action("inc"), state=final, scores={"aggregate": 3.0})
        ],
    )


def test_score_trajectory_parses_and_clamps():
    judge = Judge(MockLLM(['{"score": 8.5, "reason": "good"}']))
    assert judge.score_trajectory(_tiny_trajectory(), rubric="reach x=3") == 8.5
    judge = Judge(MockLLM(['{"score": 14}']))
    assert judge.score_trajectory(_tiny_trajectory(), rubric="r") == 10.0


def test_score_trajectory_returns_none_on_garbage():
    judge = Judge(MockLLM(["no idea"]))
    assert judge.score_trajectory(_tiny_trajectory(), rubric="r") is None


def test_judge_prompt_includes_criteria_and_context():
    llm = MockLLM(['{"choice": 0}'])
    judge = Judge(llm, criteria="prefer minimal diffs")
    judge.choose(["patch A", "patch B"], context="fix the bug")
    prompt = llm.calls[0][-1]["content"]
    assert "prefer minimal diffs" in prompt
    assert "fix the bug" in prompt
    assert "Option 1" in prompt
