"""Branch-covering verification gate + LLMTransition parse-failure instrumentation."""

from openworld.state import Action, WorldState
from openworld.transition import LLMTransition
from openworld.llm import MockLLM
from openworld.verify import Verifier

_INIT = {"backlog": 12, "shipped": 0, "bugs": 0, "debt": 0}
_INV = [("counters never negative", lambda s: all(v >= 0 for v in s.values()))]
_ACTIONS = [Action("ship")]

# ship without the backlog>0 guard: fine from the initial state, negative from backlog==0
_BRANCH_FAULT = (
    "def transition(state, action):\n"
    "    s = dict(state)\n"
    "    if action['name'] == 'ship':\n"
    "        s['backlog'] -= 1; s['shipped'] += 1\n"
    "    return s\n"
)


def _verifier(probe_states):
    return Verifier(initial_state=WorldState(dict(_INIT)), sample_actions=_ACTIONS,
                    invariants=_INV, probe_states=probe_states)


def test_single_state_gate_misses_branch_only_fault():
    ok, _ = _verifier([]).check_behavior(_BRANCH_FAULT)
    assert ok is True  # initial backlog=12 -> ship is invariant-clean; fault not seen


def test_branch_covering_gate_catches_it():
    ok, msg = _verifier([{"backlog": 0, "shipped": 9, "bugs": 0, "debt": 0}]).check_behavior(_BRANCH_FAULT)
    assert ok is False and "invariant" in msg.lower()


def test_llm_transition_tracks_parse_failures():
    t = LLMTransition(MockLLM(["not json at all"]), description="w")
    s = WorldState(dict(_INIT))
    out = t.step(s, Action("ship"))
    assert out == s                       # unparseable reply -> no-op fallback
    assert t.parse_failures == 1 and t.steps == 1 and t.parse_failure_rate == 1.0
