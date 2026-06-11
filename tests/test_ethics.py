"""Tests for the ethics module: aggregators, constraints, parliament."""

from openworld import (
    Action,
    Constraint,
    Delegate,
    MoralParliament,
    WorldState,
    constrained,
    lexicographic,
    maximin,
    permitted_actions,
    weighted_sum,
)


def test_weighted_sum_matches_classical_aggregate():
    scores = {"welfare": 3.0, "fairness": -1.0}
    assert weighted_sum(scores, {"welfare": 1.0, "fairness": 0.5}) == 2.5


def test_maximin_judges_by_worst_component():
    assert maximin({"a": 5.0, "b": -2.0}) == -2.0
    # A big gain to the best-off cannot outweigh the worst-off (unlike a sum).
    assert maximin({"a": 100.0, "b": -2.0}) == maximin({"a": 5.0, "b": -2.0})


def test_lexicographic_priority_is_untradeable():
    high_safety = lexicographic({"safety": 1.0, "profit": 0.0}, ["safety", "profit"])
    low_safety_rich = lexicographic({"safety": 0.0, "profit": 1000.0}, ["safety", "profit"])
    assert high_safety > low_safety_rich  # no profit buys back lost safety


def test_constraint_vetoes_actions():
    never_abandon = Constraint(
        name="never_abandon_critical",
        forbidden=lambda s, a: s["critical_waiting"] > 0 and a.name != "treat_critical",
    )
    state = WorldState({"critical_waiting": 2})
    allowed = permitted_actions(state, ["treat_critical", "treat_moderate", "wait"],
                                [never_abandon])
    assert allowed == ["treat_critical"]
    state = WorldState({"critical_waiting": 0})
    assert len(permitted_actions(state, ["treat_critical", "wait"], [never_abandon])) == 2


def test_constrained_policy_never_violates():
    never_wait = Constraint(name="never_wait", forbidden=lambda s, a: a.name == "wait")
    greedy_waiter = lambda state, actions: Action("wait")  # wants to violate
    policy = constrained(greedy_waiter, [never_wait])
    choice = policy(WorldState({"x": 1}), ["work", "wait"])
    assert choice.name != "wait"


def test_constrained_policy_falls_back_when_nothing_permitted():
    forbid_all = Constraint(name="all", forbidden=lambda s, a: True)
    policy = constrained(lambda s, a: Action(a[0]), [forbid_all])
    assert policy(WorldState({}), ["go"]).name == "noop"


def test_parliament_borda_vote_and_hedging():
    # Two delegates prefer opposite extremes; a third prefers the middle.
    # Borda should elect the broadly acceptable middle option.
    def ranker(order):
        return lambda state, actions, simulate: sorted(actions, key=order.index)

    parliament = MoralParliament(delegates=[
        Delegate("maximizer", ranker(["high", "mid", "low"])),
        Delegate("minimizer", ranker(["low", "mid", "high"])),
        Delegate("moderate", ranker(["mid", "high", "low"])),
    ])
    choice = parliament.choose(WorldState({}), ["high", "mid", "low"],
                               simulate=lambda s, a: {})
    assert choice == "mid"


def test_parliament_credence_weighting():
    def ranker(order):
        return lambda state, actions, simulate: sorted(actions, key=order.index)

    parliament = MoralParliament(delegates=[
        Delegate("dominant", ranker(["a", "b"]), credence=3.0),
        Delegate("minor", ranker(["b", "a"]), credence=1.0),
    ])
    assert parliament.choose(WorldState({}), ["a", "b"], lambda s, a: {}) == "a"
