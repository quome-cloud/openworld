"""Tests for PhasedTransition: rule changes over time, verified ahead of run."""

import pytest

from openworld import Action, World
from openworld.compose import CompositeWorld
from openworld.transition import PhasedTransition, Transition


class AddTransition(Transition):
    def __init__(self, amount):
        self.amount = amount

    def step(self, state, action):
        s = state.copy()
        if action.name == "work":
            s["output"] += self.amount
        return s


def make_phased(trigger):
    return PhasedTransition([(0, AddTransition(1)), (trigger, AddTransition(10))])


def run(world, n, action="work"):
    for _ in range(n):
        world.step(Action(action))
    return world.state


def make_world(transition):
    return World(name="w", description="phased", initial_state={"output": 0},
                 actions=["work", "wait"], transition=transition)


def test_step_count_trigger_switches_regime_permanently():
    w = make_world(make_phased(3))
    s = run(w, 6)
    # steps 0,1,2 in phase 0 (+1 each); steps 3,4,5 in phase 1 (+10 each)
    assert s["output"] == 3 + 30
    assert s["_phase"] == 1
    assert s["_phase_steps"] == 6


def test_predicate_trigger_fires_on_state():
    t = PhasedTransition([(0, AddTransition(1)),
                          (lambda s: s["output"] >= 2, AddTransition(100))])
    w = make_world(t)
    s = run(w, 4)
    # +1, +1 (now output==2), then trigger fires: +100, +100
    assert s["output"] == 2 + 200
    assert s["_phase"] == 1


def test_advance_is_irreversible_even_if_predicate_goes_false():
    t = PhasedTransition([(0, AddTransition(1)),
                          (lambda s: s["output"] == 2, AddTransition(-1))])
    w = make_world(t)
    s = run(w, 5)
    # +1, +1 -> trigger (output==2) -> -1 (output 1, predicate now false), -1, -1
    assert s["output"] == -1
    assert s["_phase"] == 1


def test_phase_zero_trigger_is_ignored():
    t = PhasedTransition([(999, AddTransition(5))])
    w = make_world(t)
    s = run(w, 2)
    assert s["output"] == 10
    assert s["_phase"] == 0


def test_custom_record_key():
    t = PhasedTransition([(0, AddTransition(1)), (1, AddTransition(2))],
                         record_key="_era")
    w = make_world(t)
    s = run(w, 2)
    assert s["_era"] == 1 and s["_era_steps"] == 2
    assert "_phase" not in s


def test_empty_phases_rejected():
    with pytest.raises(ValueError):
        PhasedTransition([])


def test_phased_world_composes_as_composite_child():
    city = make_world(make_phased(2))
    comp = CompositeWorld(name="c", children={"city": city})
    for _ in range(4):
        comp.step(Action("city:work"))
    assert comp.state["city"]["output"] == 2 + 20
    assert comp.state["city"]["_phase"] == 1
