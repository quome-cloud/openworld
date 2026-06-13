"""Tests for bridging-world improvements and World 1 persona generator."""

import math
import pytest

from openworld import Action, MockLLM, World
from openworld.compose import (
    AGG_KEY,
    Aggregator,
    Bridge,
    CompositeWorld,
    bridge_worlds,
    compile_bridge,
)
from openworld.sandbox import load_transition_code, run_fn
from openworld.transition import CodeTransition, FunctionTransition
from experiments.bridging.personas import (
    ISSUES,
    Persona,
    generate_personas,
    generate_sbm_edges,
)


# ---------------------------------------------------------------------------
# Improvement #1: CodeTransition caches the compiled callable
# ---------------------------------------------------------------------------

def test_code_transition_compiles_once():
    """CodeTransition._fn is set at __init__ time and stays the same object across calls."""
    from openworld.state import WorldState, Action as Act
    code = "def transition(state, action):\n    state['x'] = state.get('x', 0) + 1\n    return state"
    ct = CodeTransition(code)
    assert callable(ct._fn)
    fn1 = ct._fn
    ct.step(WorldState({"x": 0}), Act("inc"))
    ct.step(WorldState({"x": 1}), Act("inc"))
    assert ct._fn is fn1, "Cached _fn should not be replaced across calls"


def test_code_transition_step_uses_cached_fn():
    """CodeTransition.step produces correct output using the cached function."""
    from openworld.state import WorldState, Action as Act
    code = "def transition(state, action):\n    state['count'] += action['params'].get('value', 1)\n    return state"
    ct = CodeTransition(code)
    ws = WorldState({"count": 5})
    result = ct.step(ws, Act("increment", params={"value": 3}))
    assert result["count"] == 8


# ---------------------------------------------------------------------------
# Improvement #2: compile_bridge derives sample states from World objects
# ---------------------------------------------------------------------------

def _make_counter_world(name, start=0):
    def _step(state, action):
        s = dict(state)
        if action["name"] == "inc":
            s["count"] += 1
        return s
    return World(
        name=name,
        description="a counter",
        initial_state={"count": start},
        actions=["inc"],
        transition=FunctionTransition(_step),
    )


_PASSTHROUGH_BRIDGE_CODE = (
    "```python\n"
    "def transition(state, action):\n"
    "    return {k: dict(v) if isinstance(v, dict) else v for k, v in state.items()}\n"
    "```"
)


def test_compile_bridge_derives_samples_from_world_objects():
    """compile_bridge with world_a/world_b sets sample states automatically."""
    wa = _make_counter_world("alpha", start=10)
    wb = _make_counter_world("beta", start=20)
    bridge = compile_bridge(
        MockLLM([_PASSTHROUGH_BRIDGE_CODE]),
        "alpha-beta", "alpha", "beta",
        "Pass-through bridge: no state changes.",
        world_a=wa, world_b=wb,
    )
    assert bridge.name == "alpha-beta"
    assert bridge.a == "alpha"
    assert bridge.b == "beta"


def test_compile_bridge_explicit_samples_override_worlds():
    """Explicit sample_a/sample_b take precedence over world initial states."""
    wa = _make_counter_world("x", start=99)
    wb = _make_counter_world("y", start=99)
    bridge = compile_bridge(
        MockLLM([_PASSTHROUGH_BRIDGE_CODE]),
        "xy", "x", "y",
        "Pass-through bridge.",
        world_a=wa, world_b=wb,
        sample_a={"count": 1},
        sample_b={"count": 2},
    )
    assert bridge is not None


# ---------------------------------------------------------------------------
# Improvement #3: CompositeWorld validates bridge endpoints at construction
# ---------------------------------------------------------------------------

def test_composite_raises_on_unknown_bridge_endpoint():
    """CompositeWorld raises ValueError when a bridge references a missing child."""
    wa = _make_counter_world("alpha")
    wb = _make_counter_world("beta")
    bad_bridge = Bridge(
        name="bad",
        a="alpha",
        b="nonexistent",   # not in children
        transition=None,
    )
    with pytest.raises(ValueError, match="nonexistent"):
        CompositeWorld(
            name="test",
            children={"alpha": wa, "beta": wb},
            bridges=[bad_bridge],
        )


def test_composite_accepts_valid_bridge_endpoints():
    """CompositeWorld construction succeeds with valid bridge endpoints."""
    wa = _make_counter_world("alpha")
    wb = _make_counter_world("beta")
    bridge = Bridge(name="ab", a="alpha", b="beta", transition=None)
    comp = CompositeWorld(
        name="ok",
        children={"alpha": wa, "beta": wb},
        bridges=[bridge],
    )
    assert "alpha" in comp.children and "beta" in comp.children


# ---------------------------------------------------------------------------
# Improvement #4: bridge_worlds() helper
# ---------------------------------------------------------------------------

def test_bridge_worlds_returns_bridge_and_composite():
    """bridge_worlds() returns a (Bridge, CompositeWorld) pair."""
    wa = _make_counter_world("p")
    wb = _make_counter_world("q")
    bridge, composite = bridge_worlds(
        MockLLM([_PASSTHROUGH_BRIDGE_CODE]), wa, wb, "pq-bridge",
        "Pass-through bridge between p and q.",
    )
    assert bridge.name == "pq-bridge"
    assert isinstance(composite, CompositeWorld)
    assert "p" in composite.children and "q" in composite.children
    assert composite.name == "p+q"


# ---------------------------------------------------------------------------
# World 1: Persona generator tests
# ---------------------------------------------------------------------------

def test_generate_personas_returns_correct_count():
    personas = generate_personas(n=100, seed=0)
    assert len(personas) == 100


def test_persona_ideology_range():
    personas = generate_personas(n=200, seed=1)
    for p in personas:
        assert -1.0 <= p.latent_ideology <= 1.0


def test_persona_issue_weights_sum_to_one():
    personas = generate_personas(n=50, seed=2)
    for p in personas:
        total = sum(p.issue_weights[issue] for issue in ISSUES)
        assert abs(total - 1.0) < 1e-9, f"Issue weights sum to {total}"


def test_persona_ideal_stances_range():
    personas = generate_personas(n=50, seed=3)
    for p in personas:
        for issue in ISSUES:
            assert -2.0 <= p.ideal_stances[issue] <= 2.0


def test_persona_community_assignment():
    personas = generate_personas(n=300, seed=4)
    communities = set(p.network_community for p in personas)
    assert communities == {0, 1, 2, 3, 4, 5}  # all 6 communities populated


def test_persona_welfare_centrist_bundle():
    """A centrist (ideology ~0) persona should have high welfare at centrist bundle."""
    personas = generate_personas(n=300, seed=5)
    centrists = [p for p in personas if abs(p.latent_ideology) < 0.1]
    bundle = {issue: 0 for issue in ISSUES}
    assert len(centrists) > 0
    welfare_vals = [p.welfare(bundle) for p in centrists]
    assert all(w > 0.7 for w in welfare_vals), f"Min centrist welfare: {min(welfare_vals):.3f}"


def test_persona_welfare_partisan_bundle_polarized():
    """A strong progressive persona should have lower welfare at a strong conservative bundle."""
    personas = generate_personas(n=300, seed=6)
    progressives = [p for p in personas if p.latent_ideology < -0.6]
    conservative_bundle = {issue: 2 for issue in ISSUES}  # maximum conservative
    assert len(progressives) > 0
    welfare_vals = [p.welfare(conservative_bundle) for p in progressives]
    # Progressive personas should not do well under max-conservative policy
    assert all(w < 0.5 for w in welfare_vals), f"Max progressive welfare: {max(welfare_vals):.3f}"


def test_generate_sbm_edges_within_community_denser():
    """SBM edges should be denser within communities than across."""
    personas = generate_personas(n=200, seed=7)
    edges = generate_sbm_edges(personas, seed=7)

    within = 0
    across = 0
    for i, j in edges:
        if personas[i].network_community == personas[j].network_community:
            within += 1
        else:
            across += 1

    total_possible_within = sum(
        c * (c - 1) // 2
        for c in [
            sum(1 for p in personas if p.network_community == k) for k in range(6)
        ]
    )
    total_possible_across = len(personas) * (len(personas) - 1) // 2 - total_possible_within

    within_rate = within / total_possible_within if total_possible_within else 0
    across_rate = across / total_possible_across if total_possible_across else 0
    assert within_rate > across_rate * 5, (
        f"Expected much denser within-community edges; got within={within_rate:.3f} across={across_rate:.3f}"
    )


def test_generate_personas_reproducible():
    """Same seed produces identical personas."""
    p1 = generate_personas(n=50, seed=99)
    p2 = generate_personas(n=50, seed=99)
    assert all(
        a.latent_ideology == b.latent_ideology and
        a.network_community == b.network_community
        for a, b in zip(p1, p2)
    )


def test_generate_personas_different_seeds_differ():
    """Different seeds produce different outputs."""
    p1 = generate_personas(n=50, seed=1)
    p2 = generate_personas(n=50, seed=2)
    ideologies_1 = [p.latent_ideology for p in p1]
    ideologies_2 = [p.latent_ideology for p in p2]
    assert ideologies_1 != ideologies_2
