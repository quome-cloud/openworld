"""Tests for causal-DAG input: parsing, SCM compilation, and the DAGPerceptor."""

import pytest

from openworld import (
    Action, CausalDAG, DAGPerceptor, Observation, PerceptionError,
    PerceptionGate, World, dag_to_schema, dag_to_world, from_spec, parse_dag,
    to_spec, validate_spec,
)

DAGITTY = """
dag {
bb="0,0,1,1"
"Income low" [exposure,pos="0.1,0.2"]
"Prenatal Stress" [exposure]
"Brain Dev" [latent]
"Bayley Score" [outcome]
// an edge comment
"Income low" -> "Prenatal Stress"
"Prenatal Stress" -> "Bayley Score"
"Brain Dev" -> "Bayley Score"
}
"""

DOT = "digraph { a -> b; b -> c; }"


def test_parse_dagitty_nodes_edges_and_tags():
    d = parse_dag(DAGITTY)
    assert set(d.nodes) == {"Income low", "Prenatal Stress", "Brain Dev", "Bayley Score"}
    assert ("Income low", "Prenatal Stress") in d.edges
    assert d.exposures() == ["Income low", "Prenatal Stress"]
    assert d.outcomes() == ["Bayley Score"]
    assert d.latents() == ["Brain Dev"]
    assert "Brain Dev" not in d.observed()        # latent is not observable


def test_parse_minimal_dot():
    d = parse_dag(DOT)
    assert set(d.nodes) == {"a", "b", "c"}
    assert d.edges == [("a", "b"), ("b", "c")]


def test_topo_order_and_acyclicity():
    d = parse_dag(DAGITTY)
    order = d.topo_order()
    # every edge goes forward in the topological order
    for a, b in d.edges:
        assert order.index(a) < order.index(b)
    assert d.is_acyclic()


def test_cycle_is_rejected():
    cyc = CausalDAG(nodes={"a": [], "b": []}, edges=[("a", "b"), ("b", "a")])
    assert not cyc.is_acyclic()
    with pytest.raises(ValueError):
        cyc.topo_order()
    with pytest.raises(ValueError):
        dag_to_world(cyc)


def test_dag_to_schema_excludes_latent():
    schema = dag_to_schema(parse_dag(DAGITTY))
    assert "brain_dev" not in schema                # latent excluded
    assert schema["income_low"] == (int, (0, 1))


def test_dag_compiles_to_scm_world_and_do_operator_propagates():
    d = parse_dag(DAGITTY)
    w = dag_to_world(d)
    assert "do_income_low" in w.actions and "observe" in w.actions
    # do(income_low := 1) must propagate downstream in one pass
    out = w.transition.step(w.initial_state, Action("do_income_low", params={"value": 1.0}))
    assert out["income_low"] == 1.0
    assert out["prenatal_stress"] == 1.0           # child fired
    assert out["bayley_score"] == 1.0              # grandchild fired
    # with no intervention and zero exogenous terms, nothing fires
    base = w.transition.step(w.initial_state, Action("observe"))
    assert base["bayley_score"] == 0.0


def test_perceptor_graph_mode_delta_and_gate():
    p = DAGPerceptor(mode="graph")
    obs = Observation("graph", DAGITTY)
    delta = p.perceive(obs)
    assert delta["dag_nodes"] and ["Income low", "Prenatal Stress"] == delta["dag_exposures"]
    assert ["Bayley Score"] == delta["dag_outcomes"]
    # the gate accepts only fields the perceptor declares it produces
    assert PerceptionGate().check(p, delta) == delta


def test_perceptor_schema_mode():
    p = DAGPerceptor(mode="schema")
    delta = p.perceive(Observation("graph", DAGITTY))
    assert "income_low" in delta["dag_schema"]
    assert delta["dag_schema"]["income_low"] == {"type": "int", "bounds": [0, 1]}


def test_perceptor_world_mode_yields_loadable_spec():
    p = DAGPerceptor(mode="world")
    delta = p.perceive(Observation("graph", DAGITTY))
    spec = delta["dag_world"]
    assert not validate_spec(spec)                 # a valid world spec
    w = from_spec(spec, allow_code=True)
    out = w.transition.step(w.initial_state, Action("do_income_low", params={"value": 1.0}))
    assert out["bayley_score"] == 1.0              # the reconstructed SCM runs


def test_to_world_convenience_method():
    w = DAGPerceptor().to_world(Observation("graph", DOT))
    assert isinstance(w, World)
    assert {"do_a", "do_b", "do_c", "observe"} <= set(w.actions)


def test_unknown_mode_rejected():
    with pytest.raises(PerceptionError):
        DAGPerceptor(mode="nonsense")


def test_world_observe_commits_dag_delta():
    w = World(name="g", description="graph intake",
              initial_state={"dag_nodes": [], "dag_edges": [], "dag_exposures": [],
                             "dag_outcomes": []},
              actions=["noop"])
    p = DAGPerceptor(mode="graph")
    w.observe(Observation("graph", DAGITTY), p)
    assert w.state["dag_outcomes"] == ["Bayley Score"]


def test_dag_perceptor_round_trips_through_spec():
    w = World(name="g", description="graph intake",
              initial_state={"dag_schema": {}}, actions=["noop"])
    w.perceptors = [DAGPerceptor(mode="schema", node_type=int, bounds=(0, 1))]
    spec = to_spec(w)
    assert not validate_spec(spec)
    back = from_spec(spec, allow_code=True)
    rp = back.perceptors[0]
    assert isinstance(rp, DAGPerceptor)
    assert rp.mode == "schema" and rp.node_type is int and rp.bounds == (0, 1)
