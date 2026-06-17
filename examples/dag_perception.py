"""Input a causal DAG through the perception middleware (the `graph` modality).

`DAGPerceptor` ingests a dagitty / Graphviz `.dot` DAG and resolves it three ways
(selected by `mode`):

  * mode="graph"  -> a normalized graph delta (nodes / edges / exposures / outcomes)
  * mode="schema" -> the observed variables to extract downstream (DAG as schema)
  * mode="world"  -> the DAG compiled to a verified structural-causal-model world,
                     where do_<node> actions are Pearl's do-operator

Run:  python examples/dag_perception.py
"""
from openworld import (Action, DAGPerceptor, Observation, World, dag_to_world,
                       parse_dag)

# A small perinatal SDOH DAG in dagitty syntax (Income -> Stress -> Preterm -> Bayley).
DAG = """
dag {
"Income low" [exposure]
"Prenatal Stress" [exposure]
"Preterm" []
"Bayley low" [outcome]
"Income low" -> "Prenatal Stress"
"Prenatal Stress" -> "Preterm"
"Preterm" -> "Bayley low"
}
"""

if __name__ == "__main__":
    obs = Observation("graph", DAG)

    # (1) graph mode: perceive the DAG as a normalized graph object
    g = DAGPerceptor(mode="graph").perceive(obs)
    print("graph mode:")
    print(f"  exposures: {g['dag_exposures']}")
    print(f"  outcome:   {g['dag_outcomes']}")
    print(f"  edges:     {g['dag_edges']}")

    # (2) schema mode: the DAG declares which variables to perceive
    s = DAGPerceptor(mode="schema").perceive(obs)
    print("\nschema mode (observed variables to extract):")
    for k, v in s["dag_schema"].items():
        print(f"  {k}: {v}")

    # (3) world mode: compile to a verified SCM world and run the do-operator
    w = DAGPerceptor(mode="world").to_world(obs)
    print("\nworld mode (compiled SCM):")
    print(f"  actions: {w.actions}")
    out = w.transition.step(w.initial_state, Action("do_income_low", params={"value": 1.0}))
    fired = {k: v for k, v in out.items() if not k.startswith("u_")}
    print(f"  do(income_low := 1) propagates -> {fired}")

    # The perceptor also drops a DAG straight into a world's state via observe():
    intake = World(name="dag_intake", description="perceive a causal graph",
                   initial_state={"dag_nodes": [], "dag_edges": [],
                                  "dag_exposures": [], "dag_outcomes": []},
                   actions=["noop"])
    intake.observe(obs, DAGPerceptor(mode="graph"))
    print(f"\nobserve() committed outcome node to state: {intake.state['dag_outcomes']}")
