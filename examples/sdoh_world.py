"""SDOH risk world: a person-level, counterfactual world model on OpenWorld.

Concept (Frasch): take a person's data -> build an SDOH "world" -> use a
place-based population prior (PDFM-style neighborhood embedding -> baseline risk)
together with the person's own world model to produce an *individualized risk
profile*, then run *counterfactual* intervention experiments to recommend
behavioural / social changes.

Division of labour:
  * PDFM (Google Population Dynamics Foundation Model) supplies an EXOGENOUS,
    place-based scalar `pdfm_nbhd_risk` in [0,1] -- in production this is the
    downstream head of the ZIP/county embedding for the person's location. Here
    it enters the world as plain state (one number), so the world stays
    zero-dependency and verifiable. Swapping neighbourhoods (the `relocate`
    action) swaps this prior -- that is the population model "talking to" the
    personal model.
  * OpenWorld supplies the VERIFIED, INSPECTABLE counterfactual layer: symbolic
    SDOH state + declared interventions + code dynamics that map SDOH -> risk.
    No training data, deterministic, auditable.

Run:
    python examples/sdoh_world.py          # builds world, runs counterfactuals,
                                           # writes specs/sdoh.json
    openworld serve specs/ --allow-code    # then open /worlds/sdoh/view
"""
from __future__ import annotations

from pathlib import Path

from openworld import (Action, Agent, CodePerceptor, CodeTransition, Dial,
                       Objective, Observation, Simulation, World, spec_to_json,
                       to_spec, validate_spec)

# --- SDOH factors (0..10, higher = healthier circumstance) ----------------
FACTORS = ["food_security", "housing_stability", "transport_access",
           "income", "social_support", "physical_activity"]

# --- Verified risk dynamics ------------------------------------------------
# composite_risk in [0,1]: a population baseline (driven by the PDFM
# neighbourhood prior) reduced by the person's protective SDOH factors. Every
# line is auditable; invariants below pin risk into [0,1].
STEP = """
def transition(state, action):
    s = dict(state)
    name = action["name"]

    # interventions / behavioural changes raise one protective factor (capped 10)
    bumps = {
        "enroll_food_program": ("food_security", 3),
        "secure_housing":      ("housing_stability", 3),
        "gain_transport":      ("transport_access", 3),
        "join_support_group":  ("social_support", 3),
        "increase_activity":   ("physical_activity", 2),
    }
    if name in bumps:
        k, d = bumps[name]
        s[k] = min(10, s[k] + d)
        s["effort"] = s.get("effort", 0) + 1
    elif name == "relocate":
        # move to a lower-risk neighbourhood: swap in the alternative PDFM prior
        s["pdfm_nbhd_risk"] = s.get("pdfm_alt_nbhd_risk", s["pdfm_nbhd_risk"])
        s["effort"] = s.get("effort", 0) + 2

    # recompute individualized composite risk from the PDFM prior + SDOH factors
    factors = ["food_security", "housing_stability", "transport_access",
               "income", "social_support", "physical_activity"]
    protective = sum(s[f] for f in factors) / (10.0 * len(factors))   # 0..1
    # 70% population/place prior, modulated down by protective circumstances
    risk = 0.70 * s["pdfm_nbhd_risk"] + 0.30 * (1.0 - protective)
    risk = risk * (1.0 - 0.45 * protective)   # personal modulation of place prior
    s["composite_risk"] = round(max(0.0, min(1.0, risk)), 4)
    return s
"""

# --- Perception: a person's intake record (text) -> SDOH state ------------
PERCEIVE = """
def perceive(data):
    out = {}
    keys = ("food_security", "housing_stability", "transport_access",
            "income", "social_support", "physical_activity",
            "pdfm_nbhd_risk", "pdfm_alt_nbhd_risk")
    for line in str(data).splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        k = k.strip(); v = v.strip()
        if k in keys:
            try:
                out[k] = float(v) if '.' in v else int(v)
            except ValueError:
                pass
    return out
"""

INTERVENTIONS = ["enroll_food_program", "secure_housing", "gain_transport",
                 "join_support_group", "increase_activity", "relocate"]


def build() -> World:
    w = World(
        name="sdoh",
        description=("A person-level Social Determinants of Health world. A PDFM "
                     "neighbourhood prior sets baseline population risk; the "
                     "person's SDOH factors modulate it into an individualized "
                     "composite risk that interventions can lower."),
        initial_state={
            # SDOH factors (0..10) -- overwritten by perception of a real record
            "food_security": 3, "housing_stability": 4, "transport_access": 2,
            "income": 3, "social_support": 4, "physical_activity": 3,
            # PDFM place-based priors (downstream head of the location embedding)
            "pdfm_nbhd_risk": 0.62, "pdfm_alt_nbhd_risk": 0.34,
            # outputs
            "composite_risk": 0.0, "effort": 0,
        },
        actions=INTERVENTIONS + ["noop"],
        rules=[
            "Each intervention raises one protective SDOH factor (capped at 10) "
            "and costs effort.",
            "'relocate' swaps the PDFM neighbourhood prior for a lower-risk one.",
            "composite_risk = blend(PDFM place prior, lack of protective factors), "
            "modulated down by the person's protective circumstances; clamped [0,1].",
        ],
        transition=CodeTransition(STEP),
    )
    w.perceptors = [CodePerceptor(
        code=PERCEIVE, produces=FACTORS + ["pdfm_nbhd_risk", "pdfm_alt_nbhd_risk"],
        modality="text",
        schema={f: (int, (0, 10)) for f in FACTORS})]
    w.emit = [{"modality": "report",
               "fields": ["composite_risk", "effort"],
               "report": "individualized composite risk {composite_risk} "
                         "(effort spent: {effort})"}]
    w.objectives = [{"name": "lower risk", "goal": "min composite_risk"},
                    {"name": "limit burden", "goal": "min effort"}]
    return w


# --- Counterfactual experiments -------------------------------------------
def fixed_policy(sequence):
    """A deterministic policy that plays a fixed list of interventions, then noops."""
    plan = list(sequence)

    def policy(state, actions):
        return Action(plan.pop(0) if plan else "noop")
    return policy


def counterfactuals(world: World) -> None:
    """Compare intervention plans by resulting risk and effort (counterfactuals)."""
    # objectives: reward risk *reduction*; penalise effort, weighted by a dial we
    # can turn to trade aggressiveness vs. feasibility -- steerable at inference.
    feasibility = Dial("feasibility", value=0.02, minimum=0.0, maximum=0.2)
    objectives = [
        Objective("risk_drop",
                  fn=lambda s, a, ns: (s["composite_risk"] - ns["composite_risk"]),
                  weight=1.0),
        Objective("burden",
                  fn=lambda s, a, ns: -(ns["effort"] - s["effort"]),
                  weight=feasibility),
    ]

    plans = {
        "do nothing":            [],
        "food only":             ["enroll_food_program"],
        "relocate only":         ["relocate"],
        "behaviour bundle":      ["enroll_food_program", "increase_activity",
                                  "join_support_group"],
        "full SDOH package":     ["secure_housing", "enroll_food_program",
                                  "gain_transport", "join_support_group",
                                  "increase_activity"],
    }

    # establish the perceived baseline risk once
    world.reset()
    base = world.transition.step(world.initial_state, Action("noop"))["composite_risk"]
    print(f"\nBaseline individualized risk (from perceived record + PDFM prior): {base}\n")
    print(f"{'plan':<22}{'final risk':>12}{'risk drop':>12}{'effort':>9}{'drop/effort':>14}")
    print("-" * 69)

    rows = []
    for label, seq in plans.items():
        agent = Agent(name="careplan", policy=fixed_policy(seq))
        sim = Simulation(world=world, agents=[agent], objectives=objectives)
        traj = sim.run(steps=max(1, len(seq)))
        final_risk = traj.final_state["composite_risk"]
        effort = traj.final_state["effort"]
        drop = round(base - final_risk, 4)
        eff = round(drop / effort, 4) if effort else 0.0
        rows.append((label, final_risk, drop, effort, eff))
        print(f"{label:<22}{final_risk:>12.4f}{drop:>12.4f}{effort:>9}{eff:>14.4f}")

    best = max(rows, key=lambda r: r[2])
    best_eff = max((r for r in rows if r[3] > 0), key=lambda r: r[4])
    print("\nRecommendation:")
    print(f"  - Largest absolute risk reduction : '{best[0]}' "
          f"(-{best[2]} -> risk {best[1]})")
    print(f"  - Best reduction per unit effort  : '{best_eff[0]}' "
          f"({best_eff[4]} drop/effort)")


if __name__ == "__main__":
    world = build()

    # 1) ingest a person's record (perception -> SDOH state)
    record = """
    food_security: 2
    housing_stability: 3
    transport_access: 1
    income: 2
    social_support: 3
    physical_activity: 2
    pdfm_nbhd_risk: 0.68
    pdfm_alt_nbhd_risk: 0.30
    """
    world.reset()
    world.observe(Observation(modality="text", data=record), world.perceptors)
    world.initial_state = world.state.copy()   # treat perceived state as the start
    print("Perceived SDOH state:")
    for k in FACTORS + ["pdfm_nbhd_risk", "pdfm_alt_nbhd_risk"]:
        print(f"  {k:<20} {world.initial_state[k]}")

    # 2) run counterfactual intervention experiments
    counterfactuals(world)

    # 3) publish the portable, verifiable spec
    spec = to_spec(build(), card={"tags": ["health", "sdoh", "counterfactual", "leaf"],
                                  "license": "MIT", "version": "0.1",
                                  "lineage": "examples/sdoh_world.py"})
    problems = validate_spec(spec)
    assert not problems, problems
    Path("specs").mkdir(exist_ok=True)
    Path("specs/sdoh.json").write_text(spec_to_json(spec), encoding="utf-8")
    print("\nwrote specs/sdoh.json (serve with: openworld serve specs/ --allow-code)")
