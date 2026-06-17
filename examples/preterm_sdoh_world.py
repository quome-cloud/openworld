"""Preterm-birth SDOH world: a mechanism-faithful, counterfactual world model.

Grounded in two references the user supplied:

  * Marmot (2018), *Health equity, cancer, and social determinants of health* --
    the social gradient (lower social position -> higher risk) and six SDOH
    domains; behaviour change alone is insufficient, you must act on the
    structural determinants.
  * Shapiro, Fraser, Frasch & Seguin (2013), *Psychosocial stress in pregnancy
    and preterm birth: associations and mechanisms* -- psychosocial stress (PSP)
    raises preterm-birth (PTB) risk via behavioural, infectious, neuro-
    inflammatory and neuroendocrine pathways; subjective stress perception and
    pregnancy-related anxiety are the strongest predictors; social support is
    protective; effects are strongest in early pregnancy (risk ratios 1.2-2.1).

The world encodes the explicit causal chain as layered, verified code:

    PDFM neighbourhood prior  +  structural SDOH  -->  perceived stress / anxiety
                                                  -->  neuroendocrine (CRH/cortisol)
                                                     + inflammatory load
                                                  -->  individualized PTB risk

Counterfactual experiments can intervene at the STRUCTURAL layer (Marmot:
income, housing, neighbourhood) or the PROXIMAL layer (Shapiro: social support,
stress/anxiety reduction) and compare PTB-risk trajectories -> recommendations.

Run:
    python examples/preterm_sdoh_world.py     # counterfactuals + specs/preterm_sdoh.json
    openworld serve specs/ --allow-code       # then open /worlds/preterm_sdoh/view
"""
from __future__ import annotations

from pathlib import Path

from openworld import (Action, Agent, CodePerceptor, CodeTransition, Dial,
                       Objective, Observation, Simulation, World, spec_to_json,
                       to_spec, validate_spec)

# structural SDOH factors (0..10, higher = more protective) -- Marmot domains
STRUCTURAL = ["income", "housing_stability", "food_security",
              "social_support", "neighborhood_safety"]
# proximal psychosocial state (0..10, higher = worse) -- Shapiro PSP dimensions
PROXIMAL = ["perceived_stress", "pregnancy_anxiety"]

STEP = """
def transition(state, action):
    s = dict(state)
    name = action["name"]

    # --- interventions -----------------------------------------------------
    # structural (Marmot): act on the social determinants
    structural = {
        "income_support":   ("income", 3),
        "stable_housing":   ("housing_stability", 3),
        "food_program":     ("food_security", 3),
        "relocate":         ("neighborhood_safety", 4),   # also swaps PDFM prior
    }
    # proximal (Shapiro): act on the mediating stress pathway
    proximal = {
        "peer_support":     ("social_support", 3),        # protective structural+social
        "stress_program":   ("perceived_stress", -3),     # CBT / mindfulness
        "anxiety_care":     ("pregnancy_anxiety", -3),    # targeted prenatal anxiety care
    }
    if name in structural:
        k, d = structural[name]
        s[k] = max(0, min(10, s[k] + d))
        s["effort"] = s.get("effort", 0) + 1
        if name == "relocate":
            s["pdfm_nbhd_risk"] = s.get("pdfm_alt_nbhd_risk", s["pdfm_nbhd_risk"])
            s["effort"] = s.get("effort", 0) + 1   # relocating is costly
    elif name in proximal:
        k, d = proximal[name]
        s[k] = max(0, min(10, s[k] + d))
        s["effort"] = s.get("effort", 0) + 1

    # --- causal chain (recomputed every step; all auditable) ---------------
    # 1) structural disadvantage -> perceived stress & anxiety (the social
    #    gradient). Low protective SDOH pushes the proximal layer UP, unless an
    #    intervention has directly lowered it this step.
    protective = (s["income"] + s["housing_stability"] + s["food_security"]
                  + s["social_support"] + s["neighborhood_safety"]) / 50.0  # 0..1
    place_stress = 0.5 * s["pdfm_nbhd_risk"] * 10.0      # neighbourhood pushes stress up
    structural_stress = (1.0 - protective) * 10.0
    # social support buffers perceived stress (Shapiro: protective)
    buffer = 0.25 * s["social_support"]
    # blend the person's standing stress with structural drivers
    s["perceived_stress"] = round(max(0.0, min(10.0,
        0.5 * s["perceived_stress"] + 0.5 * (0.5 * structural_stress
                                             + 0.5 * place_stress) - buffer)), 3)
    s["pregnancy_anxiety"] = round(max(0.0, min(10.0,
        0.6 * s["pregnancy_anxiety"] + 0.4 * s["perceived_stress"] - 0.5 * buffer)), 3)

    # 2) PSP -> physiological load: neuroendocrine (CRH/cortisol) + inflammation
    #    (Shapiro mechanisms). Subjective stress & anxiety are the strongest
    #    drivers, so they dominate the load term.
    psp = (0.55 * s["pregnancy_anxiety"] + 0.45 * s["perceived_stress"]) / 10.0  # 0..1
    s["neuroendocrine_load"] = round(psp, 3)            # CRH / cortisol proxy
    s["inflammatory_load"] = round(0.8 * psp + 0.2 * s["pdfm_nbhd_risk"], 3)

    # 3) load + early-pregnancy sensitivity -> PTB risk. Risk ratios 1.2-2.1:
    #    map onto a baseline hazard modulated by the physiological load, with a
    #    higher weight in early gestation (Shapiro: effects strongest early).
    base_hazard = 0.08                                   # ~8% population baseline
    early = s.get("trimester", 1)
    sensitivity = {1: 1.6, 2: 1.2, 3: 1.0}.get(early, 1.2)
    load = 0.5 * s["neuroendocrine_load"] + 0.5 * s["inflammatory_load"]
    rr = 1.0 + 1.1 * load * sensitivity                  # risk ratio in ~[1.0, 2.1]
    s["ptb_risk"] = round(max(0.0, min(1.0, base_hazard * rr)), 4)
    return s
"""

PERCEIVE = """
def perceive(data):
    out = {}
    keys = ("income", "housing_stability", "food_security", "social_support",
            "neighborhood_safety", "perceived_stress", "pregnancy_anxiety",
            "pdfm_nbhd_risk", "pdfm_alt_nbhd_risk", "trimester")
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

INTERVENTIONS = ["income_support", "stable_housing", "food_program", "relocate",
                 "peer_support", "stress_program", "anxiety_care"]


def build() -> World:
    w = World(
        name="preterm_sdoh",
        description=("A perinatal SDOH world: structural social determinants and a "
                     "PDFM neighbourhood prior drive psychosocial stress and "
                     "pregnancy anxiety, which raise neuroendocrine and "
                     "inflammatory load and, in turn, individualized preterm-birth "
                     "risk. Encodes Shapiro et al. (2013) mechanisms over a "
                     "Marmot (2018) social gradient."),
        initial_state={
            "income": 3, "housing_stability": 4, "food_security": 4,
            "social_support": 3, "neighborhood_safety": 3,
            "perceived_stress": 6, "pregnancy_anxiety": 6,
            "pdfm_nbhd_risk": 0.62, "pdfm_alt_nbhd_risk": 0.30,
            "trimester": 1,
            # derived outputs
            "neuroendocrine_load": 0.0, "inflammatory_load": 0.0,
            "ptb_risk": 0.0, "effort": 0,
        },
        actions=INTERVENTIONS + ["noop"],
        rules=[
            "Structural interventions (income_support, stable_housing, "
            "food_program, relocate) raise protective SDOH; relocate also swaps "
            "the PDFM neighbourhood prior.",
            "Proximal interventions (peer_support, stress_program, anxiety_care) "
            "act on the mediating psychosocial stress pathway.",
            "Causal chain: SDOH + PDFM prior -> perceived_stress & "
            "pregnancy_anxiety -> neuroendocrine + inflammatory load -> ptb_risk; "
            "early gestation is more sensitive (Shapiro 2013).",
        ],
        transition=CodeTransition(STEP),
    )
    w.perceptors = [CodePerceptor(
        code=PERCEIVE,
        produces=STRUCTURAL + PROXIMAL + ["pdfm_nbhd_risk", "pdfm_alt_nbhd_risk",
                                          "trimester"],
        modality="text",
        schema={f: (int, (0, 10)) for f in STRUCTURAL + PROXIMAL})]
    w.emit = [{"modality": "report",
               "fields": ["ptb_risk", "perceived_stress", "pregnancy_anxiety",
                          "effort"],
               "report": "preterm-birth risk {ptb_risk} | stress "
                         "{perceived_stress} anxiety {pregnancy_anxiety} "
                         "(effort {effort})"}]
    w.objectives = [{"name": "lower PTB risk", "goal": "min ptb_risk"},
                    {"name": "limit burden", "goal": "min effort"}]
    return w


def fixed_policy(sequence):
    plan = list(sequence)

    def policy(state, actions):
        return Action(plan.pop(0) if plan else "noop")
    return policy


def counterfactuals(world: World) -> None:
    feasibility = Dial("feasibility", value=0.01, minimum=0.0, maximum=0.2)
    objectives = [
        Objective("risk_drop",
                  fn=lambda s, a, ns: (s["ptb_risk"] - ns["ptb_risk"]),
                  weight=1.0),
        Objective("burden",
                  fn=lambda s, a, ns: -(ns["effort"] - s["effort"]),
                  weight=feasibility),
    ]
    plans = {
        "do nothing":          [],
        "structural (Marmot)": ["income_support", "stable_housing", "relocate"],
        "proximal (Shapiro)":  ["peer_support", "stress_program", "anxiety_care"],
        "combined":            ["peer_support", "income_support", "stress_program",
                                "anxiety_care", "relocate"],
    }
    world.reset()
    # let the chain settle one step to read the perceived baseline risk
    base = world.transition.step(world.initial_state, Action("noop"))["ptb_risk"]
    print(f"\nBaseline individualized preterm-birth risk: {base}\n")
    print(f"{'care plan':<22}{'final PTB risk':>16}{'risk drop':>12}{'effort':>9}")
    print("-" * 59)
    rows = []
    for label, seq in plans.items():
        sim = Simulation(world=world,
                         agents=[Agent(name="careplan", policy=fixed_policy(seq))],
                         objectives=objectives)
        traj = sim.run(steps=max(1, len(seq)))
        final = traj.final_state["ptb_risk"]
        effort = traj.final_state["effort"]
        drop = round(base - final, 4)
        rows.append((label, final, drop, effort))
        print(f"{label:<22}{final:>16.4f}{drop:>12.4f}{effort:>9}")
    best = max(rows, key=lambda r: r[2])
    print(f"\nRecommendation: '{best[0]}' yields the largest PTB-risk reduction "
          f"(-{best[2]} -> {best[1]}).")
    print("Note (Marmot): structural action typically dominates behaviour-only "
          "plans; (Shapiro): proximal stress/anxiety care matters most in early "
          "gestation.")


if __name__ == "__main__":
    world = build()
    record = """
    income: 2
    housing_stability: 3
    food_security: 3
    social_support: 2
    neighborhood_safety: 2
    perceived_stress: 7
    pregnancy_anxiety: 7
    pdfm_nbhd_risk: 0.66
    pdfm_alt_nbhd_risk: 0.28
    trimester: 1
    """
    world.reset()
    world.observe(Observation(modality="text", data=record), world.perceptors)
    world.initial_state = world.state.copy()
    print("Perceived perinatal SDOH state:")
    for k in STRUCTURAL + PROXIMAL + ["pdfm_nbhd_risk", "trimester"]:
        print(f"  {k:<20} {world.initial_state[k]}")

    counterfactuals(world)

    spec = to_spec(build(), card={"tags": ["health", "sdoh", "perinatal",
                                           "preterm-birth", "counterfactual", "leaf"],
                                  "license": "MIT", "version": "0.1",
                                  "lineage": "examples/preterm_sdoh_world.py"})
    problems = validate_spec(spec)
    assert not problems, problems
    Path("specs").mkdir(exist_ok=True)
    Path("specs/preterm_sdoh.json").write_text(spec_to_json(spec), encoding="utf-8")
    print("\nwrote specs/preterm_sdoh.json")
