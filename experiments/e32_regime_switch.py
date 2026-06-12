"""E32 - Rule changes over time (regime switch at step 10).

A market whose policy changes at step 10: in phase A 'invest' turns 2 capital
into 3 output and 'harvest' converts output to capital 1:1; from step 10 on,
'invest' yields only 1 output (1 unit burned as tax) and 'harvest' is taxed
50% (floor). A hand-written oracle implements the switch exactly (the step
counter lives in state and increments on every action).

Conditions:
  (a) PHASED      - synthesize phase-A dynamics from phase-A text alone and
                    phase-B from phase-B text alone (qwen2.5:7b, 3 replicates
                    each), assemble PhasedTransition([(0, A), (10, B)]).
  (b) MONOLITHIC  - synthesize one transition from the combined
                    rules-with-change text (3 replicates).
  (c) LLM proxy   - LLMTransition with the combined rules (1 replicate).

Rollout: 20 steps of a fixed action script crossing the boundary, closed
loop. Metric: per-step exact match vs the oracle on (step, capital, output),
split into pre-boundary (steps 1-10), the boundary step (11), and
post-boundary (11-20, with 12-20 also reported).
"""

from openworld import OllamaLLM, World, WorldState
from openworld.state import Action
from openworld.transition import LLMTransition, PhasedTransition
from openworld.verify import SynthesisError

from common import GENERATOR_MODEL, Timer, require_ollama, save_results

REPLICATES = 3
MAX_ITERS = 4
SWITCH_STEP = 10
ROLLOUT_STEPS = 20
KEYS = ("step", "capital", "output")
INITIAL = {"step": 0, "capital": 10, "output": 0}
PHASE_B_SAMPLE = {"step": 10, "capital": 8, "output": 3}

ACTION_SCRIPT = [
    "invest", "invest", "harvest", "invest", "wait",
    "invest", "harvest", "invest", "invest", "harvest",
    "invest", "invest", "harvest", "invest", "wait",
    "invest", "harvest", "invest", "invest", "harvest",
]

PHASE_A_RULES = [
    "'invest' (only when capital >= 2): capital decreases by 2, output "
    "increases by 3. With less than 2 capital, 'invest' has no effect "
    "(besides the step counter).",
    "'harvest': capital increases by the current output (1:1 conversion), "
    "then output resets to 0.",
    "'wait' changes nothing (besides the step counter).",
    "After EVERY action (including 'wait' and 'noop'), step increases by 1.",
]
PHASE_B_RULES = [
    "'invest' (only when capital >= 2): capital decreases by 2, output "
    "increases by 1 (the other unit of capital is burned as tax). With less "
    "than 2 capital, 'invest' has no effect (besides the step counter).",
    "'harvest': capital increases by output // 2 (integer floor division - "
    "a 50% tax), then output resets to 0.",
    "'wait' changes nothing (besides the step counter).",
    "After EVERY action (including 'wait' and 'noop'), step increases by 1.",
]
COMBINED_RULES = (
    ["The market's policy CHANGES when the step counter reaches "
     f"{SWITCH_STEP}. All conditions read the PRE-action value of 'step'.",
     f"While step < {SWITCH_STEP} (the pre-action value), these rules apply:"]
    + ["  " + r for r in PHASE_A_RULES[:3]]
    + [f"From step >= {SWITCH_STEP} (the pre-action value) on, these rules "
       "apply instead:"]
    + ["  " + r for r in PHASE_B_RULES[:3]]
    + ["In BOTH regimes, after EVERY action (including 'wait' and 'noop'), "
       "step increases by 1."]
)
DESCRIPTION_A = ("A simple market economy: capital is invested to create "
                 "output, output is harvested back into capital.")
DESCRIPTION_B = ("A taxed market economy: investment is taxed in kind and "
                 "harvests are taxed 50%.")
DESCRIPTION_COMBINED = ("A market economy whose tax policy changes once the "
                        "step counter reaches a threshold.")


def oracle_step(state, action_name):
    s = dict(state)
    phase_b = s["step"] >= SWITCH_STEP
    if action_name == "invest" and s["capital"] >= 2:
        s["capital"] -= 2
        s["output"] += 1 if phase_b else 3
    elif action_name == "harvest":
        s["capital"] += s["output"] // 2 if phase_b else s["output"]
        s["output"] = 0
    s["step"] += 1
    return s


def oracle_trajectory():
    state = dict(INITIAL)
    states = []
    for name in ACTION_SCRIPT:
        state = oracle_step(state, name)
        states.append(dict(state))
    return states


def project(state):
    return {k: state.get(k) for k in KEYS}


def rollout_matches(transition, oracle_states):
    """Closed-loop 20-step rollout; per-step exact match on KEYS."""
    state = WorldState(dict(INITIAL))
    matches, trajectory = [], []
    for name, expected in zip(ACTION_SCRIPT, oracle_states):
        try:
            state = transition.step(state, Action(name))
        except Exception:
            matches.extend([False] * (len(oracle_states) - len(matches)))
            break
        snap = project(dict(state))
        trajectory.append(snap)
        matches.append(snap == expected)
    return matches, trajectory


def split_metrics(matches):
    pre = matches[:SWITCH_STEP]
    return {
        "per_step_match": matches,
        "pre_boundary_accuracy": sum(pre) / len(pre),
        "boundary_step_match": bool(matches[SWITCH_STEP]),
        "post_boundary_accuracy": sum(matches[SWITCH_STEP:]) / len(matches[SWITCH_STEP:]),
        "post_after_boundary_accuracy": (
            sum(matches[SWITCH_STEP + 1:]) / len(matches[SWITCH_STEP + 1:])),
        "exact_full_rollout": all(matches),
    }


def nonneg():
    return [("capital and output never negative",
             lambda s: s["capital"] >= 0 and s["output"] >= 0)]


def synth_world(name, description, rules, initial, seed):
    llm = OllamaLLM(model=GENERATOR_MODEL, temperature=0.7,
                    options={"seed": seed})
    world = World(name=name, description=description,
                  initial_state=dict(initial),
                  actions=["invest", "harvest", "wait"], rules=list(rules),
                  llm=llm)
    return world.compile(invariants=nonneg(), max_iters=MAX_ITERS)


def main():
    llm_det = require_ollama(GENERATOR_MODEL, temperature=0.0)
    oracle_states = oracle_trajectory()
    rows = []

    for replicate in range(REPLICATES):
        # (a) PHASED: each phase synthesized from its own text alone.
        record = {"condition": "phased", "replicate": replicate}
        with Timer() as t:
            try:
                phase_a = synth_world("market-phase-a", DESCRIPTION_A,
                                      PHASE_A_RULES, INITIAL,
                                      seed=15000 + replicate)
                phase_b = synth_world("market-phase-b", DESCRIPTION_B,
                                      PHASE_B_RULES, PHASE_B_SAMPLE,
                                      seed=15100 + replicate)
                record["accepted"] = True
                record["code_a"], record["code_b"] = phase_a.code, phase_b.code
                transition = PhasedTransition([(0, phase_a),
                                               (SWITCH_STEP, phase_b)])
            except SynthesisError as exc:
                record["accepted"] = False
                record["failure"] = str(exc)
                transition = None
        record["synthesis_seconds"] = round(t.elapsed, 1)
        matches = [False] * ROLLOUT_STEPS
        if transition is not None:
            matches, record["trajectory"] = rollout_matches(transition,
                                                            oracle_states)
        record.update(split_metrics(matches))
        rows.append(record)
        print(f"  phased     #{replicate}: accepted={record['accepted']} "
              f"pre={record['pre_boundary_accuracy']:.2f} "
              f"post={record['post_boundary_accuracy']:.2f}")

        # (b) MONOLITHIC: one transition from the combined text.
        record = {"condition": "monolithic", "replicate": replicate}
        with Timer() as t:
            try:
                transition = synth_world("market-combined",
                                         DESCRIPTION_COMBINED, COMBINED_RULES,
                                         INITIAL, seed=15200 + replicate)
                record["accepted"] = True
                record["code"] = transition.code
            except SynthesisError as exc:
                record["accepted"] = False
                record["failure"] = str(exc)
                transition = None
        record["synthesis_seconds"] = round(t.elapsed, 1)
        matches = [False] * ROLLOUT_STEPS
        if transition is not None:
            matches, record["trajectory"] = rollout_matches(transition,
                                                            oracle_states)
        record.update(split_metrics(matches))
        rows.append(record)
        print(f"  monolithic #{replicate}: accepted={record['accepted']} "
              f"pre={record['pre_boundary_accuracy']:.2f} "
              f"post={record['post_boundary_accuracy']:.2f}")

    # (c) LLM proxy: direct next-state prediction, 1 replicate (slow).
    record = {"condition": "llm_proxy", "replicate": 0, "accepted": True}
    with Timer() as t:
        transition = LLMTransition(llm_det, DESCRIPTION_COMBINED,
                                   COMBINED_RULES)
        matches, record["trajectory"] = rollout_matches(transition,
                                                        oracle_states)
    record["synthesis_seconds"] = 0.0
    record["rollout_seconds"] = round(t.elapsed, 1)
    record.update(split_metrics(matches))
    rows.append(record)
    print(f"  llm_proxy  #0: pre={record['pre_boundary_accuracy']:.2f} "
          f"post={record['post_boundary_accuracy']:.2f}")

    summary = []
    for condition in ("phased", "monolithic", "llm_proxy"):
        cell = [r for r in rows if r["condition"] == condition]
        summary.append({
            "condition": condition,
            "replicates": len(cell),
            "acceptance_rate": sum(r["accepted"] for r in cell) / len(cell),
            "mean_pre_boundary_accuracy": sum(
                r["pre_boundary_accuracy"] for r in cell) / len(cell),
            "boundary_step_match_rate": sum(
                r["boundary_step_match"] for r in cell) / len(cell),
            "mean_post_boundary_accuracy": sum(
                r["post_boundary_accuracy"] for r in cell) / len(cell),
            "mean_post_after_boundary_accuracy": sum(
                r["post_after_boundary_accuracy"] for r in cell) / len(cell),
            "exact_full_rollouts": sum(r["exact_full_rollout"] for r in cell),
        })

    save_results("e32_regime_switch", {
        "model": GENERATOR_MODEL,
        "switch_step": SWITCH_STEP,
        "rollout_steps": ROLLOUT_STEPS,
        "action_script": ACTION_SCRIPT,
        "initial_state": INITIAL,
        "oracle_trajectory": oracle_states,
        "phase_a_rules": PHASE_A_RULES,
        "phase_b_rules": PHASE_B_RULES,
        "combined_rules": COMBINED_RULES,
        "summary": summary,
        "rows": rows,
    })

    print(f"\n{'condition':<12} {'accept':>7} {'pre(1-10)':>10} "
          f"{'boundary(11)':>13} {'post(11-20)':>12} {'exact':>6}")
    for s in summary:
        print(f"{s['condition']:<12} {s['acceptance_rate']:>7.2f} "
              f"{s['mean_pre_boundary_accuracy']:>10.2f} "
              f"{s['boundary_step_match_rate']:>13.2f} "
              f"{s['mean_post_boundary_accuracy']:>12.2f} "
              f"{s['exact_full_rollouts']:>3}/{s['replicates']}")


if __name__ == "__main__":
    main()
