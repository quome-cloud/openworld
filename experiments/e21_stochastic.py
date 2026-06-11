"""E21 - Stochastic-world synthesis with distributional verification (Q2).

The request-queue world: serving is deterministic, but after every action a
new request arrives with probability 0.3. Replayability is preserved by
threading a seed through the state: the transition seeds random.Random with
state['rng_seed'] and writes a fresh derived seed back. The 7B generator
synthesizes this from rule text (three attempts); each accepted program is
verified DISTRIBUTIONALLY against a hand-written stochastic oracle:

- arrival frequency over 2000 seeded transitions vs the declared 0.3
- exact correctness of the deterministic fields (queue/served bookkeeping)
- bit-exact replay: same rng_seed twice -> identical next state
"""

import random as pyrandom

from openworld import OllamaLLM, WorldState
from openworld.state import Action
from openworld.verify import SynthesisError, Verifier, synthesize_transition

from common import GENERATOR_MODEL, require_ollama, save_results

ATTEMPTS = 3
N_DRAWS = 2000
ARRIVAL_P = 0.3

DESCRIPTION = (
    "A service desk. Requests wait in a queue; 'serve' completes one request. "
    "After every action a new request MAY arrive at random."
)
ACTIONS = ["serve", "wait"]
RULES = [
    "'serve': if queue > 0, queue decreases by 1 and served increases by 1.",
    "'wait' and 'noop': no deterministic effect.",
    "AFTER applying the action's deterministic effect: create "
    "rng = random.Random(state['rng_seed']); if rng.random() < 0.3, a new "
    "request arrives (queue increases by 1, arrivals increases by 1).",
    "Finally set next_state['rng_seed'] = rng.randint(0, 999999999) so "
    "rollouts stay replayable.",
]
INITIAL = {"queue": 3, "served": 0, "arrivals": 0, "rng_seed": 12345}

STOCHASTIC_SYSTEM = (
    "You write Python world-dynamics code. Reply with a single python code "
    "block defining exactly:\n"
    "    def transition(state: dict, action: dict) -> dict\n"
    "It must return the COMPLETE next state dict (copy the input, never "
    "mutate shared structures), handle every declared action including "
    "'noop', and use only pure python, the math module, and the random "
    "module EXACTLY as the rules describe (seed from state['rng_seed'], "
    "write back a new derived seed). No imports, no I/O."
)


def oracle(state, action):
    s = dict(state)
    if action["name"] == "serve" and s["queue"] > 0:
        s["queue"] -= 1
        s["served"] += 1
    rng = pyrandom.Random(s["rng_seed"])
    if rng.random() < ARRIVAL_P:
        s["queue"] += 1
        s["arrivals"] += 1
    s["rng_seed"] = rng.randint(0, 999999999)
    return s


def evaluate(transition):
    # 1. Arrival frequency across many seeds (distributional fidelity).
    arrivals = 0
    deterministic_ok = 0
    for i in range(N_DRAWS):
        state = WorldState({"queue": 3, "served": 0, "arrivals": 0,
                            "rng_seed": 50000 + i})
        out = dict(transition.step(state, Action("serve")))
        arrived = out.get("arrivals", 0) > 0
        arrivals += arrived
        expected_queue = 2 + (1 if arrived else 0)
        deterministic_ok += (out.get("served") == 1 and out.get("queue") == expected_queue)
    # 2. Bit-exact replay: identical seed -> identical output.
    a = dict(transition.step(WorldState(dict(INITIAL)), Action("serve")))
    b = dict(transition.step(WorldState(dict(INITIAL)), Action("serve")))
    # 3. Bit-exact match against the oracle (strict; may fail even when the
    #    distribution is right, if the model draws in a different order).
    oracle_match = 0
    for i in range(200):
        state = {"queue": 3, "served": 0, "arrivals": 0, "rng_seed": 90000 + i}
        ours = dict(transition.step(WorldState(dict(state)), Action("serve")))
        theirs = oracle(dict(state), Action("serve").to_dict())
        oracle_match += ours == theirs
    return {
        "arrival_rate": arrivals / N_DRAWS,
        "arrival_abs_error": abs(arrivals / N_DRAWS - ARRIVAL_P),
        "deterministic_accuracy": deterministic_ok / N_DRAWS,
        "replay_bit_exact": a == b,
        "oracle_bit_exact_rate": oracle_match / 200,
    }


def main():
    llm_check = require_ollama(GENERATOR_MODEL)
    del llm_check
    rows = []
    for attempt in range(ATTEMPTS):
        llm = OllamaLLM(model=GENERATOR_MODEL, temperature=0.7,
                        options={"seed": 12000 + attempt})
        verifier = Verifier(
            initial_state=WorldState(dict(INITIAL)),
            sample_actions=[Action(a) for a in ACTIONS],
            invariants=[("counters non-negative",
                         lambda s: s["queue"] >= 0 and s["served"] >= 0)],
        )
        record = {"attempt": attempt}
        try:
            transition = synthesize_transition(
                llm, DESCRIPTION, WorldState(dict(INITIAL)), ACTIONS, RULES,
                verifier=verifier, max_iters=4,
                generator_system=STOCHASTIC_SYSTEM,
            )
            record["accepted"] = True
            record.update(evaluate(transition))
            record["code"] = transition.code
        except SynthesisError as exc:
            record["accepted"] = False
            record["error"] = str(exc)[:200]
        rows.append(record)
        if record["accepted"]:
            print(f"  #{attempt}: arrival rate {record['arrival_rate']:.3f} "
                  f"(target {ARRIVAL_P}), det acc {record['deterministic_accuracy']:.3f}, "
                  f"replay exact {record['replay_bit_exact']}, "
                  f"oracle exact {record['oracle_bit_exact_rate']:.2f}")
        else:
            print(f"  #{attempt}: synthesis failed")

    accepted = [r for r in rows if r["accepted"]]
    save_results("e21_stochastic", {
        "model": GENERATOR_MODEL, "arrival_p": ARRIVAL_P, "n_draws": N_DRAWS,
        "attempts": ATTEMPTS,
        "summary": {
            "acceptance_rate": len(accepted) / len(rows),
            "mean_arrival_abs_error": (
                sum(r["arrival_abs_error"] for r in accepted) / len(accepted)
                if accepted else None),
            "mean_deterministic_accuracy": (
                sum(r["deterministic_accuracy"] for r in accepted) / len(accepted)
                if accepted else None),
            "replay_bit_exact_all": all(r["replay_bit_exact"] for r in accepted) if accepted else None,
            "mean_oracle_bit_exact": (
                sum(r["oracle_bit_exact_rate"] for r in accepted) / len(accepted)
                if accepted else None),
        },
        "rows": rows,
    })


if __name__ == "__main__":
    main()
