"""E39 - Perception fidelity and the error decomposition.

The multimodal design (docs/.../multimodal-perception-design.md) makes one
falsifiable claim: because the symbolic dynamics are exact, the END-TO-END
error of a perception-fed world equals the PERCEPTION error alone -- the
dynamics layer contributes zero. E39 demonstrates this deterministically and
offline, with no LLM.

Setup: a tiny symbolic world (a sensor that raises an alarm when a reading
crosses a threshold; dynamics are exact code). A perceptor reads the symbolic
field from input but is wrong a controlled fraction of the time. We sweep that
fraction and measure three things:

  perception_accuracy        - fraction of observations perceived correctly
  dynamics_exact_given_right - among correctly-perceived instances, fraction
                               where observe()+step() matches the oracle (must
                               be 1.0: the dynamics are exact)
  end_to_end_accuracy        - fraction where the full pipeline matches the
                               oracle

The claim holds iff end_to_end_accuracy == perception_accuracy at every error
rate and dynamics_exact_given_right == 1.0 throughout.
"""

from openworld import Action, Observation, PerceptionGate, Perceptor, World
from openworld.transition import FunctionTransition

from common import save_results

THRESHOLD = 50
N = 200
ERROR_RATES = [0.0, 0.1, 0.25, 0.5, 0.9]


def oracle(state, action):
    s = dict(state)
    if action["name"] == "evaluate":
        s["alarm"] = s["reading"] > THRESHOLD
    return s


def make_world(reading):
    return World(name="sensor", description="threshold alarm",
                 initial_state={"reading": reading, "alarm": False},
                 actions=["evaluate"], transition=FunctionTransition(oracle))


class NoisyPerceptor(Perceptor):
    """Reads the true 'reading' from the observation, but corrupts a fixed,
    deterministic set of instance indices to an in-range wrong value."""

    modality = "text"
    produces = ["reading"]
    schema = {"reading": (int, (0, 100))}

    def __init__(self, corrupt_indices):
        self.corrupt = set(corrupt_indices)

    def perceive(self, observation):
        idx, truth = observation.data            # (index, true reading)
        if idx in self.corrupt:
            wrong = (truth + 37) % 101           # a different, in-range value
            return {"reading": wrong}
        return {"reading": truth}


def corrupt_indices(n, rate):
    """Deterministically pick floor(rate*n) evenly-spaced indices to corrupt."""
    k = int(rate * n)
    if k == 0:
        return set()
    step = n / k
    return {int(i * step) for i in range(k)}


def run_rate(rate):
    truths = [(i, (i * 7 + 3) % 101) for i in range(N)]   # deterministic readings
    perceptor = NoisyPerceptor(corrupt_indices(N, rate))
    gate = PerceptionGate()
    perceived_right = 0
    end_to_end_right = 0
    right_and_exact = 0
    for idx, truth in truths:
        # ground-truth final state (what a perfect pipeline must produce)
        truth_world = make_world(truth)
        truth_final = truth_world.step(Action("evaluate"))

        # perceived pipeline: blank world -> observe -> step
        world = make_world(0)
        world.observe(Observation("text", (idx, truth)), perceptor)
        perceived = world.state["reading"]
        final = world.step(Action("evaluate"))

        is_right = perceived == truth
        perceived_right += is_right
        matches = final == truth_final
        end_to_end_right += matches
        if is_right:
            right_and_exact += matches
    n_right = perceived_right
    return {
        "error_rate": rate,
        "perception_accuracy": perceived_right / N,
        "end_to_end_accuracy": end_to_end_right / N,
        "dynamics_exact_given_right": (right_and_exact / n_right) if n_right else None,
    }


def main():
    rows = [run_rate(r) for r in ERROR_RATES]
    decomposition_holds = all(
        abs(r["perception_accuracy"] - r["end_to_end_accuracy"]) < 1e-9
        and (r["dynamics_exact_given_right"] in (1.0, None))
        for r in rows
    )
    save_results("e39_perception_fidelity", {
        "n": N, "threshold": THRESHOLD, "error_rates": ERROR_RATES,
        "rows": rows,
        "decomposition_holds": decomposition_holds,
        "claim": "end_to_end_accuracy == perception_accuracy; dynamics add zero error",
    })
    print(f"{'err_rate':>9} {'perception':>11} {'end_to_end':>11} {'dyn|right':>10}")
    for r in rows:
        dg = "n/a" if r["dynamics_exact_given_right"] is None else f"{r['dynamics_exact_given_right']:.2f}"
        print(f"{r['error_rate']:>9.2f} {r['perception_accuracy']:>11.3f} "
              f"{r['end_to_end_accuracy']:>11.3f} {dg:>10}")
    print(f"\nerror decomposition holds (end-to-end == perception, dynamics exact): "
          f"{decomposition_holds}")
    assert decomposition_holds, "error decomposition violated -- dynamics added error"


if __name__ == "__main__":
    main()
