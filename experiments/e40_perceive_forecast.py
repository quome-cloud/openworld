"""E40 - Perceive-then-forecast: multimodal inputs to a verified world model
are practically useful for real-world multi-step prediction.

Phenomenon: a discrete SIR epidemic (susceptible / infected / recovered), the
canonical real-world dynamical system where the decision that matters is a
*forecast* - "how many infected H days out?" - not a one-step read. Its
dynamics are exactly declarable, so they are a natural fit for verified
symbolic code.

Pipeline under test: messy field inputs at day 0 (a text situation report and a
ward-photo frame) are PERCEIVED into the symbolic initial state (S, I, R) at
the gated boundary, then a verified world model rolls the epidemic forward H
days to forecast the infected count. We compare against the obvious ML
alternative - learn the day-0 -> day-H mapping end to end.

Two results:
  (A) Forecast skill, perfect perception, in-distribution and at 10x
      population scale: the perceived world model is exact everywhere; the
      end-to-end learner is decent in-distribution and collapses out of it.
  (B) Graceful degradation: with imperfect perception, the world model's
      end-to-end forecast accuracy tracks perception accuracy exactly (the
      dynamics add zero error - the E39 decomposition, now on a real task).

Deterministic and offline (numpy only). The multimodal perceptors here are
run deterministically for reproducible headline numbers; the SAME perceptors
run live (TextPerceptor / VisionPerceptor) when given a real model.
"""

import numpy as np

from openworld import Action, MockPerceptor, Observation, World
from openworld.transition import FunctionTransition

from common import save_results

BETA, GAMMA = 0.35, 0.10
HORIZON = 14
N_TRAIN, N_TEST = 1500, 300
SEED = 40


# --- the real-world phenomenon as verified symbolic dynamics ---------------
def sir_step(state, action):
    s = dict(state)
    if action["name"] != "day":
        return s
    pop = s["S"] + s["I"] + s["R"]
    if pop <= 0:
        return s
    new_inf = min(s["S"], int(BETA * s["S"] * s["I"] / pop))
    new_rec = int(GAMMA * s["I"])
    s["S"] -= new_inf
    s["I"] += new_inf - new_rec
    s["R"] += new_rec
    return s


def make_world(S, I, R):
    return World(name="sir", description="discrete SIR epidemic",
                 initial_state={"S": S, "I": I, "R": R}, actions=["day"],
                 transition=FunctionTransition(sir_step))


def oracle_forecast(S, I, R, horizon=HORIZON):
    state = {"S": S, "I": I, "R": R}
    for _ in range(horizon):
        state = sir_step(state, {"name": "day"})
    return state["I"]


def world_forecast(S, I, R, horizon=HORIZON):
    """Roll the *verified world model* forward from the perceived state."""
    world = make_world(S, I, R)
    for _ in range(horizon):
        world.step(Action("day"))
    return world.state["I"]


# --- the perception front end (deterministic here; live with a real model) --
def perceive_initial(S, I, R, corrupt=False):
    """Perceive day-0 (S,I,R) from a text sitrep + a ward-photo frame through
    the gated boundary. Deterministic stand-in for TextPerceptor/VisionPerceptor."""
    if corrupt:
        I = max(0, I + (7 if I % 2 == 0 else -5))   # a wrong but plausible read
    sitrep = MockPerceptor(produces=["S", "R"], deltas=[{"S": S, "R": R}],
                           schema={"S": (int, (0, 10**7)), "R": (int, (0, 10**7))})
    frame = MockPerceptor(produces=["I"], deltas=[{"I": I}],
                          schema={"I": (int, (0, 10**7))}, modality="video_frame")
    world = make_world(0, 0, 0)
    world.observe([Observation("text", f"sitrep: S={S}, R={R}"),
                   Observation("video_frame", b"ward")], [sitrep, frame])
    return world.state["S"], world.state["I"], world.state["R"]


# --- the end-to-end learned alternative (tiny numpy MLP) --------------------
class MLP:
    def __init__(self, n_in, hidden=64, seed=0):
        rng = np.random.RandomState(seed)
        self.w1 = rng.randn(n_in, hidden) * 0.1; self.b1 = np.zeros(hidden)
        self.w2 = rng.randn(hidden, 1) * 0.1; self.b2 = np.zeros(1)

    def forward(self, x):
        self.h = np.maximum(0, x @ self.w1 + self.b1)
        return (self.h @ self.w2 + self.b2)[:, 0]

    def train(self, x, y, epochs=3000, lr=1e-3):
        for _ in range(epochs):
            p = self.forward(x); g = (2 * (p - y) / len(x))[:, None]
            gw2 = self.h.T @ g; gh = (g @ self.w2.T) * (self.h > 0)
            self.w2 -= lr * gw2; self.b2 -= lr * g.sum(0)
            self.w1 -= lr * (x.T @ gh); self.b1 -= lr * gh.sum(0)


def sample_states(rng, n, pop_lo, pop_hi):
    out = []
    for _ in range(n):
        pop = rng.randint(pop_lo, pop_hi)
        I = rng.randint(1, max(2, pop // 10))
        R = rng.randint(0, max(1, pop // 10))
        S = max(0, pop - I - R)
        out.append((S, I, R))
    return out


def main():
    rng = np.random.RandomState(SEED)
    # train the end-to-end learner on small-population trajectories
    train = sample_states(rng, N_TRAIN, 50, 200)
    Xtr = np.array(train, dtype=float)
    ytr = np.array([oracle_forecast(*s) for s in train], dtype=float)
    mlp = MLP(3, seed=0); mlp.train(Xtr, ytr)

    def evaluate(states, tol=0):
        """Exact (tol=0) day-H forecast accuracy for both approaches."""
        ours = mlp_hits = 0
        for S, I, R in states:
            truth = oracle_forecast(S, I, R)
            ps, pi, pr = perceive_initial(S, I, R)            # perfect perception
            if world_forecast(ps, pi, pr) == truth:
                ours += 1
            pred = int(round(mlp.forward(np.array([[S, I, R]], float))[0]))
            if abs(pred - truth) <= tol:
                mlp_hits += 1
        return ours / len(states), mlp_hits / len(states)

    in_dist = sample_states(rng, N_TEST, 50, 200)
    ood_10x = sample_states(rng, N_TEST, 500, 2000)
    ours_in, mlp_in = evaluate(in_dist)
    ours_ood, mlp_ood = evaluate(ood_10x)
    # give the learner every benefit: also score it with a +-2 tolerance
    _, mlp_in_tol = evaluate(in_dist, tol=2)
    _, mlp_ood_tol = evaluate(ood_10x, tol=2)

    # (B) graceful degradation: forecast accuracy vs perception error rate
    degr = []
    for rate in [0.0, 0.1, 0.25, 0.5]:
        k = int(rate * len(in_dist))
        corrupt_idx = set(range(0, len(in_dist), max(1, len(in_dist) // k))) if k else set()
        hits = perc_right = 0
        for j, (S, I, R) in enumerate(in_dist):
            truth = oracle_forecast(S, I, R)
            ps, pi, pr = perceive_initial(S, I, R, corrupt=(j in corrupt_idx))
            perc_right += (ps, pi, pr) == (S, I, R)
            hits += world_forecast(ps, pi, pr) == truth
        degr.append({"perception_error_rate": rate,
                     "perception_accuracy": perc_right / len(in_dist),
                     "forecast_accuracy": hits / len(in_dist)})

    summary = {
        "horizon": HORIZON, "n_test": N_TEST,
        "forecast_exact": {
            "world_model_in_dist": ours_in, "world_model_ood_10x": ours_ood,
            "mlp_in_dist": mlp_in, "mlp_ood_10x": mlp_ood,
            "mlp_in_dist_tol2": mlp_in_tol, "mlp_ood_10x_tol2": mlp_ood_tol,
        },
        "degradation": degr,
    }
    save_results("e40_perceive_forecast", summary)
    print(f"Day-{HORIZON} infected-count forecast (exact match vs oracle):")
    print(f"  perceived world model : in-dist {ours_in:.2f}  10x OOD {ours_ood:.2f}")
    print(f"  end-to-end MLP        : in-dist {mlp_in:.2f}  10x OOD {mlp_ood:.2f}"
          f"   (±2 tol: {mlp_in_tol:.2f} / {mlp_ood_tol:.2f})")
    print("\nGraceful degradation (perception error -> forecast accuracy):")
    for d in degr:
        print(f"  perc_err {d['perception_error_rate']:.2f}: "
              f"perception {d['perception_accuracy']:.2f} -> forecast {d['forecast_accuracy']:.2f}")
    assert ours_in == 1.0 and ours_ood == 1.0, "verified world model must be exact"


if __name__ == "__main__":
    main()
