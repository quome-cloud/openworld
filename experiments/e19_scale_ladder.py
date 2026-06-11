"""E19 - Scale-ladder OOD and a stronger learned baseline (round-2 item R4).

Probes every engine at 1x / 10x / 100x state magnitudes:
- the synthesized program (the verified artifact saved by E10),
- the E12 MLP and 1-NN (retrained here, same protocol, K=10,000),
- a STRONGER learned baseline: delta-state target, 256 hidden units, longer
  training with learning-rate decay (answers 'the MLP was under-tuned'),
- the LLM next-state engine (live; the only part needing Ollama).
"""

import json
import random
from pathlib import Path

import numpy as np

from openworld import WorldState
from openworld.state import Action
from openworld.transition import CodeTransition, LLMTransition

from common import (
    GENERATOR_MODEL, RESULTS_DIR, SPRINT_DESCRIPTION, SPRINT_PROBES,
    SPRINT_PROBES_SCALED, SPRINT_RULES, require_ollama, save_results,
    sprint_ground_truth, wilson_ci,
)
from e12_learned_baseline import (
    ACTIONS, FIELDS, MLP, NearestNeighbor, collect_transitions, encode,
)

SEED = 23
K = 10000


def scale_probes(probes, factor):
    scaled = []
    for state, action in probes:
        scaled.append(({k: v * factor for k, v in state.items()}, action))
    return scaled


PROBE_LADDER = {
    "1x": SPRINT_PROBES,
    "10x": SPRINT_PROBES_SCALED,
    "100x": scale_probes(SPRINT_PROBES, 100),
}


class DeltaMLP(MLP):
    """Stronger learned baseline: predicts the state DELTA with more capacity
    and longer, lr-decayed training."""

    def __init__(self, n_in, n_out, seed=0):
        super().__init__(n_in, n_out, hidden=128, seed=seed)

    def fit_delta(self, x, states, next_states, epochs=6000):
        y = next_states - states
        lr = 2e-3
        for epoch in range(epochs):
            if epoch in (2500, 4500):
                lr *= 0.3
            self.train(x, y, epochs=1, lr=lr)

    def predict_state(self, state, action):
        x = np.array([encode(state, action)])
        delta = self.forward(x)[0]
        return {
            f: max(0, int(round(state[f] + d))) for f, d in zip(FIELDS, delta)
        }


def probe_exact_fn(predict, probes):
    hits = 0
    for state, action in probes:
        expected = sprint_ground_truth(dict(state), action.to_dict())
        try:
            if predict(dict(state), action) == expected:
                hits += 1
        except Exception:
            pass
    return hits, len(probes)


def main():
    # Engines ---------------------------------------------------------------
    e10 = json.loads((Path(RESULTS_DIR) / "e10_ood_generalization.json").read_text())
    code = CodeTransition(e10["synthesized_code"])

    rng = random.Random(SEED)
    data = collect_transitions(K, rng)
    x = np.array([encode(s, a) for s, a, _ in data])
    states = np.array([[float(s[f]) for f in FIELDS] for s, _, _ in data])
    next_states = np.array([[float(ns[f]) for f in FIELDS] for _, _, ns in data])

    mlp = MLP(x.shape[1], len(FIELDS), seed=SEED)
    mlp.train(x, next_states)
    knn = NearestNeighbor(data)
    delta_mlp = DeltaMLP(x.shape[1], len(FIELDS), seed=SEED)
    delta_mlp.fit_delta(x, states, next_states)

    llm = require_ollama(GENERATOR_MODEL, temperature=0.0)
    llm_engine = LLMTransition(llm, SPRINT_DESCRIPTION, SPRINT_RULES)

    engines = {
        "code_synthesized": lambda s, a: dict(code.step(WorldState(s), a)),
        "mlp": lambda s, a: mlp.predict_state(s, a.name),
        "delta_mlp_strong": lambda s, a: delta_mlp.predict_state(s, a.name),
        "knn1": lambda s, a: knn.predict_state(s, a.name),
        "llm_next_state": lambda s, a: dict(llm_engine.step(WorldState(s), a)),
    }

    rows = []
    for engine_name, predict in engines.items():
        for scale_name, probes in PROBE_LADDER.items():
            hits, n = probe_exact_fn(predict, probes)
            rows.append({
                "engine": engine_name, "scale": scale_name,
                "exact": hits, "n": n, "rate": hits / n,
                "ci": list(wilson_ci(hits, n)),
            })
            print(f"  {engine_name:>18} @ {scale_name:>4}: {hits}/{n}")

    save_results("e19_scale_ladder", {
        "k_transitions": K, "seed": SEED, "model": GENERATOR_MODEL,
        "rows": rows,
    })


if __name__ == "__main__":
    main()
