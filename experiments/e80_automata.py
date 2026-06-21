"""E80 algorithmic (state-machine): world-time compute on finite-automaton simulation.

A WORLD is a random deterministic finite automaton (a transition table); an instance is an input
string; the answer is the final state after tracing the table from the start. The table is GIVEN
in the prompt, so held-out automata are solvable by *executing the rule* -- the shared skill.
Many random automata = many distinct worlds; small LLMs are weak at multi-step state tracking
(headroom); answers are exact. Verified-code-native (OpenWorld's home turf).

Deterministic/offline (numpy). Emits the e80_common world format.
"""

import numpy as np

CONFIG = {"ladder": [2, 8, 32, 64, 128], "abl_n": 48,
          "abl_noise": [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0],
          "cap": 60, "n_test": 40, "seeds": [0, 1], "base": "Qwen/Qwen2.5-0.5B-Instruct"}

S = 4            # states 0..S-1
ALPHA = "ab"     # input alphabet
N_WORLDS = 360


def _delta(seed):
    rng = np.random.RandomState(seed)
    return {(s, c): int(rng.randint(0, S)) for s in range(S) for c in ALPHA}


def _table(delta):
    return "; ".join(f"from state {s} on '{c}' go to state {delta[(s, c)]}"
                     for s in range(S) for c in ALPHA)


def _instance(delta, rng):
    inp = "".join(rng.choice(list(ALPHA), int(rng.randint(4, 8))))
    s = 0
    for c in inp:
        s = delta[(s, c)]
    q = (f"A machine has {S} states (0..{S - 1}) and starts in state 0. "
         f"Transition rules: {_table(delta)}. Process the input '{inp}' one symbol at a time. "
         "What state does it end in? Reply with ONLY the state number.")
    return q, str(s)


def build_worlds():
    worlds = {}
    for i in range(N_WORLDS):
        delta = _delta(2000 + i)
        rng = np.random.RandomState(5000 + i)
        rows, answers = [], set()
        for _ in range(800):
            q, a = _instance(delta, rng)
            rows.append({"prompt": q, "label": a})
            answers.add(a)
        if len(answers) >= 2:                       # non-degenerate (reaches >1 final state)
            worlds[f"dfa_{i:03d}"] = {"classes": [str(s) for s in range(S)], "rows": rows}
    return worlds


if __name__ == "__main__":
    w = build_worlds()
    print(f"automata worlds: {len(w)}")
    for k in list(w)[:3]:
        ex = w[k]["rows"][0]
        print(f"  {k}: {ex['prompt'][:140]}... => {ex['label']}")
