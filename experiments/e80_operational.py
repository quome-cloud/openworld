"""E80 operational decisions: world-time compute on priority/triage decision families.

A WORLD is a priority RULE (a weight vector over item features, e.g. severity/wait/risk); an
instance presents several items with feature values; the answer is the highest-priority item
under that rule. The rule's weights are GIVEN in the prompt, so held-out rules are solvable by
*applying the weighted-sum-then-argmax skill* -- shared across worlds. Many weight vectors = many
worlds; small LLMs are imperfect at multi-attribute arithmetic comparison (headroom); exact.

Deterministic/offline (numpy). Emits the e80_common world format.
"""

import numpy as np

CONFIG = {"ladder": [2, 8, 32, 64, 128], "abl_n": 48,
          "abl_noise": [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0],
          "cap": 60, "n_test": 40, "seeds": [0, 1], "base": "Qwen/Qwen2.5-0.5B-Instruct"}

FEATURES = ["severity", "wait", "risk"]
K_ITEMS = 4
N_WORLDS = 320


def _instance(w, rng):
    items = [rng.randint(0, 10, len(FEATURES)) for _ in range(K_ITEMS)]
    scores = [float(np.dot(w, it)) for it in items]
    ans = int(np.argmax(scores))
    desc = " + ".join(f"{w[j]:.1f}*{FEATURES[j]}" for j in range(len(FEATURES)))
    lines = "; ".join(f"item {k} ("
                      + ", ".join(f"{FEATURES[j]}={items[k][j]}" for j in range(len(FEATURES))) + ")"
                      for k in range(K_ITEMS))
    q = (f"Priority score = {desc}. Items: {lines}. Which item has the highest priority? "
         "Reply with ONLY the item number.")
    return q, str(ans)


def build_worlds():
    worlds = {}
    for i in range(N_WORLDS):
        wr = np.random.RandomState(3000 + i)
        w = np.round(wr.uniform(0.1, 2.0, len(FEATURES)), 1)
        rng = np.random.RandomState(6000 + i)
        rows, answers = [], set()
        for _ in range(800):
            q, a = _instance(w, rng)
            rows.append({"prompt": q, "label": a})
            answers.add(a)
        if len(answers) >= 2:
            worlds[f"rule_{i:03d}"] = {"classes": [str(k) for k in range(K_ITEMS)], "rows": rows}
    return worlds


if __name__ == "__main__":
    w = build_worlds()
    print(f"operational worlds: {len(w)}")
    for k in list(w)[:3]:
        ex = w[k]["rows"][0]
        print(f"  {k}: {ex['prompt'][:150]}... => {ex['label']}")
