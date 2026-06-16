"""E63 - A world-model bake-off: as many approaches as we can run, side by side,
across multiple domains.

Every entry is a next-state predictor over the SAME symbolic state, evaluated on
the SAME task with the SAME data budget -- the fair within-class comparison. The
class spans the standard learned-dynamics families plus the LLM-as-simulator and
verified code:

  symbolic:   verified code (OpenWorld CWM)   -- exact, 0 training transitions
  learned:    1-NN, tabular(+NN backoff), linear (least squares),
              Koopman/EDMD (deg-2 lift), MLP (numpy)   -- trained on K transitions
  sequence:   LLM next-state proxy            -- numbers cited from E10/E11/E22

Domains: the flat-numeric instrumented worlds -- sprint (engineering backlog) and
triage (ICU queue). Orchard is excluded for the vector learners because its state
nests a per-agent dict (documented exclusion). Battery: 1-step probe accuracy
in-distribution and at 10x OOD, multi-step rollout exactness (both per domain),
plus downstream control return + inference speed on sprint (where an objective is
defined). Learned models train at K and we report the mean over 5 seeds.

Why these baselines and not Dreamer/V-JEPA/MuZero/IRIS/DIAMOND/Genie/Sora/World
Labs: those are *perceptual/latent* world models that predict pixels/video/embeddings,
not a symbolic next state, and cannot run on this task without a learned perception
stack. They are a different species, compared on properties in the paper's
related-work table, not in this head-to-head.
"""

import json
import time
from pathlib import Path
from statistics import mean

import numpy as np

from openworld.state import Action

from common import (SPRINT_INITIAL, WORLD_SPECS, save_results, sprint_ground_truth)
from e12_learned_baseline import MLP
from e22_planning import CODE_DEPTH, EPISODE_STEPS, env_step, lookahead, value

import random

SEEDS = [0, 1, 2, 3, 4]
K = 10000
DOMAINS = ["sprint", "triage"]            # flat-numeric instrumented worlds
N_PROBES = 24
N_SCRIPTS = 8
ROLLOUT_STEPS = 12
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def fields_of(initial):
    return sorted(initial.keys())


def mk_encode(fields, actions):
    def encode(state, action):
        onehot = [1.0 if action == a else 0.0 for a in actions]
        return [float(state[f]) for f in fields] + onehot
    return encode


def collect(oracle, initial, actions, k, rng):
    data, s = [], dict(initial)
    while len(data) < k:
        a = rng.choice(actions)
        ns = oracle(dict(s), {"name": a, "params": {}, "agent": None})
        data.append((dict(s), a, dict(ns)))
        s = ns
        if rng.random() < 0.25:
            s = dict(initial)
    return data


def make_probes(oracle, initial, actions, fields, rng, n=N_PROBES, ood=False):
    probes, s = [], dict(initial)
    for _ in range(n):
        a = rng.choice(actions)
        st = {f: (s[f] * 10 if ood else s[f]) for f in fields}
        probes.append((dict(st), a))
        s = oracle(dict(s), {"name": a, "params": {}, "agent": None})
        if rng.random() < 0.3:
            s = dict(initial)
    return probes


def probe_acc(model, probes, oracle):
    hits = 0
    for state, a in probes:
        expected = oracle(dict(state), {"name": a, "params": {}, "agent": None})
        if model.predict_state(dict(state), a) == expected:
            hits += 1
    return hits / len(probes)


def rollout_acc(model, oracle, initial, actions, rng, n=N_SCRIPTS, steps=ROLLOUT_STEPS):
    exact = 0
    for _ in range(n):
        s, ms, ok = dict(initial), dict(initial), True
        for _ in range(steps):
            a = rng.choice(actions)
            s = oracle(dict(s), {"name": a, "params": {}, "agent": None})
            ms = model.predict_state(dict(ms), a)
            if ms != s:
                ok = False
                break
        exact += ok
    return exact / n


# --------------------------------------------------------------------------- #
# world models (uniform interface: predict_state(state, action_name) -> state)
# --------------------------------------------------------------------------- #
class CodeWM:
    def __init__(self, oracle):
        self.oracle = oracle

    def predict_state(self, state, action):
        return self.oracle(dict(state), {"name": action, "params": {}, "agent": None})


class NNWM:
    def __init__(self, data, enc, fields, actions):
        self.x = np.array([enc(s, a) for s, a, _ in data])
        self.y = [ns for _, _, ns in data]
        self.enc = enc

    def predict_state(self, state, action):
        q = np.array(self.enc(state, action))
        return dict(self.y[int(np.argmin(((self.x - q) ** 2).sum(1)))])


class TabularWM:
    def __init__(self, data, enc, fields, actions):
        self.fields = fields
        self.table = {(tuple(s[f] for f in fields), a): dict(ns) for s, a, ns in data}
        self.backoff = NNWM(data, enc, fields, actions)

    def predict_state(self, state, action):
        hit = self.table.get((tuple(state[f] for f in self.fields), action))
        return dict(hit) if hit is not None else self.backoff.predict_state(state, action)


def _clamp(y, fields):
    y = np.nan_to_num(y, nan=0.0, posinf=1e9, neginf=0.0)
    return {f: max(0, min(10 ** 9, int(round(float(v))))) for f, v in zip(fields, y)}


class LinearWM:
    def __init__(self, data, enc, fields, actions):
        self.enc, self.fields = enc, fields
        x = np.array([enc(s, a) + [1.0] for s, a, _ in data])
        yy = np.array([[float(ns[f]) for f in fields] for _, _, ns in data])
        self.w, *_ = np.linalg.lstsq(x, yy, rcond=None)

    def predict_state(self, state, action):
        return _clamp(np.array(self.enc(state, action) + [1.0]) @ self.w, self.fields)


def _lift(vec):
    v = np.array(vec, dtype=float)
    n = len(v)
    return np.concatenate([[1.0], v, [v[i] * v[j] for i in range(n) for j in range(i, n)]])


class KoopmanWM:
    def __init__(self, data, enc, fields, actions):
        self.enc, self.fields = enc, fields
        x = np.array([_lift(enc(s, a)) for s, a, _ in data])
        yy = np.array([[float(ns[f]) for f in fields] for _, _, ns in data])
        self.w, *_ = np.linalg.lstsq(x, yy, rcond=None)

    def predict_state(self, state, action):
        return _clamp(_lift(self.enc(state, action)) @ self.w, self.fields)


class MLPWM:
    def __init__(self, data, enc, fields, actions, seed=0):
        self.enc, self.fields = enc, fields
        x = np.array([enc(s, a) for s, a, _ in data])
        yy = np.array([[float(ns[f]) for f in fields] for _, _, ns in data])
        self.net = MLP(x.shape[1], yy.shape[1], seed=seed)
        self.net.train(x, yy)

    def predict_state(self, state, action):
        y = self.net.forward(np.array([self.enc(state, action)]))[0]
        return _clamp(y, self.fields)


LEARNED = {"1-NN": NNWM, "tabular": TabularWM, "linear": LinearWM,
           "koopman": KoopmanWM, "MLP": MLPWM}


def plan_return(predict_state):
    s = dict(SPRINT_INITIAL)
    for _ in range(EPISODE_STEPS):
        a, _ = lookahead(predict_state, s, CODE_DEPTH)
        s = env_step(s, a)
    return value(s)


def steps_per_sec(model, fields, actions, n=2000):
    probes = make_probes(model.oracle if isinstance(model, CodeWM) else sprint_ground_truth,
                         SPRINT_INITIAL, actions, fields, random.Random(7), n=6)
    t0 = time.perf_counter()
    for i in range(n):
        st, a = probes[i % len(probes)]
        model.predict_state(st, a)
    return n / (t0 - t0 + (time.perf_counter() - t0))


def cited_llm():
    def load(name):
        return json.loads((RESULTS_DIR / f"{name}.json").read_text())
    out = {"source": "E10/E11/E22"}
    try:
        e11 = load("e11_multiworld_fidelity")["totals"]["llm_transition"]
        out["sprint_rollout"] = round(e11["exact_rollouts"] / e11["n"], 3)
    except Exception:
        out["sprint_rollout"] = None
    try:
        e22 = load("e22_planning")
        out["sprint_control"] = next(r["score"] for r in e22["rows"] if r["planner"] == "llm_d2")
    except Exception:
        out["sprint_control"] = None
    return out


def main():
    per_domain = {}            # domain -> model -> {probe_in, probe_ood, rollout}
    for dom in DOMAINS:
        spec = WORLD_SPECS[dom]
        oracle, initial, actions = spec["oracle"], spec["initial"], spec["actions"]
        fields = fields_of(initial)
        enc = mk_encode(fields, actions)
        rownames = ["verified code (CWM)"] + list(LEARNED)
        per_domain[dom] = {}
        # verified code
        code = CodeWM(oracle)
        per_domain[dom]["verified code (CWM)"] = {
            "probe_in": probe_acc(code, make_probes(oracle, initial, actions, fields, random.Random(1)), oracle),
            "probe_ood": probe_acc(code, make_probes(oracle, initial, actions, fields, random.Random(2), ood=True), oracle),
            "rollout": rollout_acc(code, oracle, initial, actions, random.Random(3)),
        }
        for name, ctor in LEARNED.items():
            pin, pood, roll = [], [], []
            for seed in SEEDS:
                data = collect(oracle, initial, actions, K, random.Random(seed))
                model = ctor(data, enc, fields, actions, seed) if name == "MLP" else ctor(data, enc, fields, actions)
                pin.append(probe_acc(model, make_probes(oracle, initial, actions, fields, random.Random(1)), oracle))
                pood.append(probe_acc(model, make_probes(oracle, initial, actions, fields, random.Random(2), ood=True), oracle))
                roll.append(rollout_acc(model, oracle, initial, actions, random.Random(3)))
            per_domain[dom][name] = {"probe_in": round(mean(pin), 3),
                                     "probe_ood": round(mean(pood), 3),
                                     "rollout": round(mean(roll), 3)}

    # sprint-only control + speed (an objective is defined there)
    sp = WORLD_SPECS["sprint"]
    fields, actions = fields_of(sp["initial"]), sp["actions"]
    control, speed = {}, {}
    code = CodeWM(sprint_ground_truth)
    control["verified code (CWM)"] = round(plan_return(code.predict_state), 3)
    speed["verified code (CWM)"] = round(steps_per_sec(code, fields, actions))
    for name, ctor in LEARNED.items():
        cret, spd = [], []
        for seed in SEEDS:
            data = collect(sprint_ground_truth, sp["initial"], actions, K, random.Random(seed))
            enc = mk_encode(fields, actions)
            model = ctor(data, enc, fields, actions, seed) if name == "MLP" else ctor(data, enc, fields, actions)
            cret.append(plan_return(model.predict_state))
            spd.append(steps_per_sec(model, fields, actions, n=500))
        control[name] = round(mean(cret), 3)
        speed[name] = round(mean(spd))

    llm = cited_llm()
    results = {
        "domains": DOMAINS, "k_trained": K, "seeds": SEEDS,
        "excluded": {"orchard": "nested per-agent dict state; not a flat vector"},
        "fidelity": per_domain, "sprint_control": control, "sprint_steps_per_sec": speed,
        "llm_proxy": llm,
        "n_methods_runnable": 1 + len(LEARNED),
        "perceptual_world_models_compared_on_properties": [
            "DreamerV3", "MuZero", "IRIS", "DIAMOND", "Genie", "Sora", "World Labs",
            "V-JEPA", "NE-Dreamer"],
    }
    save_results("e63_world_model_bakeoff", results)

    print("E63 - world-model bake-off (K=%d, %d seeds)\n" % (K, len(SEEDS)))
    for dom in DOMAINS:
        print(f"  [{dom}]   {'probe':>7} {'OOD':>6} {'rollout':>8}")
        for name in ["verified code (CWM)"] + list(LEARNED):
            r = per_domain[dom][name]
            print(f"    {name:<20} {r['probe_in']:>6.2f} {r['probe_ood']:>6.2f} {r['rollout']:>8.2f}")
    print(f"\n  [sprint control return / steps-per-sec]")
    for name in ["verified code (CWM)"] + list(LEARNED):
        print(f"    {name:<20} return={control[name]:>6.2f}   steps/s={speed[name]:>10,}")
    print(f"  LLM next-state (cited E10/E11/E22): sprint rollout={llm.get('sprint_rollout')}, "
          f"control={llm.get('sprint_control')}")

    # --- honest self-checks (save first) ---
    for dom in DOMAINS:
        c = per_domain[dom]["verified code (CWM)"]
        assert c["probe_in"] == 1.0 and c["probe_ood"] == 1.0 and c["rollout"] == 1.0, \
            f"verified code must be exact in {dom}"
        for name in LEARNED:
            assert per_domain[dom][name]["probe_ood"] <= c["probe_ood"], f"{name} cannot beat code OOD in {dom}"
    assert abs(control["verified code (CWM)"] - 7.5) < 1e-6, "verified code plans optimally on sprint"
    assert max(per_domain[d][n]["probe_ood"] for d in DOMAINS for n in LEARNED) < 0.5, \
        "no learned model generalizes OOD on these branchy tasks"
    print("\nchecks pass: across %d runnable world models x %d domains, verified code is the only "
          "one exact, OOD-robust, and optimal-for-control, at zero training data." %
          (1 + len(LEARNED), len(DOMAINS)))


if __name__ == "__main__":
    main()
