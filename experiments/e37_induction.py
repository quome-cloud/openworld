"""E37 - Induction on equal footing: synthesize dynamics from TRACES, not rules.

The central comparison elsewhere in the paper hands the synthesizer the rule
text while the learned baselines must infer dynamics from sampled transitions
-- an information asymmetry. E37 removes it: the LLM is given ONLY observed
(state, action, next_state) examples (no rule text) and must induce a
deterministic transition() program; acceptance is verified by exact
reproduction of the observed transitions. It is then held to the SAME
held-out probe suites as the learned baselines (E12): branch-covering probes
in-distribution and at 10x out-of-distribution scale.

The question this answers: does verified code *induction from data* still beat
statistical learning out of distribution -- i.e., is the win about inducing
symbolic rules that extrapolate, not just about being handed the rules? Honest
either way: if code-from-traces also collapses OOD, the rule text was doing
the work and the paper's claims should narrow.

Conditions (equal information = the same K random-policy transitions):
  code_from_traces : LLM induces transition() from the traces, verified to
                     reproduce them (qwen2.5:7b).
  mlp / knn1       : the E12 learned baselines on the same K.
  code_from_rules  : reference upper anchor -- synthesis WITH the rule text
                     (0 transitions), i.e. what the specification buys.
"""

import random

import numpy as np

from openworld import OllamaLLM
from openworld.parsing import extract_code
from openworld.sandbox import run_transition_code

from common import (
    SPRINT_ACTIONS, SPRINT_DESCRIPTION, SPRINT_INITIAL, SPRINT_PROBES,
    SPRINT_PROBES_SCALED, SPRINT_RULES, require_ollama, save_results,
    sprint_ground_truth, wilson_ci,
)
from e12_learned_baseline import MLP, encode, FIELDS, ACTIONS

MODEL = "qwen2.5:7b"
KS = [100, 1000]
MAX_DISTINCT_SHOWN = 40   # examples placed in the induction prompt
SYNTH_ATTEMPTS = 4
REPLICATES = 3
SEED = 37

INDUCE_SYSTEM = (
    "You are a program-induction engine. You are given examples of a "
    "deterministic environment's transitions as (state, action) -> next_state. "
    "Infer the underlying rules and reply with ONLY a python code block "
    "defining `def transition(state, action):` that reproduces them. `state` "
    "is a dict; `action` is a dict with key 'name'. Return the next state as a "
    "dict. Use only pure python. Do NOT assume any rule not supported by the "
    "examples."
)


def collect(k, rng):
    data, state = [], dict(SPRINT_INITIAL)
    while len(data) < k:
        a = rng.choice(SPRINT_ACTIONS + ["noop"])
        nxt = sprint_ground_truth(state, {"name": a, "params": {}, "agent": None})
        data.append((dict(state), a, dict(nxt)))
        state = nxt
        if state["backlog"] == 0 and rng.random() < 0.3:
            state = dict(SPRINT_INITIAL)
    return data


def distinct(traces, cap):
    seen, out = set(), []
    for s, a, n in traces:
        key = (tuple(sorted(s.items())), a)
        if key not in seen:
            seen.add(key)
            out.append((s, a, n))
        if len(out) >= cap:
            break
    return out


def induce_prompt(traces):
    shown = distinct(traces, MAX_DISTINCT_SHOWN)
    lines = [f"State fields: {FIELDS}. Actions: {SPRINT_ACTIONS + ['noop']}.",
             "", "Observed transitions:"]
    for s, a, n in shown:
        lines.append(f"  state={s}, action={{'name': {a!r}}} -> {n}")
    lines.append("")
    lines.append("Write transition(state, action) consistent with ALL of these.")
    return "\n".join(lines)


def reproduces(code, traces):
    """Fraction of training traces the code reproduces exactly (verification)."""
    ok = 0
    for s, a, n in traces:
        try:
            pred = run_transition_code(code, dict(s), {"name": a, "params": {}, "agent": None})
        except Exception:
            return 0.0
        if pred == n:
            ok += 1
    return ok / len(traces)


def _expected(s, act):
    return sprint_ground_truth(dict(s), act.to_dict())


def probe_acc_code(code, probes):
    hits = 0
    for s, act in probes:
        expected = _expected(s, act)
        try:
            pred = run_transition_code(code, dict(s), act.to_dict())
        except Exception:
            continue
        if pred == expected:
            hits += 1
    return hits / len(probes)


def induce_from_traces(llm, traces):
    """Synthesis loop verified against the observed traces; return best code."""
    best_code, best_repro = None, -1.0
    prompt = induce_prompt(traces)
    for _ in range(SYNTH_ATTEMPTS):
        code = extract_code(llm.ask(prompt, system=INDUCE_SYSTEM))
        repro = reproduces(code, traces)
        if repro > best_repro:
            best_code, best_repro = code, repro
        if repro == 1.0:
            break
    return best_code, best_repro


# ---- learned baselines (reuse E12 machinery) -------------------------------
def _round(vec):
    return [int(round(v)) for v in vec]


def train_mlp(traces, seed):
    X = np.array([encode(s, a) for s, a, _ in traces], dtype=float)
    Y = np.array([[n[f] for f in FIELDS] for _, _, n in traces], dtype=float)
    net = MLP(X.shape[1], Y.shape[1], hidden=64, seed=seed)
    net.train(X, Y, epochs=3000, lr=1e-3)
    return net


def probe_acc_mlp(net, probes):
    hits = 0
    for s, act in probes:
        expected = _expected(s, act)
        pred = _round(net.forward(np.array([encode(s, act.name)], dtype=float))[0])
        if {f: pred[i] for i, f in enumerate(FIELDS)} == {f: expected[f] for f in FIELDS}:
            hits += 1
    return hits / len(probes)


def probe_acc_knn(traces, probes):
    X = np.array([encode(s, a) for s, a, _ in traces], dtype=float)
    Y = [n for _, _, n in traces]
    hits = 0
    for s, act in probes:
        expected = _expected(s, act)
        q = np.array(encode(s, act.name), dtype=float)
        pred = Y[int(((X - q) ** 2).sum(1).argmin())]
        if pred == expected:
            hits += 1
    return hits / len(probes)


def main():
    require_ollama(MODEL)
    rng = random.Random(SEED)
    rules_llm = OllamaLLM(model=MODEL, temperature=0.2, options={"seed": SEED})

    # Reference anchor: synthesis WITH rule text, 0 transitions (what rules buy)
    from openworld.world import World
    from openworld.state import Action
    rules_world = World(name="sprint", description=SPRINT_DESCRIPTION,
                        initial_state=SPRINT_INITIAL, actions=SPRINT_ACTIONS,
                        rules=SPRINT_RULES, llm=rules_llm)
    rules_world.compile(invariants=[("nonneg", lambda s: all(s[f] >= 0 for f in FIELDS))])
    rules_code = rules_world.transition.code
    code_from_rules = {
        "probe_in_dist": probe_acc_code(rules_code, SPRINT_PROBES),
        "probe_ood_10x": probe_acc_code(rules_code, SPRINT_PROBES_SCALED),
    }

    rows = []
    for k in KS:
        for rep in range(REPLICATES):
            r = random.Random(SEED + rep)
            traces = collect(k, r)
            llm = OllamaLLM(model=MODEL, temperature=0.4, options={"seed": SEED + rep})
            code, repro = induce_from_traces(llm, traces)
            net = train_mlp(traces, seed=rep)
            row = {
                "k": k, "replicate": rep, "n_distinct_shown": len(distinct(traces, MAX_DISTINCT_SHOWN)),
                "code_from_traces": {
                    "train_reproduction": repro,
                    "probe_in_dist": probe_acc_code(code, SPRINT_PROBES),
                    "probe_ood_10x": probe_acc_code(code, SPRINT_PROBES_SCALED),
                },
                "mlp": {
                    "probe_in_dist": probe_acc_mlp(net, SPRINT_PROBES),
                    "probe_ood_10x": probe_acc_mlp(net, SPRINT_PROBES_SCALED),
                },
                "knn1": {
                    "probe_in_dist": probe_acc_knn(traces, SPRINT_PROBES),
                    "probe_ood_10x": probe_acc_knn(traces, SPRINT_PROBES_SCALED),
                },
            }
            rows.append(row)
            c = row["code_from_traces"]
            print(f"  k={k} rep={rep}: induce repro={repro:.2f} "
                  f"in-dist={c['probe_in_dist']:.2f} ood={c['probe_ood_10x']:.2f} | "
                  f"mlp ood={row['mlp']['probe_ood_10x']:.2f} | "
                  f"knn ood={row['knn1']['probe_ood_10x']:.2f}")

    def agg(cond, key, k):
        vals = [r[cond][key] for r in rows if r["k"] == k]
        return sum(vals) / len(vals)

    summary = {"code_from_rules": code_from_rules, "by_k": []}
    for k in KS:
        summary["by_k"].append({
            "k": k,
            "code_from_traces_in_dist": agg("code_from_traces", "probe_in_dist", k),
            "code_from_traces_ood": agg("code_from_traces", "probe_ood_10x", k),
            "mlp_in_dist": agg("mlp", "probe_in_dist", k),
            "mlp_ood": agg("mlp", "probe_ood_10x", k),
            "knn1_in_dist": agg("knn1", "probe_in_dist", k),
            "knn1_ood": agg("knn1", "probe_ood_10x", k),
        })
    save_results("e37_induction", {
        "model": MODEL, "ks": KS, "replicates": REPLICATES,
        "max_distinct_shown": MAX_DISTINCT_SHOWN,
        "summary": summary, "rows": rows,
    })
    print("\ncode_from_rules (anchor):", code_from_rules)
    for s in summary["by_k"]:
        print(f"k={s['k']}: traces in-dist {s['code_from_traces_in_dist']:.2f} "
              f"ood {s['code_from_traces_ood']:.2f} | mlp ood {s['mlp_ood']:.2f} | "
              f"knn ood {s['knn1_ood']:.2f}")


if __name__ == "__main__":
    main()
