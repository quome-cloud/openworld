"""E20 - Rule-complexity stress test (round-3 review item Q1).

Generates parametric worlds with R interacting rules (R in {4, 8, 12, 16})
from a seeded spec, builds a ground-truth interpreter for the same spec, and
asks the 7B generator to synthesize dynamics from the rule text alone (three
attempts per R). Probe accuracy versus rule count locates the complexity
cliff.

Rule semantics (stated in the prompt to remove order ambiguity): all rules
matching the chosen action read the PRE-action state; their effects are
summed and applied together; fields are clamped at 0.
"""

import random as pyrandom

from openworld import OllamaLLM, World, WorldState
from openworld.state import Action
from openworld.verify import SynthesisError

from common import GENERATOR_MODEL, require_ollama, save_results, wilson_ci

R_VALUES = [4, 8, 12, 16]
ATTEMPTS = 3
N_PROBES = 20
N_ACTIONS = 4
SPEC_SEED = 31


def make_spec(n_rules, rng):
    """A list of rule dicts plus the field/action vocabulary."""
    n_fields = max(4, n_rules // 2 + 2)
    fields = [f"f{i}" for i in range(n_fields)]
    actions = [f"a{i}" for i in range(N_ACTIONS)]
    rules = []
    for _ in range(n_rules):
        kind = rng.choice(["inc", "dec", "cond_inc", "cond_dec"])
        rule = {
            "kind": kind,
            "action": rng.choice(actions),
            "field": rng.choice(fields),
            "amount": rng.randint(1, 3),
        }
        if kind.startswith("cond"):
            rule["cond_field"] = rng.choice(fields)
            rule["threshold"] = rng.randint(1, 6)
        rules.append(rule)
    return {"fields": fields, "actions": actions, "rules": rules}


def rule_text(rule):
    base = f"On action '{rule['action']}'"
    if rule["kind"] == "inc":
        return f"{base}: {rule['field']} increases by {rule['amount']}."
    if rule["kind"] == "dec":
        return f"{base}: {rule['field']} decreases by {rule['amount']}."
    cond = (f"if {rule['cond_field']} is greater than {rule['threshold']} "
            f"in the PRE-action state")
    verb = "increases" if rule["kind"] == "cond_inc" else "decreases"
    return f"{base}: {cond}, {rule['field']} {verb} by {rule['amount']}."


def spec_rules_text(spec):
    lines = [rule_text(r) for r in spec["rules"]]
    lines.append(
        "Semantics: apply ALL rules whose action matches the chosen action. "
        "Every condition reads the PRE-action state. Sum all matching effects "
        "per field, apply them together, then clamp every field at a minimum "
        "of 0. Unknown actions and 'noop' change nothing."
    )
    return lines


def oracle_fn(spec):
    def oracle(state, action):
        s = dict(state)
        deltas = {f: 0 for f in spec["fields"]}
        for rule in spec["rules"]:
            if rule["action"] != action["name"]:
                continue
            if rule["kind"].startswith("cond"):
                if not state[rule["cond_field"]] > rule["threshold"]:
                    continue
            sign = 1 if rule["kind"].endswith("inc") or rule["kind"] == "inc" else -1
            deltas[rule["field"]] += sign * rule["amount"]
        for f in spec["fields"]:
            s[f] = max(0, state[f] + deltas[f])
        return s
    return oracle


def make_probes(spec, rng):
    probes = []
    for _ in range(N_PROBES):
        state = {f: rng.randint(0, 10) for f in spec["fields"]}
        probes.append((state, Action(rng.choice(spec["actions"]))))
    return probes


def main():
    require_ollama(GENERATOR_MODEL)
    rows = []
    for n_rules in R_VALUES:
        spec_rng = pyrandom.Random(SPEC_SEED + n_rules)
        spec = make_spec(n_rules, spec_rng)
        oracle = oracle_fn(spec)
        probes = make_probes(spec, pyrandom.Random(SPEC_SEED + 100 + n_rules))
        initial = {f: 5 for f in spec["fields"]}
        for attempt in range(ATTEMPTS):
            llm = OllamaLLM(model=GENERATOR_MODEL, temperature=0.7,
                            options={"seed": 11000 + attempt})
            world = World(
                name=f"complexity-{n_rules}",
                description=(f"A synthetic system with {len(spec['fields'])} "
                             "non-negative integer counters governed by the rules."),
                initial_state=dict(initial),
                actions=list(spec["actions"]),
                rules=spec_rules_text(spec),
                llm=llm,
            )
            record = {"n_rules": n_rules, "attempt": attempt}
            try:
                transition = world.compile(max_iters=4)
                hits = 0
                for state, action in probes:
                    expected = oracle(dict(state), action.to_dict())
                    try:
                        actual = dict(transition.step(WorldState(state), action))
                    except Exception:
                        actual = None
                    hits += actual == expected
                record["accepted"] = True
                record["probe_accuracy"] = hits / len(probes)
            except SynthesisError:
                record["accepted"] = False
                record["probe_accuracy"] = 0.0
            rows.append(record)
            print(f"  R={n_rules} #{attempt}: accepted={record['accepted']} "
                  f"acc={record['probe_accuracy']:.2f}")

    summary = []
    for n_rules in R_VALUES:
        cell = [r for r in rows if r["n_rules"] == n_rules]
        hits = sum(round(r["probe_accuracy"] * N_PROBES) for r in cell)
        total = N_PROBES * len(cell)
        summary.append({
            "n_rules": n_rules,
            "attempts": len(cell),
            "acceptance_rate": sum(r["accepted"] for r in cell) / len(cell),
            "mean_probe_accuracy": sum(r["probe_accuracy"] for r in cell) / len(cell),
            "pooled_ci": list(wilson_ci(hits, total)),
        })
    save_results("e20_complexity", {
        "model": GENERATOR_MODEL, "r_values": R_VALUES, "n_probes": N_PROBES,
        "spec_seed": SPEC_SEED, "summary": summary, "rows": rows,
    })
    for s in summary:
        print(f"R={s['n_rules']}: mean probe accuracy {s['mean_probe_accuracy']:.2f}")


if __name__ == "__main__":
    main()
