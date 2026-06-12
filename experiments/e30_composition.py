"""E30 - Composition vs the complexity cliff.

E20 showed monolithic synthesis collapsing past R~8 rules (16 rules: 0.15
mean probe accuracy). This experiment holds total system complexity at 16
internal rules + 4 couplings and varies only HOW the synthesis problem is
posed: one 20-rule monolithic prompt over a flat namespaced state, versus
eight small prompts (4 sectors x 4 rules each, 4 one-rule bridges), each
verified independently and assembled into a CompositeWorld.

System: four structurally distinct industrial "sectors" (different
production costs/gains, recycle ratios, decay thresholds) coupled in a ring
(sector i ships surplus output into sector i+1's stock). The oracle is a
hand-written CompositeWorld; the monolithic oracle is the same semantics via
deterministic flat<->nested mapping. Both conditions are scored on the same
24 ground-truth probes; 3 synthesis replicates each (qwen2.5:7b, t=0.7,
distinct seeds).
"""

import random as pyrandom

from openworld import OllamaLLM, World, WorldState
from openworld.compose import AGG_KEY, Bridge, CompositeWorld, compile_bridge
from openworld.state import Action
from openworld.transition import FunctionTransition
from openworld.verify import SynthesisError

from common import GENERATOR_MODEL, Timer, require_ollama, save_results, wilson_ci

REPLICATES = 3
MAX_ITERS = 4
PROBE_SEED = 30
N_RANDOM_PROBES = 16
VERBS = ["produce", "recycle", "wait"]
SECTOR_INITIAL = {"stock": 6, "output": 2, "waste": 4}

# 4 sectors x 4 internal rules, structurally different per sector.
SECTORS = [
    dict(prod_cost=1, prod_gain=2, prod_waste=1, rec_in=2, rec_out=1,
         decay_thresh=5, decay_amt=1),
    dict(prod_cost=2, prod_gain=3, prod_waste=1, rec_in=3, rec_out=2,
         decay_thresh=4, decay_amt=2),
    dict(prod_cost=1, prod_gain=1, prod_waste=2, rec_in=2, rec_out=2,
         decay_thresh=6, decay_amt=1),
    dict(prod_cost=3, prod_gain=4, prod_waste=1, rec_in=4, rec_out=3,
         decay_thresh=3, decay_amt=1),
]
# 4 cross-sector couplings, ring i -> i+1 mod 4, also structurally varied.
BRIDGES = [
    dict(threshold=3, amount=2),
    dict(threshold=4, amount=1),
    dict(threshold=2, amount=2),
    dict(threshold=5, amount=3),
]
NS = [f"s{i}" for i in range(4)]


# ---------------------------------------------------------------------------
# Oracle (hand-written transitions, assembled into a CompositeWorld)
# ---------------------------------------------------------------------------

def sector_fn(p):
    def fn(state, action):
        s = dict(state)
        name = action["name"]
        if name == "produce" and s["stock"] >= p["prod_cost"]:
            s["stock"] -= p["prod_cost"]
            s["output"] += p["prod_gain"]
            s["waste"] += p["prod_waste"]
        elif name == "recycle" and s["waste"] >= p["rec_in"]:
            s["waste"] -= p["rec_in"]
            s["stock"] += p["rec_out"]
        if s["output"] > p["decay_thresh"]:
            s["waste"] += p["decay_amt"]
        for k in ("stock", "output", "waste"):
            s[k] = max(0, s[k])
        return s
    return fn


def bridge_fn(b):
    def fn(state, action):
        a, bb = dict(state["a"]), dict(state["b"])
        if a["output"] > b["threshold"]:
            a["output"] -= b["amount"]
            bb["stock"] += b["amount"]
        return {"a": a, "b": bb}
    return fn


def sector_rules(p, prefix=""):
    """The 4 internal rules of one sector. With prefix='s0_' the same rules
    are stated over flat namespaced fields for the monolithic condition."""
    f = lambda name: f"{prefix}{name}"
    return [
        f"'produce' (only when {f('stock')} >= {p['prod_cost']}): "
        f"{f('stock')} decreases by {p['prod_cost']}, {f('output')} increases "
        f"by {p['prod_gain']}, {f('waste')} increases by {p['prod_waste']}. "
        f"With insufficient stock, 'produce' has no direct effect.",
        f"'recycle' (only when {f('waste')} >= {p['rec_in']}): {f('waste')} "
        f"decreases by {p['rec_in']}, {f('stock')} increases by {p['rec_out']}. "
        f"With insufficient waste, 'recycle' has no direct effect.",
        f"Decay: after the action's direct effect (this applies to EVERY "
        f"action of this sector, including 'wait'), if {f('output')} is now "
        f"greater than {p['decay_thresh']}, {f('waste')} increases by "
        f"{p['decay_amt']}.",
        f"Clamp: no field ever goes below 0. 'wait' has no direct effect "
        f"(only the decay rule above applies).",
    ]


def bridge_rules(b):
    return [
        f"On action 'flow': if slot a's 'output' is greater than "
        f"{b['threshold']}, then a's 'output' decreases by {b['amount']} and "
        f"b's 'stock' increases by {b['amount']}. Otherwise nothing changes.",
        "All other fields of both slots are always left untouched. 'noop' "
        "changes nothing.",
    ]


def make_oracle_composite():
    children = {
        ns: World(
            name=ns, description=f"Industrial sector {ns}.",
            initial_state=dict(SECTOR_INITIAL), actions=list(VERBS),
            transition=FunctionTransition(sector_fn(SECTORS[i])),
        )
        for i, ns in enumerate(NS)
    }
    bridges = [
        Bridge(name=f"b{i}", a=NS[i], b=NS[(i + 1) % 4],
               transition=FunctionTransition(bridge_fn(BRIDGES[i])))
        for i in range(4)
    ]
    return CompositeWorld("plant", children=children, bridges=bridges)


def flatten(nested):
    return {f"{ns}_{k}": nested[ns][k] for ns in NS
            for k in ("stock", "output", "waste")}


def nest(flat):
    return {ns: {k: flat[f"{ns}_{k}"] for k in ("stock", "output", "waste")}
            for ns in NS}


def strip_agg(state):
    return {ns: dict(state[ns]) for ns in NS}


def oracle_expected(oracle, probe_state, token):
    out = oracle.transition.step(WorldState(probe_state), Action(token))
    return strip_agg(dict(out))


# ---------------------------------------------------------------------------
# Probes (>= 20 deterministic cases covering branches and couplings)
# ---------------------------------------------------------------------------

def base_state(**overrides):
    s = {ns: dict(SECTOR_INITIAL) for ns in NS}
    for key, val in overrides.items():
        ns, field = key.split("__")
        s[ns][field] = val
    return s


def make_probes():
    probes = [
        # produce fires + downstream bridge fires
        (base_state(), "s0:produce"),
        # produce + decay + bridge (s3: thresh 3, gain 4)
        (base_state(), "s3:produce"),
        # produce blocked by insufficient stock (s3 cost 3)
        (base_state(s3__stock=2), "s3:produce"),
        # recycle fires (s1: 3 waste -> 2 stock)
        (base_state(), "s1:recycle"),
        # recycle blocked by insufficient waste (s2 needs 2)
        (base_state(s2__waste=1), "s2:recycle"),
        # wait with decay and bridge both firing
        (base_state(s0__output=7), "s0:wait"),
        # wait where ALL four bridges fire in the same step
        (base_state(s0__output=4, s1__output=5, s2__output=3, s3__output=6),
         "s1:wait"),
        # produce drains stock to exactly 0; bridge fires (s2 thresh 2)
        (base_state(s2__stock=1), "s2:produce"),
    ]
    rng = pyrandom.Random(PROBE_SEED)
    for _ in range(N_RANDOM_PROBES):
        state = {ns: {"stock": rng.randint(0, 8), "output": rng.randint(0, 8),
                      "waste": rng.randint(0, 8)} for ns in NS}
        token = f"{rng.choice(NS)}:{rng.choice(VERBS)}"
        probes.append((state, token))
    return probes


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def monolithic_rules():
    lines = []
    for i, ns in enumerate(NS):
        lines.append(
            f"Sector {ns} rules - they apply ONLY when the action is one of "
            f"'{ns}_produce', '{ns}_recycle', '{ns}_wait' (the verb acts on "
            f"the {ns}_* fields):")
        lines.extend("  " + r for r in sector_rules(SECTORS[i], prefix=f"{ns}_"))
    lines.append(
        "Couplings: after the acting sector's rules (including its decay "
        "rule) are applied, ALL FOUR coupling rules below fire in order 1-4, "
        "each reading the values as already updated by the action and by "
        "earlier couplings:")
    for i in range(4):
        b = BRIDGES[i]
        lines.append(
            f"  Coupling {i + 1}: if {NS[i]}_output is greater than "
            f"{b['threshold']}, then {NS[i]}_output decreases by "
            f"{b['amount']} and {NS[(i + 1) % 4]}_stock increases by "
            f"{b['amount']}.")
    lines.append(
        "On 'noop' or any unknown action the state is returned completely "
        "unchanged (the couplings do NOT fire).")
    return lines


def nonneg_invariant(keys):
    return ("no field is ever negative",
            lambda s, keys=tuple(keys): all(s[k] >= 0 for k in keys))


def synthesize_monolithic(replicate):
    llm = OllamaLLM(model=GENERATOR_MODEL, temperature=0.7,
                    options={"seed": 13000 + replicate})
    flat_initial = flatten(base_state())
    world = World(
        name="plant-monolithic",
        description=(
            "Four industrial sectors (s0..s3), each with stock/output/waste "
            "counters, coupled in a ring. Field names are namespaced "
            "'<sector>_<field>'; action names are '<sector>_<verb>'."),
        initial_state=dict(flat_initial),
        actions=[f"{ns}_{verb}" for ns in NS for verb in VERBS],
        rules=monolithic_rules(),
        llm=llm,
    )
    record = {"condition": "monolithic", "replicate": replicate, "pieces": []}
    with Timer() as t:
        try:
            transition = world.compile(
                invariants=[nonneg_invariant(flat_initial)],
                max_iters=MAX_ITERS)
            record["accepted"] = True
            record["code"] = transition.code
        except SynthesisError as exc:
            record["accepted"] = False
            record["failure"] = str(exc)
            transition = None
    record["synthesis_seconds"] = round(t.elapsed, 1)
    return record, transition


def synthesize_compositional(replicate):
    record = {"condition": "compositional", "replicate": replicate,
              "pieces": []}
    children, bridges = {}, []
    total = 0.0
    for i, ns in enumerate(NS):
        llm = OllamaLLM(model=GENERATOR_MODEL, temperature=0.7,
                        options={"seed": 14000 + replicate * 100 + i})
        world = World(
            name=f"sector-{ns}",
            description=(
                "An industrial sector with stock/output/waste counters: "
                "production consumes stock, recycling reclaims waste, high "
                "output decays into waste."),
            initial_state=dict(SECTOR_INITIAL), actions=list(VERBS),
            rules=sector_rules(SECTORS[i]), llm=llm,
        )
        piece = {"piece": ns}
        with Timer() as t:
            try:
                transition = world.compile(
                    invariants=[nonneg_invariant(SECTOR_INITIAL)],
                    max_iters=MAX_ITERS)
                piece["accepted"] = True
                piece["code"] = transition.code
                children[ns] = World(
                    name=ns, description=world.description,
                    initial_state=dict(SECTOR_INITIAL), actions=list(VERBS),
                    transition=transition)
            except SynthesisError as exc:
                piece["accepted"] = False
                piece["failure"] = str(exc)
        piece["seconds"] = round(t.elapsed, 1)
        total += t.elapsed
        record["pieces"].append(piece)
    for i in range(4):
        b = BRIDGES[i]
        a_ns, b_ns = NS[i], NS[(i + 1) % 4]
        sample_a = {"stock": 4, "output": b["threshold"] + 2, "waste": 1}
        sample_b = {"stock": 2, "output": 1, "waste": 0}
        conserved = sample_a["output"] + sample_b["stock"]
        llm = OllamaLLM(model=GENERATOR_MODEL, temperature=0.7,
                        options={"seed": 14500 + replicate * 100 + i})
        piece = {"piece": f"bridge-{a_ns}->{b_ns}"}
        with Timer() as t:
            try:
                bridge = compile_bridge(
                    llm, name=f"b{i}", a=a_ns, b=b_ns,
                    description=(
                        f"A one-way coupling shipping surplus output from "
                        f"sector {a_ns} (slot 'a') into the stock of sector "
                        f"{b_ns} (slot 'b')."),
                    rules=bridge_rules(b),
                    sample_a=sample_a, sample_b=sample_b,
                    invariants=[
                        ("a.output + b.stock is conserved by the flow",
                         lambda s, c=conserved:
                         s["a"]["output"] + s["b"]["stock"] == c),
                        ("no field is ever negative",
                         lambda s: all(v >= 0 for slot in ("a", "b")
                                       for v in s[slot].values())),
                    ],
                    max_iters=MAX_ITERS)
                piece["accepted"] = True
                piece["code"] = bridge.transition.code
                bridges.append(bridge)
            except SynthesisError as exc:
                piece["accepted"] = False
                piece["failure"] = str(exc)
        piece["seconds"] = round(t.elapsed, 1)
        total += t.elapsed
        record["pieces"].append(piece)
    record["synthesis_seconds"] = round(total, 1)
    record["accepted"] = all(p["accepted"] for p in record["pieces"])
    composite = None
    if record["accepted"]:
        composite = CompositeWorld("plant-candidate", children=children,
                                   bridges=bridges)
    return record, composite


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_monolithic(transition, probes, expectations):
    hits = 0
    for (state, token), expected in zip(probes, expectations):
        flat_token = token.replace(":", "_")
        try:
            actual = dict(transition.step(WorldState(flatten(state)),
                                          Action(flat_token)))
        except Exception:
            actual = None
        hits += actual == flatten(expected)
    return hits


def score_compositional(composite, probes, expectations):
    hits = 0
    for (state, token), expected in zip(probes, expectations):
        try:
            out = composite.transition.step(WorldState(state), Action(token))
            actual = strip_agg(dict(out))
        except Exception:
            actual = None
        hits += actual == expected
    return hits


def main():
    require_ollama(GENERATOR_MODEL)
    oracle = make_oracle_composite()
    probes = make_probes()
    expectations = [oracle_expected(oracle, state, token)
                    for state, token in probes]
    n_probes = len(probes)
    print(f"{n_probes} probes against the hand-written oracle composite.")

    rows = []
    for replicate in range(REPLICATES):
        record, transition = synthesize_monolithic(replicate)
        hits = (score_monolithic(transition, probes, expectations)
                if transition else 0)
        record["probe_hits"] = hits
        record["probe_accuracy"] = hits / n_probes
        rows.append(record)
        print(f"  monolithic    #{replicate}: accepted={record['accepted']} "
              f"acc={record['probe_accuracy']:.2f} "
              f"({record['synthesis_seconds']}s)")

        record, composite = synthesize_compositional(replicate)
        hits = (score_compositional(composite, probes, expectations)
                if composite else 0)
        record["probe_hits"] = hits
        record["probe_accuracy"] = hits / n_probes
        rows.append(record)
        pieces_ok = sum(p["accepted"] for p in record["pieces"])
        print(f"  compositional #{replicate}: accepted={record['accepted']} "
              f"(pieces {pieces_ok}/8) acc={record['probe_accuracy']:.2f} "
              f"({record['synthesis_seconds']}s)")

    summary = []
    for condition in ("monolithic", "compositional"):
        cell = [r for r in rows if r["condition"] == condition]
        hits = sum(r["probe_hits"] for r in cell)
        total = n_probes * len(cell)
        summary.append({
            "condition": condition,
            "replicates": len(cell),
            "acceptance_rate": sum(r["accepted"] for r in cell) / len(cell),
            "mean_probe_accuracy": sum(r["probe_accuracy"] for r in cell) / len(cell),
            "pooled_ci": list(wilson_ci(hits, total)),
            "mean_synthesis_seconds": round(
                sum(r["synthesis_seconds"] for r in cell) / len(cell), 1),
        })

    save_results("e30_composition", {
        "model": GENERATOR_MODEL,
        "n_probes": n_probes,
        "probe_seed": PROBE_SEED,
        "replicates": REPLICATES,
        "sectors": SECTORS,
        "bridges": BRIDGES,
        "probes": [{"state": s, "action": t} for s, t in probes],
        "expected": expectations,
        "summary": summary,
        "rows": rows,
    })

    print(f"\n{'condition':<15} {'accept':>7} {'accuracy':>9} "
          f"{'95% CI':>16} {'synth s':>8}")
    for s in summary:
        lo, hi = s["pooled_ci"]
        print(f"{s['condition']:<15} {s['acceptance_rate']:>7.2f} "
              f"{s['mean_probe_accuracy']:>9.2f} "
              f"{f'[{lo:.2f}, {hi:.2f}]':>16} "
              f"{s['mean_synthesis_seconds']:>8.1f}")


if __name__ == "__main__":
    main()
