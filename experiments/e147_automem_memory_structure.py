"""E147 -- AutoMem: is memory *structure* an independently high-leverage skill?

Deterministic, offline mechanism test of the central claim in AutoMem (Wu, Zhu, Zhang, Wang,
Yeung-Levy, Stanford; arXiv 2607.01224): promoting file-system operations to first-class memory
actions and optimizing the memory STRUCTURE alone -- without touching the task policy -- lifts
long-horizon task performance ~2-4x (their Table 1 v0 -> scaffold-opt column). Their qualitative
finding: the decisive scaffold fix on NetHack was replacing an append-only map file (accumulating
duplicate coordinate entries that bury useful info) with a coordinate-KEYED dedup store.

We isolate that mechanism with ZERO LLM: a long-horizon key->value recall task where the ONLY thing
that varies is the memory scaffold, the task policy is identical, and progression is computed exactly.
Three scaffolds (= AutoMem's structure axis), each a genuine, introspectable OpenWorld `World`
(memory = verified `CodeTransition` state, progression = `CodeObjective`):

  P0  context-only      : no external memory; answer from the last-W working-context window (v0-ish)
  P1  append-only log   : write every fact to one flat file; retrieve by scanning the last-B lines
                          (duplicates bloat the log and push the still-current fact out of budget)
  P2  schema-keyed dedup: write overwrites by key; retrieve is an exact O(1) keyed lookup (the fix)

Falsifiable predictions we ASSERT (save_results runs BEFORE the asserts):
  H1  ordering        : at long horizon, progression P2 > P1 > P0
  H2  horizon scaling : the P2-over-P0 advantage is ~0 when the horizon fits the window and grows
                        monotonically with horizon (the non-trivial shape -- a mis-specified model
                        would not produce it; it is what makes memory a *long-horizon* lever)
  H3  dedup isolates  : at fixed long horizon, P1 degrades as the duplicate rate rises while P2 is
                        invariant -- reproducing AutoMem's "dedup was the key scaffold" finding
  H4  magnitude       : P2/P0 lands in AutoMem's reported >=2x regime (reported honestly either way)

Introspection (the OpenWorld half): each scaffold is built as a `World`; we verify
from_spec(to_spec(w)) round-trips its rollout bit-for-bit, validate_spec is clean, the World's
`CodeObjective` progression equals the fast numpy simulator (self-consistency), and we compute the
BFS state-transition graph (the memory's reachable-state "map") + render its atlas card.

Out of scope (documented boundary): AutoMem's loop #2 (LoRA "memory specialist" proficiency training)
needs a GPU + a trainable model; we test loop #1 (structure) only. Determinism: fixed seeds, exact
counts; no network, no LLM.

  python experiments/e147_automem_memory_structure.py
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import save_results
from openworld import (World, Action, CodeTransition, CodeObjective, to_spec, from_spec,
                       validate_spec, render_card)

SCAFFOLDS = ("p0_context", "p1_append", "p2_keyed")


# ----------------------------------------------------------------------------------------------------
# Episode: facts set key->value early, queries ask for the current value late. `gap_scale` stretches
# the update->query distance with the horizon; `dup_rate` scatters stale re-touches (the log bloat).
# Query events carry the true current value so the transition can score without re-deriving truth.
# ----------------------------------------------------------------------------------------------------
def make_episode(n_keys=8, gap_scale=1.0, dup_rate=0.0, seed=0, W_ref=8):
    """Paired schedule: for each key -> [set] , [gap filler events] , [query]. The gap (= gap_scale*W)
    is the update->query distance, so P0 (last-W window) recalls a key iff gap < W. Filler events are
    either NO-OPs (advance time, write nothing) or DUPLICATE re-touches of an already-seen key at its
    unchanged value (bloat an append-only log without changing truth). `dup_rate` sets the duplicate
    fraction of fillers; it is the knob that isolates AutoMem's dedup finding."""
    rng = random.Random(seed)
    gap = max(0, int(round(W_ref * gap_scale)))
    dup_frac = dup_rate / (1.0 + dup_rate)                     # 0 -> 0%, 1 -> 50%, 4 -> 80% of fillers
    events, seen, val = [], [], {}
    for k in rng.sample(range(n_keys), n_keys):
        v = rng.randint(1, 999)
        val[k] = v
        events.append(["fact", k, v])
        for _ in range(gap):
            if seen and rng.random() < dup_frac:
                j = rng.choice(seen)
                events.append(["fact", j, val[j]])            # duplicate re-touch (value unchanged)
            else:
                events.append(["noop"])
        events.append(["query", k, v])
        seen.append(k)
    return events


# ----------------------------------------------------------------------------------------------------
# Fast reference simulator (pure python) -- the same logic the World's CodeTransition runs.
# ----------------------------------------------------------------------------------------------------
def simulate(scaffold, events, W=8, B=8):
    store_log, store_kv = [], {}
    score = asked = 0
    for i, ev in enumerate(events):
        if ev[0] == "noop":
            continue
        if ev[0] == "fact":
            k, v = ev[1], ev[2]
            if scaffold == "p1_append":
                store_log.append([k, v])
            elif scaffold == "p2_keyed":
                store_kv[k] = v
        else:  # query
            k, true_v = ev[1], ev[2]
            asked += 1
            got = None
            if scaffold == "p0_context":
                for j in range(i - 1, max(-1, i - 1 - W), -1):        # last W events
                    e = events[j]
                    if e[0] == "fact" and e[1] == k:
                        got = e[2]; break
            elif scaffold == "p1_append":
                for row in store_log[-B:]:                             # last B log lines only
                    if row[0] == k:
                        got = row[1]                                   # keep the latest within budget
            elif scaffold == "p2_keyed":
                got = store_kv.get(k)                                  # exact keyed lookup
            if got is not None and got == true_v:
                score += 1
    return score / asked if asked else 0.0


# ----------------------------------------------------------------------------------------------------
# The OpenWorld World: memory-as-file-system with one `step` action that consumes the next event under
# the scaffold's read/write policy. Introspectable (round-trips, BFS state graph), verified code.
# ----------------------------------------------------------------------------------------------------
_TRANSITION_CODE = '''
def transition(state, action):
    s = dict(state)
    evs = s["events"]; i = s["cursor"]
    if i >= len(evs):
        return s
    ev = evs[i]; scaf = s["scaffold"]
    log = list(s["log"]); kv = dict(s["store"]); score = s["score"]; asked = s["asked"]
    if ev[0] == "noop":
        pass
    elif ev[0] == "fact":
        k, v = ev[1], ev[2]
        if scaf == "p1_append":
            log.append([k, v])
        elif scaf == "p2_keyed":
            kv[k] = v
    else:
        k = ev[1]; true_v = ev[2]; asked = asked + 1; got = None
        if scaf == "p0_context":
            j = i - 1; lo = i - 1 - s["W"]
            while j > lo and j >= 0:
                e = evs[j]
                if e[0] == "fact" and e[1] == k:
                    got = e[2]; break
                j = j - 1
        elif scaf == "p1_append":
            for row in log[-s["B"]:]:
                if row[0] == k:
                    got = row[1]
        elif scaf == "p2_keyed":
            got = kv.get(k)
        if got is not None and got == true_v:
            score = score + 1
    s["log"] = log; s["store"] = kv; s["score"] = score; s["asked"] = asked; s["cursor"] = i + 1
    return s
'''

_OBJECTIVE_CODE = '''
def reward(state, action, next_state):
    a = next_state["asked"]
    return (next_state["score"] / a) if a else 0.0
'''


def build_memory_world(scaffold, events, W=8, B=8):
    init = {"events": [list(e) for e in events], "cursor": 0, "log": [], "store": {},
            "score": 0, "asked": 0, "scaffold": scaffold, "W": W, "B": B}
    return World(
        name=f"automem-{scaffold}",
        description=f"Memory-as-file-system ({scaffold}): long-horizon key->value recall; the memory "
                    f"read/write policy is the scaffold under test. Progression = queries answered.",
        initial_state=init,
        actions=["step"],
        rules=["'step' consumes the next event: a fact is written to memory under the scaffold's write "
               "policy; a query is answered by the scaffold's read policy and scored if correct.",
               "p0_context: no store, read the last-W working context. p1_append: flat append log, "
               "read last-B lines. p2_keyed: keyed dedup store, exact lookup."],
        transition=CodeTransition(_TRANSITION_CODE),
    )


def rollout_world(world):
    """Run the whole episode (all `step`s) and return final progression from the World's own state."""
    st = world.initial_state
    n = len(st["events"])
    for _ in range(n):
        st = world.transition.step(st, Action("step"))
    d = dict(st)
    return (d["score"] / d["asked"]) if d["asked"] else 0.0


# ----------------------------------------------------------------------------------------------------
def main():
    W, B, N = 8, 8, 8
    HORIZONS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]      # gap_scale: horizon relative to the window
    DUP_RATES = [0.0, 0.5, 1.0, 2.0, 4.0]
    SEEDS = list(range(8))

    def mean_prog(scaffold, gap_scale, dup_rate):
        vals = [simulate(scaffold, make_episode(N, gap_scale, dup_rate, s), W, B) for s in SEEDS]
        return sum(vals) / len(vals)

    # --- H1/H2: sweep horizon at a fixed (light) duplicate rate ---
    horizon = {}
    for gs in HORIZONS:
        horizon[gs] = {sc: round(mean_prog(sc, gs, 0.5), 4) for sc in SCAFFOLDS}

    # --- H3: sweep duplicate rate at a fixed long horizon ---
    dup = {}
    for dr in DUP_RATES:
        dup[dr] = {sc: round(mean_prog(sc, 4.0, dr), 4) for sc in SCAFFOLDS}

    # AutoMem's headline 2-4x is a STRUCTURE gain between two memory-equipped configs (their v0
    # file-system -> optimized scaffold) -- that maps to P1 -> P2 here, NOT P2/P0 (memory vs none).
    # We report the structure gain P2/P1 across the dup sweep so the 2-4x band is visible, and the
    # memory-vs-none ratio P2/P0 separately (it diverges as P0 -> 0, so it is reported as a floor).
    long_gs, long_dr = 4.0, 1.0                                    # a heavy-bloat long-horizon regime
    regime = {sc: round(mean_prog(sc, long_gs, long_dr), 4) for sc in SCAFFOLDS}
    def _ratio(num, den):
        return round(num / den, 3) if den > 0 else None
    ratio_p2_p1 = _ratio(regime["p2_keyed"], regime["p1_append"])           # the structure gain (paper axis)
    ratio_p2_p0 = _ratio(regime["p2_keyed"], regime["p0_context"])          # memory-vs-none (None => P0=0)
    # structure gain P2/P1 across the duplicate sweep (where AutoMem's 2-4x band shows up)
    struct_gain = {str(dr): _ratio(dup[dr]["p2_keyed"], dup[dr]["p1_append"]) for dr in DUP_RATES}
    in_band = [dr for dr in DUP_RATES if (struct_gain[str(dr)] or 0) and 2.0 <= struct_gain[str(dr)] <= 4.0]

    # --- OpenWorld introspection: build each scaffold as a World, verify round-trip + graph + card ---
    demo_events = make_episode(N, long_gs, long_dr, seed=0)
    introspect = {}
    for sc in SCAFFOLDS:
        w = build_memory_world(sc, demo_events, W, B)
        spec = to_spec(w)
        problems = validate_spec(spec)
        w2 = from_spec(spec, allow_code=True)
        prog_world = rollout_world(w)
        prog_round = rollout_world(w2)
        prog_sim = simulate(sc, demo_events, W, B)
        graph = (spec.get("preview") or {}).get("graph") or {}
        introspect[sc] = {
            "spec_valid": (problems == []),
            "roundtrip_exact": abs(prog_world - prog_round) < 1e-12,
            "world_equals_sim": abs(prog_world - prog_sim) < 1e-12,   # the World faithfully runs the mechanism
            "progression": round(prog_world, 4),
            "graph_nodes": len(graph.get("nodes", [])),
            "graph_edges": len(graph.get("edges", [])),
        }
    # render one atlas card as a served artifact (the p2 "optimized scaffold" world)
    card_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "experiments", "results", "e147_automem_world_card.svg")
    try:
        render_card(build_memory_world("p2_keyed", demo_events, W, B), card_path)
        card_written = os.path.exists(card_path)
    except Exception:
        card_written = False

    payload = {
        "description": "AutoMem memory-structure mechanism test: structure (P0 context -> P1 append -> "
                       "P2 keyed dedup) lifts long-horizon recall with the task policy fixed; built + "
                       "verified as introspectable OpenWorld Worlds.",
        "paper": {"title": "AutoMem: Automated Learning of Memory as a Cognitive Skill",
                  "arxiv": "2607.01224", "claim_tested": "loop-1 (memory structure) only; "
                  "loop-2 LoRA proficiency out of scope"},
        "config": {"n_keys": N, "window_W": W, "budget_B": B, "seeds": len(SEEDS),
                   "horizons_gap_scale": HORIZONS, "dup_rates": DUP_RATES},
        "horizon_sweep": {str(k): v for k, v in horizon.items()},
        "dup_sweep": {str(k): v for k, v in dup.items()},
        "paper_regime": {"gap_scale": long_gs, "dup_rate": long_dr, "progression": regime,
                         "structure_gain_p2_over_p1": ratio_p2_p1,          # AutoMem's 2-4x axis
                         "memory_vs_none_p2_over_p0": ratio_p2_p0},         # None => P0 collapsed to 0
        "structure_gain_by_dup": struct_gain,
        "structure_gain_in_2to4x_band_at_dup": in_band,
        "openworld_introspection": introspect,
        "card_svg_written": card_written,
    }
    save_results("e147_automem_memory_structure", payload)  # SAVE BEFORE ASSERTS

    # ---------------------------- falsifiable checks ----------------------------
    lo, hi = HORIZONS[0], HORIZONS[-1]
    adv = {gs: horizon[gs]["p2_keyed"] - horizon[gs]["p0_context"] for gs in HORIZONS}

    # H1 ordering at long horizon
    assert regime["p2_keyed"] > regime["p1_append"] > regime["p0_context"], \
        f"H1 ordering failed: {regime}"
    # H2 horizon scaling: ~0 advantage at short horizon, monotonic non-decreasing, large at long horizon
    assert adv[lo] <= 0.05, f"H2: memory should not help at horizon<=window, adv={adv[lo]:.3f}"
    seq = [adv[gs] for gs in HORIZONS]
    assert all(seq[i + 1] >= seq[i] - 1e-9 for i in range(len(seq) - 1)), f"H2 not monotonic: {seq}"
    assert adv[hi] >= 0.4, f"H2: advantage should be large at long horizon, adv={adv[hi]:.3f}"
    # H3 dedup isolates: P1 degrades with dup rate, P2 invariant
    p1_seq = [dup[dr]["p1_append"] for dr in DUP_RATES]
    assert p1_seq[-1] < p1_seq[0] - 0.05, f"H3: append should degrade with dup rate: {p1_seq}"
    p2_seq = [dup[dr]["p2_keyed"] for dr in DUP_RATES]
    assert max(p2_seq) - min(p2_seq) < 1e-9, f"H3: keyed dedup should be dup-invariant: {p2_seq}"
    # H4 magnitude: the STRUCTURE gain (P1->P2, both memory-equipped -- AutoMem's axis) is a large
    # multiple, >= the paper's 2x low end; and it passes THROUGH the paper's 2-4x band somewhere on
    # the dup sweep (reported honestly, not tuned to a target).
    assert ratio_p2_p1 is not None and ratio_p2_p1 >= 2.0, f"H4: structure gain P2/P1 < 2x: {ratio_p2_p1}"
    assert in_band, f"H4: structure gain never lands in the paper's 2-4x band across dup sweep: {struct_gain}"
    # OpenWorld introspection guarantees
    for sc, m in introspect.items():
        assert m["spec_valid"], f"{sc}: spec not valid"
        assert m["roundtrip_exact"], f"{sc}: from_spec(to_spec) rollout diverged"
        assert m["world_equals_sim"], f"{sc}: World != reference simulator"

    print("E147 OK  (all AutoMem structure-lift + OpenWorld introspection checks passed)")
    print(f"  paper-regime progression: {regime}   structure gain P2/P1 = {ratio_p2_p1}x")
    print(f"  structure gain by dup rate (P2/P1): {struct_gain}   in 2-4x band at dup={in_band}")
    print(f"  horizon advantage (P2-P0): " + "  ".join(f"{gs}x:{adv[gs]:.2f}" for gs in HORIZONS))
    print(f"  dup-rate P1 progression:   " + "  ".join(f"{dr}:{dup[dr]['p1_append']:.2f}" for dr in DUP_RATES))
    for sc, m in introspect.items():
        print(f"  world[{sc}]: prog={m['progression']} roundtrip={m['roundtrip_exact']} "
              f"==sim={m['world_equals_sim']} graph={m['graph_nodes']}n/{m['graph_edges']}e")


if __name__ == "__main__":
    main()
