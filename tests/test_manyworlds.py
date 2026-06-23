"""Tests for the factored many-worlds store: factored answers must match brute
force exactly, and the store must scale past where enumeration can go."""

from itertools import product

from openworld import BOOLEAN, COUNTING, Mechanism, WorldStore


# A small sprint-style candidate family parameterizing the dynamics.
def sprint_mechanisms():
    def debt_on_ship(s, a, p):
        return s["debt"] + p["ship_debt"] if a["name"] == "ship" else None

    def bugs_on_ship(s, a, p):
        # uses the debt AFTER the ship increment, as in the real rule
        return s["bugs"] + (s["debt"] + 1) // p["k"] if a["name"] == "ship" else None

    def bugs_on_fix(s, a, p):
        return max(0, s["bugs"] - p["fix"]) if a["name"] == "fix" else None

    def debt_on_refactor(s, a, p):
        return max(0, s["debt"] - p["refactor"]) if a["name"] == "refactor" else None

    return [
        Mechanism("debt_on_ship", "debt", ("ship_debt",), debt_on_ship),
        Mechanism("bugs_on_ship", "bugs", ("k",), bugs_on_ship),
        Mechanism("bugs_on_fix", "bugs", ("fix",), bugs_on_fix),
        Mechanism("debt_on_refactor", "debt", ("refactor",), debt_on_refactor),
    ]


PARAMS = {"ship_debt": [1, 2], "k": [2, 3, 4, 5], "fix": [1, 2, 3],
          "refactor": [1, 2, 3]}
TRUE = {"ship_debt": 1, "k": 4, "fix": 2, "refactor": 2}


def true_step(state, action):
    s = dict(state)
    for m in sprint_mechanisms():
        v = m.fn(state, action, TRUE)
        if v is not None:
            s[m.observable] = v
    return s


def brute_consistent(observations):
    """Explicit version space (E43 style) for cross-checking."""
    worlds = [dict(zip(PARAMS, c)) for c in product(*PARAMS.values())]
    for st, act, nxt in observations:
        keep = []
        for w in worlds:
            ok = True
            for m in sprint_mechanisms():
                pred = m.fn(st, act, {p: w[p] for p in m.scope})
                if pred is not None and pred != nxt.get(m.observable):
                    ok = False
                    break
            if ok:
                keep.append(w)
        worlds = keep
    return worlds


def make_observations():
    state = {"backlog": 12, "shipped": 0, "bugs": 0, "debt": 9}
    obs = []
    for a in ["ship", "fix", "refactor", "ship", "ship"]:
        act = {"name": a}
        nxt = true_step(state, act)
        obs.append((dict(state), act, dict(nxt)))
        state = nxt
    return obs


def test_count_matches_bruteforce():
    obs = make_observations()
    store = WorldStore(PARAMS, sprint_mechanisms(), COUNTING)
    for st, act, nxt in obs:
        store.observe(st, act, nxt)
    assert store.count() == len(brute_consistent(obs))


def test_is_possible_matches_bruteforce():
    obs = make_observations()
    store = WorldStore(PARAMS, sprint_mechanisms(), BOOLEAN)
    for st, act, nxt in obs:
        store.observe(st, act, nxt)
    survivors = {tuple(sorted(w.items())) for w in brute_consistent(obs)}
    for combo in product(*PARAMS.values()):
        w = dict(zip(PARAMS, combo))
        assert store.is_possible(w) == (tuple(sorted(w.items())) in survivors)


def test_true_world_survives_and_identified():
    obs = make_observations()
    store = WorldStore(PARAMS, sprint_mechanisms(), BOOLEAN)
    for st, act, nxt in obs:
        store.observe(st, act, nxt)
    assert store.is_possible(TRUE)
    # marginal puts all surviving mass for k on values consistent with truth
    assert store.marginal("k")[4] > 0


def test_predict_matches_bruteforce_expectation():
    obs = make_observations()
    store = WorldStore(PARAMS, sprint_mechanisms(), BOOLEAN)
    for st, act, nxt in obs:
        store.observe(st, act, nxt)
    worlds = brute_consistent(obs)
    state = {"backlog": 5, "shipped": 7, "bugs": 1, "debt": 11}
    act = {"name": "ship"}
    # brute-force expected bugs across consistent worlds
    vals = [m.fn for m in sprint_mechanisms() if m.name == "bugs_on_ship"][0]
    exp = sum(vals(state, act, {"k": w["k"]}) for w in worlds) / len(worlds)
    assert abs(store.expected_next(state, act)["bugs"] - exp) < 1e-9


def test_scales_past_enumeration():
    # widen domains so the world space is enormous; the store stays cheap
    big = {"ship_debt": list(range(1, 50)), "k": list(range(2, 200)),
           "fix": list(range(1, 100)), "refactor": list(range(1, 100))}
    store = WorldStore(big, sprint_mechanisms(), COUNTING)
    assert store.total_worlds() > 10 ** 7
    state = {"backlog": 12, "shipped": 0, "bugs": 0, "debt": 9}
    # one ship observation under a fixed hidden rule
    hidden = {"ship_debt": 1, "k": 4, "fix": 2, "refactor": 2}
    nxt = dict(state)
    nxt["debt"] = state["debt"] + hidden["ship_debt"]
    nxt["bugs"] = state["bugs"] + (state["debt"] + 1) // hidden["k"]
    store.observe(state, {"name": "ship"}, nxt)
    assert 0 < store.count() < store.total_worlds()
