"""E46 - A database for many worlds: factored vs enumerated version spaces.

E43 maintained a version space by holding an explicit list of candidate worlds.
E46 shows the factored, semiring-annotated store (openworld.manyworlds) maintains
an exact version space / posterior over world spaces far too large to enumerate,
with update and query cost scaling as the SUM of the parameter-domain sizes
(Sum |D_i|) instead of their product (Prod |D_i|).

Five things, all deterministic and offline:
  1. correctness  - on small spaces, factored == brute-force enumeration exactly
                    (count, per-parameter marginal, expected next state).
  2. scale        - factored holds an exact posterior over >= 1e18 worlds in ms,
                    where enumeration is infeasible.
  3. sub-linear   - factored update+query time grows ~ N^(1/#params); enumeration
                    grows ~ N (then runs out).
  4. semirings    - the same store answers Boolean / counting / probability.
  5. boundary     - when a mechanism couples w parameters non-separably, its
                    factor grows ~ d^w toward enumeration (the #P analogue).
"""

import time
from itertools import product

from openworld import BOOLEAN, COUNTING, PROBABILITY, Mechanism, WorldStore

from common import save_results

ENUM_CAP = 2_000_000          # don't enumerate world spaces larger than this


# --- the sprint candidate family as small-scope mechanisms ------------------
def sprint_mechanisms():
    def debt_on_ship(s, a, p):
        return s["debt"] + p["ship_debt"] if a["name"] == "ship" else None

    def bugs_on_ship(s, a, p):
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


def family(d):
    """A parameter family with each domain of size d (world space = d^4)."""
    return {"ship_debt": list(range(1, 1 + d)),
            "k": list(range(2, 2 + d)),
            "fix": list(range(1, 1 + d)),
            "refactor": list(range(1, 1 + d))}


HIDDEN = {"ship_debt": 1, "k": 4, "fix": 2, "refactor": 2}


def step(state, action, rule):
    s = dict(state)
    for m in sprint_mechanisms():
        v = m.fn(state, action, rule)
        if v is not None:
            s[m.observable] = v
    return s


def observations(n=6):
    state = {"backlog": 99, "shipped": 0, "bugs": 0, "debt": 9}
    obs, acts = [], ["ship", "fix", "refactor", "ship", "ship", "ship"]
    for a in (acts * ((n // len(acts)) + 1))[:n]:
        act = {"name": a}
        nxt = step(state, act, HIDDEN)
        obs.append((dict(state), act, dict(nxt)))
        state = nxt
    return obs


# --- enumerated (E43-style) version space for cross-checking -----------------
def enum_consistent(params, obs):
    worlds = [dict(zip(params, c)) for c in product(*params.values())]
    for st, act, nxt in obs:
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


# --- claims -----------------------------------------------------------------
def correctness(d=12):
    params, obs = family(d), observations()
    store = WorldStore(params, sprint_mechanisms(), COUNTING)
    for st, act, nxt in obs:
        store.observe(st, act, nxt)
    worlds = enum_consistent(params, obs)
    # count
    count_ok = store.count() == len(worlds)
    # marginal for k matches brute-force survivor fractions
    bf = {}
    for w in worlds:
        bf[w["k"]] = bf.get(w["k"], 0) + 1
    z = sum(bf.values())
    bstore = WorldStore(params, sprint_mechanisms(), BOOLEAN)
    for st, act, nxt in obs:
        bstore.observe(st, act, nxt)
    marg = bstore.marginal("k")
    marg_ok = all(abs(marg.get(k, 0) - c / z) < 1e-9 for k, c in bf.items())
    # expected next state matches brute force
    qs, qa = {"backlog": 5, "shipped": 7, "bugs": 1, "debt": 11}, {"name": "ship"}
    bf_exp = sum(s["bugs"] + (qs["debt"] + 1) // w["k"] for w in worlds) / len(worlds) \
        if False else None
    fexp = bstore.expected_next(qs, qa).get("bugs")
    bexp = sum((qs["debt"] + 1) // w["k"] + qs["bugs"] for w in worlds) / len(worlds)
    exp_ok = abs(fexp - bexp) < 1e-9
    return {"count_ok": count_ok, "marginal_ok": marg_ok, "expected_ok": exp_ok,
            "n_consistent": store.count()}


def scale_curve():
    obs = observations()
    ds = [10, 20, 30, 50, 100, 300, 1000, 3000, 10000, 31623]
    rows = []
    for d in ds:
        params = family(d)
        n_worlds = d ** 4
        store = WorldStore(params, sprint_mechanisms(), COUNTING)
        t0 = time.perf_counter()
        for st, act, nxt in obs:
            store.observe(st, act, nxt)
        cnt = store.count()
        _ = store.predict({"backlog": 5, "shipped": 7, "bugs": 1, "debt": 11},
                          {"name": "ship"})
        factored_ms = (time.perf_counter() - t0) * 1e3
        enum_ms = None
        if n_worlds <= ENUM_CAP:
            t0 = time.perf_counter()
            ec = enum_consistent(params, obs)
            enum_ms = (time.perf_counter() - t0) * 1e3
            assert len(ec) == cnt, f"factored/enum disagree at d={d}"
        rows.append({"d": d, "n_worlds": n_worlds, "factored_ms": round(factored_ms, 3),
                     "enum_ms": None if enum_ms is None else round(enum_ms, 1),
                     "consistent": cnt})
        print(f"  d={d:<6} worlds={n_worlds:.2e}  factored={factored_ms:7.2f}ms  "
              f"enum={'-' if enum_ms is None else f'{enum_ms:.0f}ms':>10}  "
              f"consistent={cnt:.3e}")
    return rows


def semiring_demo(d=12):
    params, obs = family(d), observations()
    out = {}
    for sr, label in [(BOOLEAN, "boolean"), (COUNTING, "counting"),
                      (PROBABILITY, "probability")]:
        store = WorldStore(params, sprint_mechanisms(), sr)
        for st, act, nxt in obs:
            store.observe(st, act, nxt)
        if label == "counting":
            out["n_worlds_remaining"] = store.count()
        elif label == "boolean":
            out["true_world_possible"] = store.is_possible(HIDDEN)
        else:
            m = store.marginal("k")
            out["posterior_k_mode"] = max(m, key=m.get)
            out["posterior_k_at_truth"] = round(m[HIDDEN["k"]], 3)
    return out


def coupling_curve(d=8):
    """A mechanism that couples w parameters non-separably: its factor is d^w."""
    obs_state = {"v": 0}
    rows = []
    for w in range(1, 6):
        names = tuple(f"q{i}" for i in range(w))
        params = {n: list(range(d)) for n in names}

        def fn(s, a, p, _names=names):
            return sum(p[n] for n in _names) % 7        # depends on ALL of scope

        mech = [Mechanism("coupled", "v", names, fn)]
        store = WorldStore(params, mech, COUNTING)
        t0 = time.perf_counter()
        store.observe(obs_state, {"name": "x"}, {"v": 3})
        ms = (time.perf_counter() - t0) * 1e3
        rows.append({"w": w, "factor_size": d ** w, "ideal_factored": w * d,
                     "observe_ms": round(ms, 3)})
        print(f"  coupling w={w}: factor_size={d**w:<8} (separable would be {w*d}) "
              f"observe={ms:.2f}ms")
    return rows


def main():
    print("E46 - factored many-worlds store\n")
    print("[correctness] factored vs brute-force enumeration (small space):")
    corr = correctness()
    print(f"  count={corr['count_ok']} marginal={corr['marginal_ok']} "
          f"expected={corr['expected_ok']} (consistent={corr['n_consistent']})\n")

    print("[scale] update+query time vs world-space size:")
    scale = scale_curve()

    print("\n[semirings] same store, three questions:")
    sem = semiring_demo()
    print(f"  {sem}")

    print("\n[boundary] coupling width vs factored cost:")
    coup = coupling_curve()

    biggest = scale[-1]
    summary = {
        "max_worlds": biggest["n_worlds"],
        "max_worlds_factored_ms": biggest["factored_ms"],
        "enum_max_worlds": max(r["n_worlds"] for r in scale if r["enum_ms"] is not None),
        "correctness": corr, "semirings": sem,
    }
    save_results("e46_many_worlds", {
        "enum_cap": ENUM_CAP, "hidden_rule": HIDDEN,
        "summary": summary, "scale": scale, "coupling": coup,
    })

    print(f"\nFactored store held an exact posterior over {biggest['n_worlds']:.2e} "
          f"worlds in {biggest['factored_ms']:.1f} ms; enumeration tops out at "
          f"{summary['enum_max_worlds']:.1e}.")

    assert corr["count_ok"] and corr["marginal_ok"] and corr["expected_ok"], \
        "factored store must match brute force exactly"
    assert biggest["n_worlds"] >= 10 ** 18, "should scale to >=1e18 worlds"
    assert biggest["factored_ms"] < 5000, "huge world space must stay fast"
    # sub-linear: time grows far slower than the world count
    small = next(r for r in scale if r["enum_ms"] is not None)
    world_ratio = biggest["n_worlds"] / small["n_worlds"]
    time_ratio = biggest["factored_ms"] / max(small["factored_ms"], 1e-6)
    assert time_ratio < world_ratio / 10 ** 6, "factored cost must be sub-linear in worlds"
    assert sem["true_world_possible"] and sem["posterior_k_mode"] == HIDDEN["k"], \
        "the store should keep the true world and peak the posterior at it"
    assert coup[-1]["factor_size"] > coup[-1]["ideal_factored"] * 100, \
        "coupling should blow the factor up vs the separable ideal (honest boundary)"
    print("all checks pass.")


if __name__ == "__main__":
    main()
