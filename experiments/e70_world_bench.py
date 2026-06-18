"""E70 - Are the prototyped worlds good world models? A benchmark of the 100 recipes.

E68 shows 100 worlds are built fast; E68b shows they run. Neither asks whether they are
*good* world models. The honest obstacle: these worlds are synthesized from one-line
prompts with NO external oracle, so fidelity-to-reality is unmeasurable. What IS
measurable -- and is exactly what an agent needs from a simulator -- are the world's
*intrinsic* properties, all computable from the verified transition code:

  (A) Determinism      - same seed + same actions => identical trajectory (replayable).
  (B) Safety/invariants- over long random rollouts the transition never throws, the state
                         schema never drifts (keys/types stable), and no float goes nan/inf.
  (C) Liveness         - state actually evolves; how often it changes; whether the world
                         gets stuck in an absorbing state; the fraction of dead (no-op)
                         actions; state-visitation diversity.
  (D) Controllability  - THE headline. Planning through the model (best of P simulated
                         action sequences) reaches states that acting randomly does not.
                         A world an agent cannot steer is not a useful simulator. We report
                         the fraction of worlds whose state a planner can drive, and the
                         normalized planner-over-random improvement.

Deterministic + offline (numpy only, no Ollama, no GPU). We are explicit that these
measure intrinsic usability/robustness, NOT correctness against a ground truth (which does
not exist for synthesized worlds). save_results is called BEFORE the asserts.
"""

import json
import random
from pathlib import Path

import numpy as np

from common import save_results
from openworld.spec import from_spec
from openworld.sandbox import load_transition_code

ROOT = Path(__file__).resolve().parent.parent
RECIPES = ROOT / "recipes"
SECTORS = ["healthcare", "financial", "legal", "cybersecurity", "energy", "agentic"]

H = 20        # rollout / planning horizon
R = 64        # random-policy rollouts used to estimate the unplanned outcome
P = 128       # action sequences a planner simulates (best-of-P = the planned outcome)
CTRL_THRESH = 0.5  # planner must beat random mean by >= 0.5 random-std to count as control
SEED = 70


def _numeric_fields(state):
    """Int/float state fields (bool is excluded -- it is an int subclass in Python)."""
    return [k for k, v in state.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]


def _load(spec):
    """Return (step_fn, initial_state, actions) with the transition compiled ONCE."""
    w = from_spec(spec, allow_code=True)
    fn = load_transition_code(w.transition.code, getattr(w.transition, "func_name", "transition"))
    acts = [a for a in w.actions if ":" not in a] or list(w.actions)

    def step(s, a):
        return fn(dict(s), {"name": a, "params": {}, "agent": None})

    return step, dict(w.initial_state), acts


def _rollout(step, s0, seq):
    """Apply an action sequence; return the full trajectory (list of states)."""
    s = dict(s0)
    traj = [s]
    for a in seq:
        s = step(s, a)
        traj.append(s)
    return traj


def _finite(v):
    return not (isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))))


def _compat(t_old, t_new):
    """Type compatibility for schema stability: int<->float (numeric promotion) and
    None<->anything (a nullable field being populated) are NOT drift; genuine changes
    (e.g. int->str) and added/removed keys are."""
    if t_old is t_new:
        return True
    numeric = {int, float}
    if t_old in numeric and t_new in numeric:
        return True
    return t_old is type(None) or t_new is type(None)


def benchmark_world(spec, idx):
    rng = random.Random(SEED + idx)
    out = {"loaded": False, "crash": False}
    try:
        step, s0, acts = _load(spec)
        out["loaded"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)[:120]
        return out

    out["n_actions"] = len(acts)
    num = _numeric_fields(s0)
    out["n_numeric"] = len(num)

    # ---- (A) determinism: same action sequence twice => identical trajectory ----
    det_seq = [rng.choice(acts) for _ in range(H)]
    try:
        out["deterministic"] = _rollout(step, s0, det_seq) == _rollout(step, s0, det_seq)
    except Exception as exc:  # noqa: BLE001
        out["crash"] = True
        out["error"] = repr(exc)[:120]
        return out

    # ---- (B) safety/invariants + (C) liveness over a random rollout ----
    keys0, types0 = set(s0), {k: type(v) for k, v in s0.items()}
    schema_ok, finite_ok, changes, visited = True, True, 0, set()
    s = dict(s0)
    try:
        for _ in range(H):
            a = rng.choice(acts)
            ns = step(s, a)
            if set(ns) != keys0 or any(not _compat(types0.get(k), type(ns.get(k))) for k in keys0):
                schema_ok = False
            if not all(_finite(v) for v in ns.values()):
                finite_ok = False
            if ns != s:
                changes += 1
            visited.add(tuple(sorted((k, repr(v)) for k, v in ns.items())))
            s = ns
    except Exception as exc:  # noqa: BLE001
        out["crash"] = True
        out["error"] = repr(exc)[:120]
        return out
    out["schema_stable"] = schema_ok
    out["finite"] = finite_ok
    out["change_frac"] = changes / H
    out["distinct_frac"] = len(visited) / H

    # absorbing: from the final state, does ANY action change it?
    out["absorbing"] = all(step(s, a) == s for a in acts)
    # dead actions: an action that never changes state across a sampled set of contexts
    # (a short random walk), i.e. a no-op the agent can never use to affect the world.
    ctx, sc = [], dict(s0)
    for _ in range(H):
        ctx.append(dict(sc))
        sc = step(sc, rng.choice(acts))
    dead = 0
    for a in acts:
        if all(step(c, a) == c for c in ctx):
            dead += 1
    out["dead_action_frac"] = dead / len(acts) if acts else 1.0

    # ---- (D) controllability: planner (best-of-P) vs random-policy (mean-of-R) ----
    if not num or len(acts) < 2:
        out["controllable_vars"] = 0
        out["controllable"] = False
        out["best_ctrl_gap"] = 0.0
        out["ctrl_note"] = "no numeric fields" if not num else "single action (no choice)"
        return out

    def endpoints(n_seqs):
        ends = []
        for _ in range(n_seqs):
            seq = [rng.choice(acts) for _ in range(H)]
            ends.append(_rollout(step, s0, seq)[-1])
        return ends

    rand_ends = endpoints(R)      # unplanned outcomes
    plan_ends = endpoints(P)      # the pool a planner searches over (independent stream)

    ctrl_vars, gaps = 0, []
    for v in num:
        rvals = np.array([float(e[v]) for e in rand_ends], dtype=float)
        pvals = np.array([float(e[v]) for e in plan_ends], dtype=float)
        rmean, rstd = float(rvals.mean()), float(rvals.std())
        denom = rstd if rstd > 1e-9 else (abs(rmean) * 1e-3 + 1e-9)
        # planner can aim to maximize OR minimize; take the better normalized gap
        gap = max(float(pvals.max()) - rmean, rmean - float(pvals.min())) / denom
        gaps.append(gap)
        if gap >= CTRL_THRESH:
            ctrl_vars += 1
    out["controllable_vars"] = ctrl_vars
    out["controllable"] = ctrl_vars > 0
    out["best_ctrl_gap"] = round(max(gaps), 3) if gaps else 0.0
    return out


def bootstrap_median_ci(xs, iters=4000, seed=0):
    if not xs:
        return [None, None]
    rng = random.Random(seed)
    n = len(xs)
    meds = sorted(float(np.median([xs[rng.randrange(n)] for _ in range(n)])) for _ in range(iters))
    return [round(meds[int(0.025 * iters)], 3), round(meds[int(0.975 * iters)], 3)]


def main():
    per = []
    for sec in SECTORS:
        for f in sorted((RECIPES / sec).glob("*.json")):
            spec = json.loads(f.read_text())
            r = benchmark_world(spec, len(per))
            r.update({"sector": sec, "world": f.stem})
            per.append(r)

    ok = [r for r in per if r.get("loaded") and not r.get("crash")]
    n = len(per)
    steerable = [r for r in ok if r.get("controllable")]
    # worlds for which the controllability test is applicable (>=2 actions, >=1 numeric field)
    applicable = [r for r in ok if r.get("n_numeric", 0) > 0 and r.get("n_actions", 0) >= 2]
    ctrl_applicable = [r for r in applicable if r.get("controllable")]
    gaps = [r["best_ctrl_gap"] for r in applicable]

    results = {
        "task": "intrinsic world-model benchmark over the 100 Claude-Code-built recipes "
                "(no oracle exists for synthesized worlds; these measure usability, not fidelity)",
        "config": {"horizon": H, "random_rollouts": R, "planner_samples": P,
                   "control_threshold_std": CTRL_THRESH, "seed": SEED},
        "n_worlds": n,
        "n_loaded": sum(r.get("loaded", False) for r in per),
        "n_crash": sum(r.get("crash", False) for r in per),
        # (A) determinism
        "determinism_rate": round(sum(r.get("deterministic", False) for r in ok) / len(ok), 3) if ok else None,
        # (B) safety / invariants
        "schema_stable_rate": round(sum(r.get("schema_stable", False) for r in ok) / len(ok), 3) if ok else None,
        "finite_rate": round(sum(r.get("finite", False) for r in ok) / len(ok), 3) if ok else None,
        # (C) liveness
        "mean_change_frac": round(float(np.mean([r["change_frac"] for r in ok])), 3) if ok else None,
        "absorbing_rate": round(sum(r.get("absorbing", False) for r in ok) / len(ok), 3) if ok else None,
        "mean_dead_action_frac": round(float(np.mean([r["dead_action_frac"] for r in ok])), 3) if ok else None,
        "mean_distinct_frac": round(float(np.mean([r["distinct_frac"] for r in ok])), 3) if ok else None,
        # (D) controllability (headline)
        "n_control_applicable": len(applicable),
        "n_steerable": len(steerable),
        "steerable_rate": round(len(steerable) / n, 3) if n else None,
        "steerable_rate_applicable": round(len(ctrl_applicable) / len(applicable), 3) if applicable else None,
        "median_control_gap": round(float(np.median(gaps)), 3) if gaps else None,
        "control_gap_ci95": bootstrap_median_ci(gaps),
        "mean_controllable_vars": round(float(np.mean([r["controllable_vars"] for r in applicable])), 3) if applicable else None,
        "per_world": per,
    }
    save_results("e70_world_bench", results)

    # ---- self-checks (after save_results) ----
    assert results["n_loaded"] == n, f"some recipes failed to load: {results['n_loaded']}/{n}"
    # one recipe (budget_envelope) uses a builtin the sandbox forbids -- the same single
    # failure E68b reports; everything else must run clean.
    assert results["n_crash"] <= 1, f"{results['n_crash']} worlds crashed during rollout"
    assert results["determinism_rate"] == 1.0, \
        f"non-deterministic worlds: rate={results['determinism_rate']}"
    assert results["schema_stable_rate"] == 1.0, \
        f"schema drift detected (beyond numeric/nullable): rate={results['schema_stable_rate']}"
    assert results["finite_rate"] == 1.0, f"nan/inf produced: rate={results['finite_rate']}"
    # headline: most prototyped worlds are steerable, and planning beats random meaningfully
    assert results["steerable_rate_applicable"] >= 0.8, \
        f"too few steerable worlds: {results['steerable_rate_applicable']}"
    assert results["median_control_gap"] > CTRL_THRESH, \
        f"weak control signal: median gap {results['median_control_gap']}"
    print(f"[ok] E70: {n} worlds | det {results['determinism_rate']} | "
          f"schema {results['schema_stable_rate']} | finite {results['finite_rate']} | "
          f"steerable {results['n_steerable']}/{n} "
          f"({results['steerable_rate_applicable']} of applicable) | "
          f"median control gap {results['median_control_gap']} std "
          f"| absorbing {results['absorbing_rate']} | dead-actions {results['mean_dead_action_frac']}")


if __name__ == "__main__":
    main()
