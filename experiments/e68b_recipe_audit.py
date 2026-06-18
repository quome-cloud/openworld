"""E68b - Hardening the prototyping benchmark: behavioral correctness + timing distribution.

E68 reports that 100 Claude-Code-built worlds pass validate_spec (a structural gate). A
reviewer rightly asks: validated != correct, and a single median hides the spread. This
audit, run offline on the committed recipes + E68 timings, adds:

  (1) Behavioral correctness -- load each recipe via from_spec(allow_code=True), roll it
      out under random actions, and require it to step without error AND evolve state
      (a world that never changes is not a world). Reports the executable rate.
  (2) The full build-time distribution -- percentiles + standard deviation + a bootstrap
      95% CI on the median -- not just the median.

Deterministic + offline (no rebuilds, no GPU). save_results before asserts.
"""

import json
import random
from pathlib import Path
from statistics import mean, median, pstdev

from common import save_results
from openworld.spec import from_spec
from openworld.state import Action

ROOT = Path(__file__).resolve().parent.parent
RECIPES = ROOT / "recipes"
SECTORS = ["healthcare", "financial", "legal", "cybersecurity", "energy", "agentic"]


def is_executable(spec, steps=12, seed=0):
    """A recipe is behaviorally executable if it loads, steps without error, and the
    state actually changes over a random rollout."""
    try:
        w = from_spec(spec, allow_code=True)
        s = dict(w.initial_state)
        acts = [a for a in w.actions if ":" not in a] or list(w.actions)
        if not acts:
            return False
        rng = random.Random(seed)
        changed = False
        for _ in range(steps):
            ns = dict(w.transition.step(s, Action(rng.choice(acts))))
            if ns != s:
                changed = True
            s = ns
        return changed
    except Exception:
        return False


def bootstrap_median_ci(xs, iters=4000, seed=0):
    rng = random.Random(seed)
    n = len(xs)
    meds = sorted(median([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(iters))
    return round(meds[int(0.025 * iters)], 2), round(meds[int(0.975 * iters)], 2)


def pctl(xs, p):
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(p / 100 * len(s)))], 2)


def main():
    # (1) behavioral audit over committed recipes
    per = []
    for sec in SECTORS:
        for f in sorted((RECIPES / sec).glob("*.json")):
            spec = json.loads(f.read_text())
            per.append({"sector": sec, "world": f.stem, "executable": is_executable(spec)})
    n_exec = sum(r["executable"] for r in per)

    # (2) timing distribution from the committed E68 build results
    e68 = json.loads((ROOT / "experiments" / "results" / "e68_prototyping_latency.json").read_text())
    mins = [w["minutes"] for w in e68["worlds"] if w.get("validated")]
    lo, hi = bootstrap_median_ci(mins)

    results = {
        "task": "audit of the 100 Claude-Code-built world recipes (behavioral correctness + timing spread)",
        "n_recipes": len(per), "n_executable": n_exec,
        "behavioral_executable_rate": round(n_exec / len(per), 3) if per else None,
        "executable_note": "loads via from_spec(allow_code=True), steps without error, and "
                           "state evolves under a random rollout -- a stronger check than "
                           "validate_spec's structural gate",
        "timing_distribution_minutes": {
            "n": len(mins), "mean": round(mean(mins), 2), "median": round(median(mins), 2),
            "std": round(pstdev(mins), 2), "p10": pctl(mins, 10), "p25": pctl(mins, 25),
            "p75": pctl(mins, 75), "p90": pctl(mins, 90), "p99": pctl(mins, 99),
            "median_ci95": [lo, hi]},
        "not_executable": [r for r in per if not r["executable"]],
    }
    save_results("e68b_recipe_audit", results)   # BEFORE asserts

    assert n_exec >= 0.9 * len(per), f"only {n_exec}/{len(per)} recipes behaviorally executable"
    print(f"[ok] behavioral correctness: {n_exec}/{len(per)} recipes execute + evolve state "
          f"({results['behavioral_executable_rate']:.0%})")
    d = results["timing_distribution_minutes"]
    print(f"[ok] build-time minutes: median {d['median']} (95% CI {d['median_ci95']}), "
          f"p90 {d['p90']}, std {d['std']}")
    if results["not_executable"]:
        print("  non-executable:", [r["world"] for r in results["not_executable"]])


if __name__ == "__main__":
    main()
