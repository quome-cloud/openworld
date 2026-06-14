"""E49 - Path integrals over composite-world learning trajectories.

E46 gave a many-worlds store that searches the space of worlds for a problem.
E49 adds the missing piece: how should an agent TRAVERSE that space - in what
order to master worlds/skills - to solve a NEW, out-of-distribution problem as
directly as possible? We cast this as a path integral over learning
trajectories: each trajectory is an ordering of skills to learn, weighted by
exp(-beta * action) where the action is the total learning cost; the sum is
dominated by the least-action path - the most direct curriculum. It is computed
without enumerating the (combinatorially many) trajectories, as a semiring DP
over capability-states - the same Semiring abstraction as E46, with the value
semiring swapped (tropical for least action, log for the full integral).

Skills compose cheaply once their prerequisites are mastered and are expensive to
learn from scratch; the OOD target is a NOVEL composition. So the least-action
path reuses mastered sub-worlds (compositional transfer) instead of learning the
target blind. We build the agent's trajectory FROM ITS SPEC (initial
capabilities) - a senior SWE, a director, and a CEO start in different places and
get different optimal curricula to the same goal.

Five results, deterministic/offline:
  1. agent spec -> trajectory: least-action curriculum + cost per role.
  2. least action beats unplanned baselines (random / eager / greedy).
  3. compositional transfer: path cost << learning the goal from scratch.
  4. path-integral marginals + a beta sweep: which worlds matter, and how the
     distribution concentrates on the least-action path as beta grows.
  5. infinitely-many trajectories, tractably: trajectory count vs DP states.
"""

import math
import random

from openworld import Skill, TrajectorySpace
from openworld.pathintegral import COUNTING, LOG

from common import save_results

# A skill library. Primitives (no prereqs) are cheap; composites compose cheaply
# once prereqs are mastered but are dear from scratch; the goal is a novel
# composition (the OOD problem: turn around a stalling division - cf. E48).
LIBRARY = [
    Skill("market_analysis", compose_cost=2, scratch_cost=2),
    Skill("team_management", compose_cost=2, scratch_cost=2),
    Skill("financial_modeling", compose_cost=2, scratch_cost=2),
    Skill("data_perception", compose_cost=2, scratch_cost=2),
    Skill("revenue_dynamics", ("market_analysis", "financial_modeling"), 2, 25),
    Skill("budget_allocation", ("market_analysis", "financial_modeling"), 2, 25),
    Skill("division_growth", ("revenue_dynamics", "team_management"), 2, 35),
    Skill("portfolio_optimization", ("budget_allocation", "revenue_dynamics"), 2, 40),
    Skill("turnaround_division",
          ("division_growth", "portfolio_optimization", "data_perception"), 3, 90),
]
GOAL = "turnaround_division"

# agent specs = starting capabilities (cf. E48 roles)
AGENTS = {
    "senior_swe": ["data_perception", "financial_modeling"],
    "director": ["team_management", "revenue_dynamics", "market_analysis"],
    "ceo": ["budget_allocation", "portfolio_optimization", "market_analysis"],
}


def cost_fn(skills, name, learned):
    s = skills[name]
    return s.compose_cost if set(s.prereqs) <= learned else s.scratch_cost


def simulate(skills, universe, initial, order):
    """Learn skills in `order`, paying compose-cost if prereqs are already
    learned else scratch-cost; stop once the goal is learned. Returns total."""
    learned, total = set(initial), 0.0
    for n in order:
        if n in learned:
            continue
        total += cost_fn(skills, n, learned)
        learned.add(n)
        if n == GOAL:
            break
    return total


def greedy(skills, universe, initial):
    learned, total = set(initial), 0.0
    while GOAL not in learned:
        opts = [n for n in universe if n not in learned]
        n = min(opts, key=lambda m: cost_fn(skills, m, learned))
        total += cost_fn(skills, n, learned)
        learned.add(n)
    return total


def baselines(sp, initial):
    skills, uni = sp.skills, sorted(sp.universe)
    rng = random.Random(49)
    rand = sum(simulate(skills, uni, initial,
                        rng.sample(uni, len(uni))) for _ in range(300)) / 300
    # eager: rush the most-dependent skills first (goal, then composites) -> scratch
    depth = {n: len(skills[n].prereqs) for n in uni}
    eager_order = sorted(uni, key=lambda n: -depth[n])
    eager = simulate(skills, uni, initial, eager_order)
    return {"random_mean": round(rand, 1), "eager": round(eager, 1),
            "greedy": round(greedy(skills, uni, initial), 1)}


def main():
    # 1. agent spec -> trajectory
    per_agent = {}
    for role, init in AGENTS.items():
        sp = TrajectorySpace(LIBRARY, initial=init, goal=GOAL)
        steps, cost = sp.least_action_path()
        per_agent[role] = {"initial": init, "curriculum": steps,
                           "least_action_cost": cost,
                           "baselines": baselines(sp, frozenset(init))}

    # 2-5 analyses for a representative cold-start agent (the senior SWE)
    sp = TrajectorySpace(LIBRARY, initial=AGENTS["senior_swe"], goal=GOAL)
    steps, la_cost = sp.least_action_path()
    scratch = sp.goal_cost_from_scratch()
    n_traj = sp.count_trajectories()
    n_states = len(sp.reachable())
    betas = [0.2, 0.5, 1.0, 3.0, 10.0]
    marg_by_beta = {b: sp.node_marginals(beta=b) for b in betas}
    free_energy = {b: round(-sp.partition(LOG, beta=b) / b, 2) for b in betas}

    results = {
        "goal": GOAL, "library_size": len(LIBRARY),
        "per_agent": per_agent,
        "transfer": {"least_action_cost": la_cost, "from_scratch": scratch,
                     "speedup": round(scratch / la_cost, 1)},
        "tractability": {"n_trajectories": n_traj, "n_dp_states": n_states},
        "marginals_by_beta": {str(b): marg_by_beta[b] for b in betas},
        "free_energy_by_beta": free_energy,
        "betas": betas,
    }
    save_results("e49_path_integral", results)

    print("E49 - path integrals over learning trajectories "
          f"(library {len(LIBRARY)}, goal '{GOAL}')\n")
    print("1. agent spec -> least-action curriculum:")
    for role, d in per_agent.items():
        print(f"   {role:<11} cost {d['least_action_cost']:.0f} "
              f"(baselines: greedy {d['baselines']['greedy']:.0f}, "
              f"random {d['baselines']['random_mean']:.0f}, "
              f"eager {d['baselines']['eager']:.0f})  ->  "
              f"{' -> '.join(d['curriculum'])}")
    print(f"\n3. transfer: least-action {la_cost:.0f} vs from-scratch {scratch:.0f} "
          f"({results['transfer']['speedup']:.1f}x cheaper)")
    print(f"4. marginals (beta=3): " + ", ".join(
        f"{n} {marg_by_beta[3.0][n]:.2f}" for n in sorted(
            marg_by_beta[3.0], key=lambda k: -marg_by_beta[3.0][k])[:5]))
    print(f"   free energy by beta: {free_energy}")
    print(f"5. tractability: {n_traj:,} trajectories summed over {n_states} DP states")

    # --- self-checks ---
    for role, d in per_agent.items():
        b = d["baselines"]
        assert d["least_action_cost"] <= b["greedy"] + 1e-9, f"{role}: not optimal vs greedy"
        assert d["least_action_cost"] < b["random_mean"], f"{role}: should beat random"
        assert d["least_action_cost"] < b["eager"], f"{role}: should beat eager"
    assert la_cost < scratch, "compositional transfer should beat from-scratch"
    # marginals concentrate on the least-action path as beta grows
    spread_lo = sum(1 for v in marg_by_beta[0.2].values() if v > 0.05)
    spread_hi = sum(1 for v in marg_by_beta[10.0].values() if v > 0.05)
    assert spread_hi <= spread_lo, "higher beta should concentrate the path distribution"
    # the goal and its prereqs are on every optimal path -> marginal ~1 at high beta
    assert marg_by_beta[10.0][GOAL] > 0.95, "goal must be on the path"
    assert n_traj > n_states, "there are more trajectories than DP states (summed implicitly)"
    print("\nall path-integral checks pass.")


if __name__ == "__main__":
    main()
