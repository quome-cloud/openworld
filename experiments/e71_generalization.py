"""E71 - Cross-domain generalization on the 100-world composite ("can a lawyer become a
physician?").

The 100 prototyped recipes (E68) form a composite of heterogeneous worlds: a legal world's
actions mean nothing in a sepsis world. That heterogeneity is the point. We ask how three
kinds of agent generalize ACROSS the sectors:

  - PLANNER (model-based, spec-agnostic): plans through each world's verified model by
    simulating short rollouts. Needs no training and works on ANY world -- the generalist
    by construction. It defines the competence ceiling (1.0).
  - SPECIALIST (model-free, tabular Q over an abstract state-bin x action-index interface):
    a learner trained on one sector's worlds. We build the 6x6 transfer matrix -- sector-X
    policy evaluated on sector-Y worlds -- and adaptation curves (how fast a sector-X
    specialist reaches competence on a foreign sector). Expectation: strong on the diagonal,
    weak off it -- the lawyer who cannot practice medicine.
  - LLM AGENT (a local model reading each world's rules and acting): a "real agent" proxy
    that, like a person, reads the spec and acts across domains without per-world training.

Competence is normalized per world to [random -> planner] = [0 -> 1], so it aggregates
across worlds with different scales/objectives. Restricted to the worlds E70 found
steerable (a control signal exists). Panels 1-2 (planner, specialists) are deterministic
and offline; panel 3 (LLM agent) needs Ollama and is skipped gracefully if absent.
save_results is called BEFORE the asserts.
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
RESULTS = ROOT / "experiments" / "results"
SECTORS = ["healthcare", "financial", "legal", "cybersecurity", "energy", "agentic"]

H = 14            # episode horizon
R_RAND = 24       # random-policy episodes (baseline + range estimation)
PLAN_K = 20       # action sequences the planner simulates per decision
PLAN_H = 6        # planner lookahead
Q_EPISODES = 250  # model-free training episodes per world
N_BINS = 4        # state discretization for the abstract policy interface
ADAPT_STEPS = [0, 10, 25, 50, 100]  # adaptation-curve checkpoints (episodes)
# LLM "real agent" proxies: a small and a larger local model (does a stronger agent
# generalize better across domains?). Bigger/cloud models can be appended.
LLM_MODELS = ["qwen2.5:7b", "qwen2.5-coder:32b"]
LLM_WORLDS_PER_SECTOR = 2
LLM_H = 8
SEED = 71

MIN_WORDS = ("minimi", "reduce", "avoid", "lower", "prevent", "decreas", "fewer",
             "less", "limit", "shrink", "cut")
MAX_WORDS = ("maximi", "increas", " more", "improve", "higher", "grow", "raise",
             "optimi", "boost", "expand")


def numeric_fields(state):
    return [k for k, v in state.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)]


def load_world(spec):
    w = from_spec(spec, allow_code=True)
    fn = load_transition_code(w.transition.code, getattr(w.transition, "func_name", "transition"))
    acts = [a for a in w.actions if ":" not in a] or list(w.actions)

    def step(s, a):
        return fn(dict(s), {"name": a, "params": {}, "agent": None})

    return step, dict(w.initial_state), acts, spec


def parse_task(spec, s0):
    """Pick (target_var, direction) from the declared objectives; fall back to the most
    variable numeric field (maximize) if no objective cleanly names one."""
    nums = numeric_fields(s0)
    for obj in spec.get("objectives", []) or []:
        goal = str(obj.get("goal", "")).lower()
        direction = (-1 if any(w in goal for w in MIN_WORDS)
                     else 1 if any(w in goal for w in MAX_WORDS) else 0)
        if not direction:
            continue
        cand = [f for f in nums if f.replace("_", " ") in goal or f in goal]
        if cand:
            return max(cand, key=len), direction, True
    return None, 1, False


def most_variable_field(step, s0, actions, seed):
    """Fallback task: the numeric field that moves the most under a random rollout."""
    nums = numeric_fields(s0)
    if not nums:
        return None
    rng = random.Random(seed)
    lo = {f: float(s0[f]) for f in nums}
    hi = dict(lo)
    s = dict(s0)
    for _ in range(H * 2):
        s = step(s, rng.choice(actions))
        for f in nums:
            v = float(s[f]); lo[f] = min(lo[f], v); hi[f] = max(hi[f], v)
    spreads = {f: hi[f] - lo[f] for f in nums}
    best = max(spreads, key=spreads.get)
    return best if spreads[best] > 0 else None


def rollout_return(step, s0, actions, target, direction, rng):
    s = dict(s0)
    for _ in range(H):
        s = step(s, rng.choice(actions))
    return direction * (float(s[target]) - float(s0[target]))


def random_stats(step, s0, actions, target, direction, seed):
    """Random-policy return distribution + the target's observed range (for binning)."""
    rng = random.Random(seed)
    rets, lo, hi = [], float(s0[target]), float(s0[target])
    for _ in range(R_RAND):
        s = dict(s0)
        for _ in range(H):
            s = step(s, rng.choice(actions))
            v = float(s[target]); lo, hi = min(lo, v), max(hi, v)
        rets.append(direction * (float(s[target]) - float(s0[target])))
    return float(np.mean(rets)), (lo, hi)


def planner_return(step, s0, actions, target, direction, seed):
    """Closed-loop random-shooting: each step, simulate PLAN_K short sequences and take the
    first action of the best one (receding horizon)."""
    rng = random.Random(seed)
    s = dict(s0)
    for _ in range(H):
        best_a, best_v = actions[0], -1e18
        for a in actions:
            tot = 0.0
            for _ in range(PLAN_K // len(actions) + 1):
                sim = step(s, a)
                for _ in range(PLAN_H - 1):
                    sim = step(sim, rng.choice(actions))
                tot += direction * float(sim[target])
            avg = tot / (PLAN_K // len(actions) + 1)
            if avg > best_v:
                best_v, best_a = avg, a
        s = step(s, best_a)
    return direction * (float(s[target]) - float(s0[target]))


def _bin(v, rng_lohi):
    lo, hi = rng_lohi
    if hi <= lo:
        return 0
    return int(min(N_BINS - 1, max(0, (v - lo) / (hi - lo) * N_BINS)))


def train_q(step, s0, actions, target, direction, rng_lohi, seed, episodes, q0=None, max_a=None):
    """Tabular Q over (target-bin x action-index). Returns a [N_BINS, max_a] table; action
    index is the abstract, transferable interface (its meaning differs per world)."""
    max_a = max_a or len(actions)
    q = np.zeros((N_BINS, max_a)) if q0 is None else q0.copy()
    rng = random.Random(seed)
    alpha, gamma = 0.3, 0.9
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / max(1, episodes * 0.7))
        s = dict(s0)
        b = _bin(float(s[target]), rng_lohi)
        for _ in range(H):
            valid = list(range(len(actions)))
            ai = (rng.choice(valid) if rng.random() < eps
                  else max(valid, key=lambda i: q[b, i]))
            ns = step(s, actions[ai])
            r = direction * (float(ns[target]) - float(s[target]))
            nb = _bin(float(ns[target]), rng_lohi)
            q[b, ai] += alpha * (r + gamma * q[nb].max() - q[b, ai])
            s, b = ns, nb
    return q


def eval_q(step, s0, actions, target, direction, rng_lohi, q):
    s = dict(s0)
    for _ in range(H):
        b = _bin(float(s[target]), rng_lohi)
        ai = int(np.argmax(q[b, :len(actions)]))
        s = step(s, actions[ai])
    return direction * (float(s[target]) - float(s0[target]))


def competence(g, g_rand, g_plan):
    if g_plan - g_rand < 1e-9:
        return None
    return float(np.clip((g - g_rand) / (g_plan - g_rand), -0.25, 1.25))


def main():
    e70 = json.loads((RESULTS / "e70_world_bench.json").read_text())
    steer = {(r["sector"], r["world"]) for r in e70["per_world"] if r.get("controllable")}

    # ---- load steerable worlds with a well-defined task ----
    worlds = {s: [] for s in SECTORS}
    for sec in SECTORS:
        for f in sorted((RECIPES / sec).glob("*.json")):
            if (sec, f.stem) not in steer:
                continue
            spec = json.loads(f.read_text())
            step, s0, acts, _ = load_world(spec)
            if len(acts) < 2:
                continue
            target, direction, named = parse_task(spec, s0)
            if target is None:
                target, direction, named = most_variable_field(step, s0, acts, SEED), 1, False
            if target is None:
                continue
            idx = sum(len(v) for v in worlds.values())
            g_rand, lohi = random_stats(step, s0, acts, target, direction, SEED + idx)
            g_plan = planner_return(step, s0, acts, target, direction, SEED + idx)
            if g_plan - g_rand < 1e-9:
                continue
            worlds[sec].append({
                "world": f.stem, "spec": spec, "step": step, "s0": s0, "acts": acts,
                "target": target, "dir": direction, "lohi": lohi,
                "g_rand": g_rand, "g_plan": g_plan,
                "n_actions": len(acts), "objective_named": named})
    flat = [w for s in SECTORS for w in worlds[s]]
    max_a = max(w["n_actions"] for w in flat)
    n_used = len(flat)

    # ---- train one specialist Q per world; record its competence on its OWN world ----
    for i, w in enumerate(flat):
        w["q"] = train_q(w["step"], w["s0"], w["acts"], w["target"], w["dir"], w["lohi"],
                         SEED + i, Q_EPISODES, max_a=max_a)
        w["comp_self"] = competence(
            eval_q(w["step"], w["s0"], w["acts"], w["target"], w["dir"], w["lohi"], w["q"]),
            w["g_rand"], w["g_plan"])

    # ---- transfer: a world-W specialist applied zero-shot to OTHER worlds, bucketed by
    #      whether the target world is in the same sector or a different one. If learned
    #      expertise carried sector structure, same-sector would beat cross-sector. ----
    rngp = random.Random(SEED)
    same_c, cross_c = [], []
    by_sec = {s: worlds[s] for s in SECTORS}
    for w in flat:
        sec = next(s for s in SECTORS if w in by_sec[s])
        same_targets = [u for u in by_sec[sec] if u is not w]
        cross_targets = [u for s in SECTORS if s != sec for u in by_sec[s]]
        for u in rngp.sample(same_targets, min(3, len(same_targets))):
            c = competence(eval_q(u["step"], u["s0"], u["acts"], u["target"], u["dir"],
                                  u["lohi"], w["q"]), u["g_rand"], u["g_plan"])
            if c is not None:
                same_c.append(c)
        for u in rngp.sample(cross_targets, min(3, len(cross_targets))):
            c = competence(eval_q(u["step"], u["s0"], u["acts"], u["target"], u["dir"],
                                  u["lohi"], w["q"]), u["g_rand"], u["g_plan"])
            if c is not None:
                cross_c.append(c)
    transfer = {
        "same_sector_mean": round(float(np.mean(same_c)), 3) if same_c else None,
        "cross_sector_mean": round(float(np.mean(cross_c)), 3) if cross_c else None,
        "n_same": len(same_c), "n_cross": len(cross_c),
    }

    # ---- learning curve: competence on a world's OWN task vs episodes of experience
    #      (model-free pays a per-world data cost; planner/LLM pay none). Tracks the SAME
    #      worlds across checkpoints, averaged. ----
    lc_worlds = rngp.sample(flat, min(24, len(flat)))
    learning_curve = {"episodes": ADAPT_STEPS, "competence": []}
    for nep in ADAPT_STEPS:
        cs = []
        for w in lc_worlds:
            q = (train_q(w["step"], w["s0"], w["acts"], w["target"], w["dir"], w["lohi"],
                         SEED + 13, nep, max_a=max_a) if nep else np.zeros((N_BINS, max_a)))
            c = competence(eval_q(w["step"], w["s0"], w["acts"], w["target"], w["dir"],
                                  w["lohi"], q), w["g_rand"], w["g_plan"])
            if c is not None:
                cs.append(c)
        learning_curve["competence"].append(round(float(np.mean(cs)), 3) if cs else None)

    # ---- panel 3: LLM "real agent(s)" read each world's rules + state and act zero-shot ----
    def run_llm_agent(model):
        from openworld.llm import OllamaLLM
        llm = OllamaLLM(model=model, options={"num_ctx": 4096}, timeout=180, temperature=0.2)
        llm.ask("Reply OK.")
        by_sector = {}
        for sec in SECTORS:
            comps = []
            for w in worlds[sec][:LLM_WORLDS_PER_SECTOR]:
                rules = "; ".join(w["spec"].get("rules", []) or [])[:600]
                s = dict(w["s0"])
                dirw = "increase" if w["dir"] > 0 else "decrease"
                for _ in range(LLM_H):
                    prompt = (f"World: {w['spec'].get('description','')[:200]}\n"
                              f"Rules: {rules}\nActions: {w['acts']}\n"
                              f"State: {json.dumps({k: s[k] for k in list(s)[:14]})}\n"
                              f"Goal: {dirw} '{w['target']}'. Reply with ONLY one action name.")
                    try:
                        ans = llm.ask(prompt).strip().split()[0].strip(".,'\"").lower()
                        a = next((x for x in w["acts"] if x.lower() == ans),
                                 next((x for x in w["acts"] if x.lower() in ans or ans in x.lower()),
                                      None))
                    except Exception:  # noqa: BLE001
                        a = None
                    s = w["step"](s, a if a else random.Random(0).choice(w["acts"]))
                c = competence(w["dir"] * (float(s[w["target"]]) - float(w["s0"][w["target"]])),
                               w["g_rand"], w["g_plan"])
                if c is not None:
                    comps.append(c)
            by_sector[sec] = round(float(np.mean(comps)), 3) if comps else None
        vals = [v for v in by_sector.values() if v is not None]
        return {"by_sector": by_sector, "mean": round(float(np.mean(vals)), 3) if vals else None}

    llm_agents = {}
    for model in LLM_MODELS:
        try:
            llm_agents[model] = run_llm_agent(model)
        except Exception as exc:  # noqa: BLE001
            llm_agents[model] = {"by_sector": {}, "mean": None, "note": f"skipped ({type(exc).__name__})"}

    n_named = sum(w["objective_named"] for w in flat)
    results = {
        "task": "cross-domain generalization across the 100-world composite: where does "
                "competence on a new world come from -- the model, learned weights, or "
                "reading the spec?",
        "config": {"horizon": H, "planner_k": PLAN_K, "planner_h": PLAN_H,
                   "q_episodes": Q_EPISODES, "n_bins": N_BINS, "seed": SEED},
        "n_worlds_used": n_used,
        "n_by_sector": {s: len(worlds[s]) for s in SECTORS},
        "n_objective_named": n_named,
        "note": "competence normalized per world to [random=0, planner=1]; the model-based "
                "planner is 1.0 everywhere by construction (generalist via the model).",
        # model-free: expert on its own world, but does the expertise transfer?
        "specialist_self_competence": round(float(np.mean(
            [w["comp_self"] for w in flat if w["comp_self"] is not None])), 3),
        "transfer": transfer,
        "learning_curve": learning_curve,
        # zero-shot competence on unseen worlds, by agent kind
        "zero_shot": {
            "planner_model_based": 1.0,
            "llm_read_spec": {m: a["mean"] for m, a in llm_agents.items()},
            "model_free_foreign_weights": transfer["cross_sector_mean"],
            "random": 0.0,
        },
        "llm_agents": llm_agents,
    }
    save_results("e71_generalization", results)

    # ---- self-checks (after save_results) ----
    assert n_used >= 60, f"too few usable worlds: {n_used}"
    # model-free expertise is world-specific: competent on its OWN world, not on others.
    assert results["specialist_self_competence"] > transfer["cross_sector_mean"], \
        "own-world competence should exceed zero-shot transfer to other worlds"
    # learning curve rises with experience (model-free pays a per-world data cost).
    lc = [c for c in learning_curve["competence"] if c is not None]
    assert lc[-1] > lc[0], f"learning curve did not rise: {lc}"
    # reading the spec (LLM) should beat applying foreign learned weights zero-shot.
    best_llm = max([a["mean"] for a in llm_agents.values() if a["mean"] is not None], default=None)
    if best_llm is not None:
        assert best_llm > transfer["cross_sector_mean"], \
            f"LLM zero-shot ({best_llm}) should beat foreign weights ({transfer['cross_sector_mean']})"
    print(f"[ok] E71: {n_used} worlds ({n_named} objective-named) | "
          f"model-free own-world {results['specialist_self_competence']} vs transfer "
          f"same {transfer['same_sector_mean']} / cross {transfer['cross_sector_mean']} | "
          f"learning curve {learning_curve['competence']} over {ADAPT_STEPS} eps")
    print(f"  zero-shot competence: planner 1.0 | LLM(read spec) "
          f"{ {m: a['mean'] for m, a in llm_agents.items()} } | "
          f"foreign weights {transfer['cross_sector_mean']} | random 0.0")


if __name__ == "__main__":
    main()
