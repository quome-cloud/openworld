"""E73 (data + constitution stage) - Traverse a domain's worlds, distill rule-grounded
expert decisions into a training set, and emit the domain constitution.

The vision: an LLM traverses many worlds of one domain, parses their rules, and comes out
with updated weights that generalize the domain (a "superphysician"), with the domain's
morality (its objectives) emerging as an emitted constitution. We evaluate this the way
scikit-learn would: WORLDS are the data points, and we hold out whole worlds for testing,
so "generalization" means competence on worlds never seen during fine-tuning.

This stage is fully offline and produces the artifacts the GPU fine-tune (e73_finetune) and
eval (e73_eval) consume:
  - a world-level train/test split of the domain;
  - an SFT dataset: for each TRAIN world, (rules + state + actions + goal) -> the
    model-based planner's objective-optimal action (behavior cloning of a planner that
    USES each verified world model);
  - the emitted constitution: the domain's objectives, aggregated across train worlds, the
    morality the traversal must honor (held out from the agent at train time);
  - per-test-world planner/random references for competence normalization at eval time.

Deterministic, numpy-only, no GPU, no LLM. save_results before asserts.
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
ART = ROOT / "experiments" / "results" / "e73_artifacts"

DOMAIN = "healthcare"
TEST_FRACTION = 0.30          # hold out ~30% of worlds for testing (sklearn-style)
H = 14                        # episode / planning horizon
PLAN_K = 20                   # planner action-sequence simulations per decision
PLAN_H = 6                    # planner lookahead
R_RAND = 24                   # random-policy reference episodes
STATES_PER_WORLD = 60         # SFT examples generated per train world
SEED = 73

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

    return step, dict(w.initial_state), acts


def parse_task(spec, s0):
    nums = numeric_fields(s0)
    for obj in spec.get("objectives", []) or []:
        goal = str(obj.get("goal", "")).lower()
        direction = (-1 if any(w in goal for w in MIN_WORDS)
                     else 1 if any(w in goal for w in MAX_WORDS) else 0)
        if not direction:
            continue
        cand = [f for f in nums if f.replace("_", " ") in goal or f in goal]
        if cand:
            return max(cand, key=len), direction
    return None, 1


def most_variable_field(step, s0, acts, seed):
    nums = numeric_fields(s0)
    if not nums:
        return None
    rng = random.Random(seed)
    lo = {f: float(s0[f]) for f in nums}
    hi = dict(lo)
    s = dict(s0)
    for _ in range(H * 2):
        s = step(s, rng.choice(acts))
        for f in nums:
            v = float(s[f]); lo[f] = min(lo[f], v); hi[f] = max(hi[f], v)
    spreads = {f: hi[f] - lo[f] for f in nums}
    best = max(spreads, key=spreads.get)
    return best if spreads[best] > 0 else None


def planner_action(step, s, acts, target, direction, rng):
    """The objective-optimal action at state s (the expert label): simulate short rollouts
    per candidate action and pick the one with the best expected target."""
    best_a, best_v = acts[0], -1e18
    for a in acts:
        tot, reps = 0.0, PLAN_K // len(acts) + 1
        for _ in range(reps):
            sim = step(s, a)
            for _ in range(PLAN_H - 1):
                sim = step(sim, rng.choice(acts))
            tot += direction * float(sim[target])
        if tot / reps > best_v:
            best_v, best_a = tot / reps, a
    return best_a


def planner_return(step, s0, acts, target, direction, seed):
    rng = random.Random(seed)
    s = dict(s0)
    for _ in range(H):
        s = step(s, planner_action(step, s, acts, target, direction, rng))
    return direction * (float(s[target]) - float(s0[target]))


def random_return(step, s0, acts, target, direction, seed):
    rng = random.Random(seed)
    rets = []
    for _ in range(R_RAND):
        s = dict(s0)
        for _ in range(H):
            s = step(s, rng.choice(acts))
        rets.append(direction * (float(s[target]) - float(s0[target])))
    return float(np.mean(rets))


def visited_states(step, s0, acts, n, seed):
    rng = random.Random(seed)
    out, s = [], dict(s0)
    for _ in range(n):
        for _ in range(rng.randint(1, H)):
            s = step(s, rng.choice(acts))
        out.append(dict(s))
    return out


def prep_world(sec, name, spec, idx):
    step, s0, acts = load_world(spec)
    if len(acts) < 2:
        return None
    target, direction = parse_task(spec, s0)
    named = target is not None
    if target is None:
        target = most_variable_field(step, s0, acts, SEED)
        direction = 1
    if target is None:
        return None
    return {"sector": sec, "world": name, "spec": spec, "step": step, "s0": s0,
            "acts": acts, "target": target, "dir": direction, "objective_named": named,
            "idx": idx}


def make_sft(world):
    """Behavior-clone the planner: (rules+state+actions+goal) -> objective-optimal action."""
    rng = random.Random(SEED + world["idx"])
    rules = "; ".join(world["spec"].get("rules", []) or [])[:700]
    desc = world["spec"].get("description", "")[:200]
    dirw = "increase" if world["dir"] > 0 else "decrease"
    rows = []
    for st in visited_states(world["step"], world["s0"], world["acts"], STATES_PER_WORLD,
                             SEED + world["idx"]):
        a = planner_action(world["step"], st, world["acts"], world["target"], world["dir"], rng)
        nums = numeric_fields(st)
        prompt = (f"You operate a {DOMAIN} world model.\n"
                  f"World: {desc}\nRules: {rules}\n"
                  f"State: {json.dumps({k: st[k] for k in nums})}\n"
                  f"Actions: {world['acts']}\n"
                  f"Goal: {dirw} '{world['target']}'. Reply with ONLY the single best action.")
        rows.append({"prompt": prompt, "completion": a,
                     "world": world["world"], "sector": world["sector"]})
    return rows


def main():
    ART.mkdir(parents=True, exist_ok=True)
    specs = [(DOMAIN, f.stem, json.loads(f.read_text()))
             for f in sorted((RECIPES / DOMAIN).glob("*.json"))]

    worlds = []
    for i, (sec, name, spec) in enumerate(specs):
        w = prep_world(sec, name, spec, i)
        if w:
            worlds.append(w)

    # ---- world-level train/test split (sklearn-style: hold out whole worlds) ----
    rng = random.Random(SEED)
    order = list(range(len(worlds)))
    rng.shuffle(order)
    n_test = max(1, round(len(worlds) * TEST_FRACTION))
    test_idx = set(order[:n_test])
    train = [worlds[i] for i in order[n_test:]]
    test = [worlds[i] for i in order[:n_test]]

    # ---- SFT data from TRAIN worlds only ----
    sft = []
    for w in train:
        sft.extend(make_sft(w))
    (ART / "sft_train.jsonl").write_text("\n".join(json.dumps(r) for r in sft) + "\n")

    # ---- emitted constitution: aggregate the TRAIN domain's objectives (the morality the
    #      agent must honor -- held out from the agent at train time) ----
    bare = {"maximize", "minimize", "max", "min", "increase", "decrease", "optimize"}
    seen, principles = set(), []
    for w in train:
        for obj in w["spec"].get("objectives", []) or []:
            goal = str(obj.get("goal", "")).strip()
            key = goal.lower()
            # drop degenerate goals: empty, a bare direction word, or fewer than two tokens
            if not goal or key in bare or len(goal.split()) < 2 or key in seen:
                continue
            seen.add(key)
            principles.append({"name": obj.get("name", ""), "goal": goal,
                               "source_world": w["world"]})
    constitution = {"domain": DOMAIN, "n_source_worlds": len(train),
                    "n_principles": len(principles), "principles": principles}
    (ART / "constitution.json").write_text(json.dumps(constitution, indent=2))

    # ---- per-test-world references for competence normalization at eval ----
    test_manifest = []
    for w in test:
        g_rand = random_return(w["step"], w["s0"], w["acts"], w["target"], w["dir"], SEED + w["idx"])
        g_plan = planner_return(w["step"], w["s0"], w["acts"], w["target"], w["dir"], SEED + w["idx"])
        test_manifest.append({
            "world": w["world"], "sector": w["sector"], "target": w["target"],
            "dir": w["dir"], "objective_named": w["objective_named"],
            "actions": w["acts"], "g_random": round(g_rand, 4), "g_planner": round(g_plan, 4),
            "controllable": bool(g_plan - g_rand > 1e-9),
            "rules": "; ".join(w["spec"].get("rules", []) or [])[:700],
            "description": w["spec"].get("description", "")[:200],
            "initial_state": {k: w["s0"][k] for k in numeric_fields(w["s0"])}})
    (ART / "test_manifest.json").write_text(json.dumps(test_manifest, indent=2))

    results = {
        "task": "E73 data + constitution stage: world-level train/test split, planner-labeled "
                "SFT data, and the emitted domain constitution for the de-risk run",
        "domain": DOMAIN,
        "n_worlds": len(worlds),
        "n_train_worlds": len(train),
        "n_test_worlds": len(test),
        "train_worlds": [w["world"] for w in train],
        "test_worlds": [w["world"] for w in test],
        "n_sft_examples": len(sft),
        "n_objective_named_train": sum(w["objective_named"] for w in train),
        "constitution_n_principles": len(principles),
        "test_controllable": sum(t["controllable"] for t in test_manifest),
        "artifacts": {"sft": str((ART / "sft_train.jsonl").relative_to(ROOT)),
                      "constitution": str((ART / "constitution.json").relative_to(ROOT)),
                      "test_manifest": str((ART / "test_manifest.json").relative_to(ROOT))},
        "config": {"test_fraction": TEST_FRACTION, "horizon": H, "states_per_world": STATES_PER_WORLD,
                   "planner_k": PLAN_K, "planner_h": PLAN_H, "seed": SEED},
    }
    save_results("e73_constitution", results)

    # ---- self-checks (after save_results) ----
    assert set(results["train_worlds"]).isdisjoint(results["test_worlds"]), "train/test leak!"
    assert len(train) >= 8 and len(test) >= 3, f"bad split: {len(train)}/{len(test)}"
    assert len(sft) >= 200, f"too little SFT data: {len(sft)}"
    assert len(principles) >= 5, f"thin constitution: {len(principles)}"
    assert results["test_controllable"] >= 1, "no controllable held-out world to evaluate on"
    print(f"[ok] E73 data: domain={DOMAIN} | split {len(train)} train / {len(test)} test worlds "
          f"(held out: {results['test_worlds']}) | {len(sft)} SFT examples | "
          f"constitution {len(principles)} principles | {results['test_controllable']}/{len(test)} "
          f"test worlds controllable")


if __name__ == "__main__":
    main()
