"""E58 - Brain simulator: a world-of-worlds with tree-of-thoughts ReAct + real memory.

A composable *brain*: a CompositeWorld of a CONSCIOUS world (working memory /
current thoughts), an UNCONSCIOUS world (long-term, content-addressable memory: a
learned transition model + cached plans), and an ENVIRONMENT world the brain acts
on. Perception maps an outside situation into the conscious world; the brain acts
back on the environment through tools.

An agent runs a ReAct loop with tree-of-thoughts: perceive -> retrieve memory ->
think (expand a depth-D tree over the learned model to find a plan to the goal) ->
act (take the best tool on the real environment) -> observe -> consolidate. Memory
is real and persistent, so a re-encountered situation is recalled, not re-derived.

Two deterministic, self-checking claims:
  A. memory wins  - across episodes on the same environment, a brain with a
     persistent unconscious learns to the optimal path length L; a memoryless
     agent re-explores every episode; random flails.
  B. how much to think - with the model known, steps-to-goal fall as the planning
     depth D rises and PLATEAU once D >= the needed horizon (no over-think gain).

Deterministic/offline; an optional LLM perceptor is available for a live demo but
is not part of the asserts.
"""

from collections import deque

import numpy as np

from openworld import (Aggregator, Bridge, CodePerceptor, CodeTransition,
                       CompositeWorld, World, from_spec, render_card, spec_to_json,
                       to_spec, validate_spec)
from openworld.state import Action

from common import save_results

SEED = 58
K, T = 12, 4                       # environment nodes, tools per node
EPISODES = 8                       # Panel A: episodes on the same env
N_ENVS = 24                        # Panel B: envs averaged per depth
DEPTHS = [0, 1, 2, 3, 4, 5]        # Panel B: planning depths
BUDGET = 40                        # step cap


# --------------------------------------------------------------------------- #
# environment: a seeded deterministic graph (tools are edges)
# --------------------------------------------------------------------------- #
def _shortest_len(edges, start, goal):
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in edges[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist.get(goal)


def make_env(seed):
    rng = np.random.RandomState(seed)
    edges = [[int(rng.randint(K)) for _ in range(T)] for _ in range(K)]
    start, goal = 0, K - 1
    if _shortest_len(edges, start, goal) is None:           # guarantee reachable
        reach = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in edges[u]:
                if v not in reach:
                    reach.add(v)
                    q.append(v)
        edges[max(reach)][0] = goal
    L = _shortest_len(edges, start, goal)
    return {"edges": edges, "start": start, "goal": goal, "L": L}


def full_model(env):
    return {(u, t): env["edges"][u][t] for u in range(K) for t in range(T)}


# --------------------------------------------------------------------------- #
# the agent: tree-of-thoughts over the learned model + persistent memory
# --------------------------------------------------------------------------- #
def plan_tree(model, node, goal, depth):
    """BFS over KNOWN edges up to `depth`; shortest tool-path to goal, else None."""
    q = deque([(node, [])])
    seen = {node}
    while q:
        u, path = q.popleft()
        if u == goal:
            return path
        if len(path) >= depth:
            continue
        for t in range(T):
            if (u, t) in model:
                v = model[(u, t)]
                if v not in seen:
                    seen.add(v)
                    q.append((v, path + [t]))
    return None


def refresh_cache(mem, goal):
    """Content-addressable recall: node -> shortest KNOWN tool-path to goal."""
    rev = {}
    for (u, t), v in mem["model"].items():
        rev.setdefault(v, []).append((u, t))
    cache = {goal: []}
    q = deque([goal])
    while q:
        v = q.popleft()
        for (u, t) in rev.get(v, []):
            if u not in cache:
                cache[u] = [t] + cache[v]
                q.append(u)
    mem["cache"] = cache


def run_episode(env, mem, depth, max_steps=BUDGET, random_mode=False,
                use_cache=True, rng=None):
    edges, start, goal = env["edges"], env["start"], env["goal"]
    node, steps = start, 0
    while node != goal and steps < max_steps:
        if random_mode:
            t = rng.randint(T)
        else:
            plan = mem["cache"].get(node) if use_cache else None
            if not plan:
                plan = plan_tree(mem["model"], node, goal, depth)
            if plan:
                t = plan[0]
            else:                                           # explore: untried tool
                untried = [tt for tt in range(T) if (node, tt) not in mem["model"]]
                t = untried[0] if untried else steps % T
        mem["model"][(node, t)] = edges[node][t]            # observe + consolidate
        node = edges[node][t]
        steps += 1
    if node == goal and not random_mode:
        refresh_cache(mem, goal)
    return steps


def fresh_mem():
    return {"model": {}, "cache": {}}


# --------------------------------------------------------------------------- #
# the brain as a CompositeWorld (serializable architecture + demo)
# --------------------------------------------------------------------------- #
ENV_CODE = """
def transition(state, action):
    s = dict(state); name = action["name"]
    if name.startswith("tool"):
        t = int(name[4:]); e = s["edges"]
        if 0 <= s["node"] < len(e) and 0 <= t < len(e[s["node"]]):
            s["node"] = e[s["node"]][t]; s["solved"] = (s["node"] == s["goal"])
    return s
"""
CONSCIOUS_CODE = """
def transition(state, action):
    s = dict(state)
    if action["name"] == "think":
        s["step"] = s.get("step", 0) + 1
    return s
"""
UNCONSCIOUS_CODE = "def transition(state, action):\n    return dict(state)"
BRIDGE_CODE = ('def transition(state, action):\n'
               '    return {"a": dict(state["a"]), "b": dict(state["b"])}')
PERCEIVE_CODE = """
def perceive(data):
    out = {}
    for line in str(data).splitlines():
        if ":" in line:
            k, v = line.split(":", 1); k = k.strip(); v = v.strip()
            if k == "node" and v.lstrip("-").isdigit():
                out["node"] = int(v)
    return out
"""


def brain_progress(children):
    return 1 if children.get("environment", {}).get("solved") else 0


def brain_world(env):
    """The brain: conscious + unconscious + environment, coupled by bridges,
    with a perception boundary. The serializable expression of the architecture."""
    conscious = World(name="conscious", description="working memory: current thought + plan",
                      initial_state={"node": env["start"], "plan": [], "step": 0},
                      actions=["think", "act"],
                      rules=["'think' advances the working step counter."],
                      transition=CodeTransition(CONSCIOUS_CODE))
    unconscious = World(name="unconscious", description="long-term content-addressable memory",
                        initial_state={"model": {}, "cache": {}},
                        actions=["consolidate", "retrieve"],
                        rules=["holds the learned transition model and cached plans."],
                        transition=CodeTransition(UNCONSCIOUS_CODE))
    environment = World(name="environment", description="the world the brain acts on",
                        initial_state={"node": env["start"], "goal": env["goal"],
                                       "edges": env["edges"], "solved": False},
                        actions=[f"tool{i}" for i in range(T)],
                        rules=["each 'toolK' applies tool K, moving along the graph; "
                               "solved when node == goal."],
                        transition=CodeTransition(ENV_CODE))
    bridges = [
        Bridge(name="retrieve", a="unconscious", b="conscious",
               transition=CodeTransition(BRIDGE_CODE),
               description="surface memories into working memory"),
        Bridge(name="consolidate", a="conscious", b="unconscious",
               transition=CodeTransition(BRIDGE_CODE),
               description="encode the experience into long-term memory"),
        Bridge(name="act", a="conscious", b="environment",
               transition=CodeTransition(BRIDGE_CODE),
               description="apply the chosen tool to the environment"),
    ]
    brain = CompositeWorld(
        name="brain", children={"conscious": conscious, "unconscious": unconscious,
                                "environment": environment},
        bridges=bridges, aggregators=[Aggregator(name="goal_progress", fn=brain_progress)],
        default_actions={"conscious": "think", "unconscious": "retrieve",
                         "environment": "tool0"},
        description="A composable brain: conscious working memory and unconscious "
                    "long-term memory, perceiving the outside and acting back on it.")
    brain.perceptors = [CodePerceptor(code=PERCEIVE_CODE, produces=["node"],
                                      schema={"node": (int, (0, K - 1))}, modality="text")]
    brain.emit = [{"modality": "action", "fields": ["node", "solved"],
                   "report": "at node {node}, solved={solved}"}]
    brain.objectives = [{"name": "reach the goal", "goal": "min steps to goal"}]
    return brain


def _rollout(world, actions):
    s, out = world.initial_state.copy(), []
    for a in actions:
        s = dict(world.transition.step(s, Action(a)))
        out.append(s)
    return out


def main():
    # --- Panel A: memory wins (episodes on the same env) ---
    env = next(e for e in (make_env(s) for s in range(SEED, SEED + 300))
               if e["L"] is not None and 3 <= e["L"] <= 6)   # a non-trivial env
    L = env["L"]
    brain_mem = fresh_mem()
    rng = np.random.RandomState(SEED)
    curves = {"brain": [], "no_memory": [], "random": []}
    for ep in range(EPISODES):
        curves["brain"].append(run_episode(env, brain_mem, depth=K))   # persistent
        curves["no_memory"].append(run_episode(env, fresh_mem(), depth=K))  # wiped
        curves["random"].append(run_episode(env, fresh_mem(), depth=K,
                                            random_mode=True, rng=rng))
    means = {k: round(float(np.mean(v)), 2) for k, v in curves.items()}

    # --- Panel B: how much to think (model known; depth sweep) ---
    depth_steps, depth_success = [], []
    for d in DEPTHS:
        st = []
        for j in range(N_ENVS):
            e = make_env(1000 + j)
            mem = {"model": full_model(e), "cache": {}}
            st.append(run_episode(e, mem, depth=d, use_cache=False))
        depth_steps.append(round(float(np.mean(st)), 2))
        depth_success.append(round(float(np.mean([s < BUDGET for s in st])), 3))

    # --- the brain world: serialize, validate, round-trip, render a card ---
    brain = brain_world(env)
    spec = to_spec(brain, card={"tags": ["brain", "composite", "perception"],
                                "license": "MIT", "version": "0.1",
                                "lineage": "E58 brain simulator"})
    problems = validate_spec(spec)
    acts = ["tick", "environment:tool0", "tick"]
    try:
        reloaded = from_spec(spec, allow_code=True)
        round_trip = _rollout(brain, acts) == _rollout(reloaded, acts)
    except Exception:
        round_trip = False
    from pathlib import Path
    gal = Path(__file__).resolve().parent.parent / "gallery"
    gal.mkdir(exist_ok=True)
    render_card(spec, path=str(gal / "brain.svg"))

    results = {
        "env": {"nodes": K, "tools": T, "optimal_len": L},
        "episodes": EPISODES,
        "panelA_curves": curves, "panelA_means": means,
        "panelA_brain_final": curves["brain"][-1],
        "panelA_optimal_gap": curves["brain"][-1] - L,
        "depths": DEPTHS, "panelB_steps": depth_steps, "panelB_success": depth_success,
        "plateau_depth": next((DEPTHS[i] for i in range(1, len(DEPTHS))
                               if depth_steps[i] == depth_steps[-1]), DEPTHS[-1]),
        "brain_validated": problems == [], "brain_round_trip": round_trip,
        "brain_children": len(spec.get("composite", {}).get("children", {})),
        "problems": problems,
    }
    save_results("e58_brain", results)

    print("E58 - brain simulator: tree-of-thoughts ReAct with real memory\n")
    print(f"  env: {K} nodes, {T} tools, optimal path L={L}")
    print(f"  Panel A mean steps  brain={means['brain']}  no_memory={means['no_memory']}"
          f"  random={means['random']}")
    print(f"    brain learning curve (steps/episode): {curves['brain']}")
    print(f"  Panel B steps by depth {DEPTHS}: {depth_steps}")
    print(f"    success by depth: {depth_success}  (plateau at D={results['plateau_depth']})")
    print(f"  brain world: {results['brain_children']} child worlds · "
          f"validated={results['brain_validated']} · round_trip={round_trip}")

    # --- self-checks ---
    # memoryless ~ random by design (neither persists across episodes); the brain's
    # persistent unconscious is what wins.
    assert means["brain"] < means["no_memory"], "memory should beat memoryless"
    assert means["brain"] < means["random"], "memory should beat random"
    assert curves["brain"][-1] < curves["brain"][0], "the brain should learn"
    assert curves["brain"][-1] == curves["brain"][-2], "recall should be stable"
    assert curves["brain"][-1] <= L + 1, "the brain should converge near-optimal"
    assert depth_steps[-1] < depth_steps[0], "deeper planning should reduce steps"
    assert depth_steps[-1] == depth_steps[-2], "steps should plateau (no over-think gain)"
    assert depth_success[-1] >= depth_success[0], "deeper planning shouldn't lower success"
    assert problems == [] and round_trip, "the brain world must validate and round-trip"
    print("\nchecks pass: persistent memory + bounded lookahead beat the baselines; "
          "brain world serializes losslessly.")


if __name__ == "__main__":
    main()
