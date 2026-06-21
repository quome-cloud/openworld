"""E78 - Does giving an LLM an OpenWorld world model AS A TOOL make it a better planner?

The agentic payoff of the paper's core findings: LLMs compound error over multi-step
dynamics (E01) and planning *through* an unverified model is worse than no model (E22). So
on a benchmark LLMs are known to fail at -- Blocksworld / PlanBench (Valmeekam et al.,
NeurIPS 2023) -- we hand the model a verified OpenWorld world model as a callable tool and
ask whether it now plans.

Three arms on identical instances (paired):
  A0  llm_only      - propose a full plan, no checking (must track state in its head).
  A1  llm_sim_tool  - propose, then check the plan through the LLM's OWN next-state
                      predictions (an UNVERIFIED simulator); retry while it *believes*
                      the plan wrong. Tests whether a 'simulate' button alone helps.
  A2  verified_tool - propose, then check the plan through the VERIFIED OpenWorld world
                      (exact dynamics + goal test); retry while it is *actually* wrong.
                      The tool only EXECUTES the plan the LLM proposes -- it does not
                      search -- so any win is exact-state-tracking offload, not a solver.
Reference arm: ORACLE (BFS over the verified world; the solvable upper bound).

Every arm's final plan is scored by the SAME verified validator. We stratify by optimal
plan length (the horizon): the paper predicts the A2-over-A0 gap GROWS with horizon
(compounding error), and that A1 does not beat A0 (E22). Paired McNemar across arms.

Default run is an OFFLINE DRY-RUN with a deterministic mock planner -- it validates the
harness (parser, loop, scorer, stats, the measurable arm ordering) with no Ollama/GPU.
Pass --live to run a real model via Ollama (qwen2.5). Deterministic; save_results runs
BEFORE the asserts so a failed self-check never loses the run.
"""

from __future__ import annotations

import argparse
import random

import blocksworld as bw
from common import GENERATOR_MODEL, mcnemar_p, save_results, wilson_ci

HORIZONS = [2, 4, 6, 8, 10, 12]
N_BLOCKS = 4
N_PER_BUCKET = 25
MAX_ROUNDS = 4          # tool-use rounds for A1/A2


# ---------------------------------------------------------------------------
# Plan text protocol (what a real model emits; exercised by the mock too)
# ---------------------------------------------------------------------------

def parse_plan(text):
    """Extract a plan (list of (name, params)) from free-form model text. Lines like
    'stack(a, b)' / 'pickup(a)' are picked up; prose and a leading 'PLAN:' are ignored."""
    plan = []
    for raw in text.splitlines():
        line = raw.strip().strip("-*0123456789. ").lower()
        if "(" not in line or ")" not in line:
            continue
        name = line[:line.index("(")].strip()
        if name not in bw.ACTIONS:
            continue
        args = [a for a in line[line.index("(") + 1:line.index(")")].replace(" ", "").split(",") if a]
        if name in ("pickup", "putdown") and len(args) == 1:
            plan.append((name, {"x": args[0]}))
        elif name in ("stack", "unstack") and len(args) == 2:
            plan.append((name, {"x": args[0], "y": args[1]}))
    return plan


def _plan_text(plan):
    def fmt(n, p):
        return f"{n}({p['x']})" if n in ("pickup", "putdown") else f"{n}({p['x']}, {p['y']})"
    return "PLAN:\n" + "\n".join(fmt(n, p) for n, p in plan)


def build_prompt(init, goal, history):
    lines = [bw.DESCRIPTION, "Actions: pickup(x), putdown(x), stack(x,y), unstack(x,y).",
             f"Initial: on={init['on']} table={sorted(init['table'])} holding={init['holding']}",
             f"Goal: on={goal.get('on', {})} table={sorted(goal.get('table', []))}"]
    for i, (plan, fb) in enumerate(history):
        lines.append(f"Attempt {i + 1} was rejected: {fb}")
    lines.append("Reply with the plan, one action per line, after a 'PLAN:' header.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The arm runner: identical control flow, swap only the checker
# ---------------------------------------------------------------------------

def _fmt_state(s):
    return f"on={s.get('on', {})} table={sorted(s.get('table', []))} holding={s.get('holding')}"


def feedback(rollout):
    """Same-shape feedback for A1 and A2 from a {first_illegal, reached, final_state} rollout.
    A2 passes the VERIFIED rollout; A1 passes the LLM-PREDICTED one -- identical wording, so
    the only thing that differs between the arms is whether the rollout is true."""
    fi = rollout["first_illegal"]
    if fi:
        n, p = fi["action"]
        return f"action {fi['index'] + 1} {n}({p}) is illegal: {fi['reason']}"
    if not rollout["reached"]:
        return f"all actions ran but the goal is not reached; resulting state: {_fmt_state(rollout['final_state'])}"
    return "valid"


def run_arm(arm, problem, propose, sim_check=None, max_rounds=MAX_ROUNDS):
    """propose(prompt)->text. sim_check(init,goal,plan)-> a rollout dict with the same keys as
    bw.validate_plan (believed via the LLM's own predictions) for A1. A0 commits its first
    plan; A1/A2 simulate-and-revise with IDENTICAL feedback, differing only in the simulator.
    The committed plan is ALWAYS scored by the verified scorer."""
    init, goal = problem["init"], problem["goal"]
    history, plan = [], []
    rounds = 0
    for r in range(max_rounds if arm != "A0" else 1):
        rounds = r + 1
        plan = parse_plan(propose(build_prompt(init, goal, history)))
        if arm == "A0":
            break
        rollout = bw.validate_plan(init, goal, plan) if arm == "A2" else sim_check(init, goal, plan)
        believed_valid = (rollout["first_illegal"] is None and rollout["reached"])
        if believed_valid:
            break
        history.append((plan, feedback(rollout)))
    score = bw.validate_plan(init, goal, plan)                 # ALWAYS the verified scorer
    score["rounds"] = rounds
    return score


def oracle_score(problem):
    plan = [(n, p) for n, p in bw.bfs_plan(problem["init"], problem["goal"])]
    return bw.validate_plan(problem["init"], problem["goal"], plan)


# ---------------------------------------------------------------------------
# Offline mock: a planner whose competence is the true plan corrupted at a rate that the
# tool can help fix. Exercises the full pipeline and the measurable arm ordering.
# ---------------------------------------------------------------------------

class MockPlanner:
    """Returns the optimal plan corrupted by per-action noise; noise drops on retry (the
    'agent' uses tool feedback to try harder). Independent of arm -- the ARM decides whether
    a retry happens, so A2 (retries on true failure) ends valid more often than A0 (no
    retry) and A1 (retries only when its noisy belief flags failure)."""

    def __init__(self, problem, rng, err0=0.45, err1=0.12):
        self.truth = [(n, p) for n, p in bw.bfs_plan(problem["init"], problem["goal"])]
        self.blocks = [chr(ord("a") + i) for i in range(problem["n_blocks"])]
        self.rng, self.err0, self.err1 = rng, err0, err1
        self.attempt = -1

    def ask(self, prompt, system=None):
        self.attempt += 1
        err = self.err0 if self.attempt == 0 else self.err1
        plan = []
        for n, p in self.truth:
            if self.rng.random() < err:                # corrupt -> usually illegal/goal-miss
                q = dict(p)
                q["x"] = self.rng.choice(self.blocks)
                plan.append((n, q))
            else:
                plan.append((n, dict(p)))
        return _plan_text(plan)


def mock_sim_check(init, goal, plan, rng, fidelity=0.7):
    """Stand-in for rolling a plan through the LLM's own predictions: each step is predicted
    correctly only with prob `fidelity` (compounding drift), so the predicted rollout -- and
    thus the agent's belief AND its feedback -- can diverge from the truth. Returns the same
    shape as bw.validate_plan so A1 and A2 feedback is identical except for the source."""
    state = dict(init)
    first_illegal = None
    for i, (n, p) in enumerate(plan):
        nxt = bw.step(state, n, **p)
        if rng.random() > fidelity:                    # predicted state drifts from truth
            nxt = dict(state)
            nxt["_ok"] = True
        if not nxt.get("_ok"):
            first_illegal = {"index": i, "action": [n, p], "reason": nxt.get("_msg")}
            break
        state = nxt
    return {"first_illegal": first_illegal, "reached": bw.goal_satisfied(state, goal),
            "final_state": {"on": dict(state.get("on", {})),
                            "table": sorted(state.get("table", [])),
                            "holding": state.get("holding")}}


# ---------------------------------------------------------------------------
# Live adapters (only touched with --live; keep import-time offline)
# ---------------------------------------------------------------------------

def live_propose_and_sim(model):
    from openworld.transition import LLMTransition
    from openworld import Action, WorldState
    from common import require_ollama
    llm = require_ollama(model, temperature=0.0, options={"num_ctx": 8192})
    sim = LLMTransition(llm, bw.DESCRIPTION, bw.RULES)

    def propose(prompt):
        try:
            return llm.ask(prompt)
        except Exception:
            return ""

    def sim_check(init, goal, plan):
        # Roll the plan through the LLM's OWN next-state predictions. Unlike the verified
        # world, the LLM-simulator has no notion of legality -- it just predicts a state for
        # whatever it's given (often hallucinating that an illegal move worked); the only
        # signal is whether the PREDICTED final state looks like the goal. Same return shape
        # as bw.validate_plan, so feedback() renders it identically.
        state = dict(init)
        for n, p in plan:
            try:
                state = dict(sim.step(WorldState(state), Action(n, p)))
            except Exception:
                break
        return {"first_illegal": None, "reached": bw.goal_satisfied(state, goal),
                "final_state": {"on": state.get("on", {}),
                                "table": sorted(state.get("table", []) or []),
                                "holding": state.get("holding")}}

    return propose, sim_check


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_instances(rng):
    insts = []
    for L in HORIZONS:
        got = 0
        while got < N_PER_BUCKET:
            prob = bw.gen_problem(N_BLOCKS, L, rng)
            if prob is None:
                continue
            insts.append(prob)
            got += 1
    return insts


def _rate(flags):
    n, k = len(flags), sum(flags)
    lo, hi = wilson_ci(k, n)
    return {"valid_rate": round(k / n, 4) if n else 0.0, "n": n,
            "ci": [round(lo, 4), round(hi, 4)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="run a real model via Ollama")
    ap.add_argument("--model", default=GENERATOR_MODEL)
    args = ap.parse_args()
    mode = "live" if args.live else "mock_dryrun"

    rng = random.Random(78)
    instances = make_instances(rng)

    live = None
    if args.live:
        live = live_propose_and_sim(args.model)

    # per-arm validity flags, aligned across instances (for paired McNemar)
    arms = ["A0", "A1", "A2", "ORACLE"]
    flags = {a: [] for a in arms}
    by_horizon = {a: {L: [] for L in HORIZONS} for a in arms}
    rounds = {"A1": [], "A2": []}

    for i, prob in enumerate(instances):
        L = prob["optimal_len"]
        if args.live:
            propose_fn, sim_fn = live  # one shared model; seeds N/A

            def sc(init, goal, plan, _f=sim_fn):
                return _f(init, goal, plan)
        else:
            # index-based seeds keep the whole dry-run reproducible (hash() is salted)
            propose_fn = MockPlanner(prob, random.Random(1000 + i)).ask
            srng = random.Random(2000 + i)

            def sc(init, goal, plan, _r=srng):
                return mock_sim_check(init, goal, plan, _r)

        a0 = run_arm("A0", prob, propose_fn)
        a1 = run_arm("A1", prob, propose_fn, sim_check=sc)
        a2 = run_arm("A2", prob, propose_fn)
        orc = oracle_score(prob)

        for a, res in (("A0", a0), ("A1", a1), ("A2", a2), ("ORACLE", orc)):
            flags[a].append(bool(res["valid"]))
            by_horizon[a][L].append(bool(res["valid"]))
        rounds["A1"].append(a1["rounds"])
        rounds["A2"].append(a2["rounds"])

    def mcnemar(x, y):
        b = sum(1 for xv, yv in zip(flags[x], flags[y]) if xv and not yv)
        c = sum(1 for xv, yv in zip(flags[x], flags[y]) if yv and not xv)
        return {"b_only_%s" % x: b, "b_only_%s" % y: c, "p": round(mcnemar_p(b, c), 5)}

    summary = {a: _rate(flags[a]) for a in arms}
    horizon_curve = {a: {str(L): _rate(by_horizon[a][L]) for L in HORIZONS} for a in arms}

    save_results("e78_world_model_tool", {
        "mode": mode,
        "model": args.model if args.live else "mock-deterministic-planner",
        "benchmark": "blocksworld (PlanBench-style), %d blocks" % N_BLOCKS,
        "n_instances": len(instances), "horizons": HORIZONS, "n_per_bucket": N_PER_BUCKET,
        "max_tool_rounds": MAX_ROUNDS,
        "arms": {"A0": "llm_only", "A1": "llm_sim_tool (unverified)",
                 "A2": "verified_tool", "ORACLE": "bfs"},
        "summary": summary,
        "by_horizon": horizon_curve,
        "mcnemar": {"A2_vs_A0": mcnemar("A2", "A0"), "A2_vs_A1": mcnemar("A2", "A1"),
                    "A1_vs_A0": mcnemar("A1", "A0")},
        "mean_rounds": {a: round(sum(v) / len(v), 2) for a, v in rounds.items()},
    })

    print(f"[e78] mode={mode}  instances={len(instances)}")
    for a in arms:
        print(f"  {a:7} valid {summary[a]['valid_rate']:.2f}  CI{summary[a]['ci']}")
    print(f"  A2 vs A0 McNemar p={summary and mcnemar('A2', 'A0')['p']}")

    # ---- self-checks AFTER saving (a failed assert must not lose the run) ----
    assert summary["ORACLE"]["valid_rate"] == 1.0, "BFS oracle must solve every instance"
    if mode == "mock_dryrun":
        # the harness must be able to MEASURE the predicted ordering with the mock:
        assert summary["A2"]["valid_rate"] >= summary["A0"]["valid_rate"], \
            "verified tool should not underperform no-tool in the mock"
        assert summary["A2"]["valid_rate"] >= summary["A1"]["valid_rate"] - 1e-9, \
            "verified tool should be >= unverified-sim tool in the mock"
    print("[e78] self-checks passed")


if __name__ == "__main__":
    main()
