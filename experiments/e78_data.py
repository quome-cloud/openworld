"""E78 (QLoRA data gen) - distill the VERIFIED world model into SFT data.

The agentic payoff of E78, in the repo's fine-tune idiom (cf. E73 "behavior cloning of a
model-based planner"): Blocksworld/PlanBench is the canonical "LLMs can't plan" benchmark,
and a base qwen2.5 floors on it (E78's runtime-tool arms: A0/A1/A2 ~ 0). Here the verified
world model is used not as a runtime tool but as a TEACHER: its BFS oracle produces optimal
plans (perfect, free labels), and the verified validator later scores generations. We
fine-tune on those labels and test whether planning skill transfers.

Each SFT example is one instance: prompt = the same state+goal+rules prompt the eval asks
(build_prompt from the tool experiment), completion = the optimal plan as PLAN text.

Splits (world-level, disjoint by canonical (init,goal) key -- no train instance leaks):
  train      : horizons {2,4,6,8}            -- the lengths we teach.
  test_id    : horizons {2,4,6,8}, held-out  -- in-distribution generalization.
  test_long  : horizons {10,12}              -- LONGER than anything trained: tests whether
                                               the model learned an algorithm or memorized.

Writes experiments/results/e78_artifacts/{sft_train.jsonl, test.jsonl, data_meta.json}.
Deterministic (stdlib + openworld core; no LLM, no GPU). Pure CPU.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import blocksworld as bw
from e78_world_model_tool import _plan_text, build_prompt

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "experiments" / "results" / "e78_artifacts"

# 4 blocks: matches the E78 runtime-tool substrate, has ample distinct instances at every
# (even) horizon 2..12, and BFS is fast. Blocksworld optimal lengths from a hand-empty start
# are always even (each block move = lift+place), so horizons are even.
N_BLOCKS = 4
TRAIN_HORIZONS = [2, 4, 6, 8]
TEST_ID_HORIZONS = [2, 4, 6, 8]
TEST_LONG_HORIZONS = [10, 12]
N_TRAIN_PER = 150          # target distinct train instances per horizon (accept fewer)
N_TEST_PER = 30            # target distinct test instances per horizon
SEED = 7800
MAX_SAMPLES = 200000       # global sample budget (early-exit once every bucket is full)


def inst_key(prob):
    """Canonical, hashable instance identity for train/test disjointness."""
    init, goal = prob["init"], prob["goal"]
    return (frozenset(init["on"].items()), frozenset(init["table"]), init.get("holding"),
            frozenset(goal.get("on", {}).items()), frozenset(goal.get("table", [])))


def gen_one(n_blocks, rng):
    """One random solvable instance, labeled with its optimal plan (single BFS, no rejection)."""
    blocks = [chr(ord("a") + i) for i in range(n_blocks)]
    init = bw._random_config(list(blocks), rng)
    gcfg = bw._random_config(list(blocks), rng)
    goal = {"on": dict(gcfg["on"]), "table": [b for b in blocks if b in set(gcfg["table"])]}
    plan = bw.bfs_plan(init, goal)
    if plan is None or len(plan) == 0:
        return None
    return {"n_blocks": n_blocks, "init": init, "goal": goal, "optimal_len": len(plan),
            "optimal_plan": [[n, p] for n, p in plan]}


def to_sft(prob):
    plan = [(n, p) for n, p in prob["optimal_plan"]]
    return {"prompt": build_prompt(prob["init"], prob["goal"], []),
            "completion": _plan_text(plan)}


def to_test(prob, split):
    return {"prompt": build_prompt(prob["init"], prob["goal"], []),
            "init": prob["init"], "goal": prob["goal"],
            "optimal_len": prob["optimal_len"], "optimal_plan": prob["optimal_plan"],
            "split": split}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_blocks", type=int, default=N_BLOCKS)
    ap.add_argument("--n_train", type=int, default=N_TRAIN_PER)
    ap.add_argument("--n_test", type=int, default=N_TEST_PER)
    args = ap.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    seen = set()
    train_b = {L: [] for L in TRAIN_HORIZONS}
    test_b = {L: [] for L in TEST_ID_HORIZONS + TEST_LONG_HORIZONS}
    long_set = set(TEST_LONG_HORIZONS)

    def full():
        return (all(len(v) >= args.n_train for v in train_b.values())
                and all(len(v) >= args.n_test for v in test_b.values()))

    # Single pass: one BFS per random instance, dropped into its true horizon bucket. Train is
    # filled first (its keys reserved in `seen`), so test draws only disjoint instances.
    for i in range(MAX_SAMPLES):
        if full():
            break
        prob = gen_one(args.n_blocks, rng)
        if prob is None:
            continue
        L, k = prob["optimal_len"], inst_key(prob)
        if k in seen:
            continue
        if L in train_b and len(train_b[L]) < args.n_train:
            seen.add(k)
            train_b[L].append(prob)
        elif L in test_b and len(test_b[L]) < args.n_test:
            seen.add(k)
            test_b[L].append(prob)

    counts = {"train": {L: len(v) for L, v in train_b.items()},
              "test_id": {L: len(test_b[L]) for L in TEST_ID_HORIZONS},
              "test_long": {L: len(test_b[L]) for L in TEST_LONG_HORIZONS}}
    train = [p for L in TRAIN_HORIZONS for p in train_b[L]]
    test = ([(p, "test_id") for L in TEST_ID_HORIZONS for p in test_b[L]]
            + [(p, "test_long") for L in TEST_LONG_HORIZONS for p in test_b[L]])

    rng.shuffle(train)
    (ART / "sft_train.jsonl").write_text(
        "\n".join(json.dumps(to_sft(p)) for p in train) + "\n", encoding="utf-8")
    (ART / "test.jsonl").write_text(
        "\n".join(json.dumps(to_test(p, s)) for p, s in test) + "\n", encoding="utf-8")

    meta = {"n_blocks": args.n_blocks, "seed": SEED,
            "train_horizons": TRAIN_HORIZONS, "test_id_horizons": TEST_ID_HORIZONS,
            "test_long_horizons": TEST_LONG_HORIZONS,
            "n_train": len(train), "n_test": len(test), "counts": counts}
    (ART / "data_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[e78-data] n_blocks={args.n_blocks}  train={len(train)}  test={len(test)}")
    print(f"  train per horizon : {counts['train']}")
    print(f"  test_id per horizon : {counts['test_id']}")
    print(f"  test_long per horizon: {counts['test_long']}")
    print(f"  wrote {ART/'sft_train.jsonl'} , {ART/'test.jsonl'}")


if __name__ == "__main__":
    main()
