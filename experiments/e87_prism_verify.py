"""e87_prism_verify.py — verify Prism's predict() candidates on a freshly collected held-out set.

End-to-end pipeline:
  1. Parse Prism's DM reply (extract Python code blocks)
  2. Re-collect transitions via collect(game, steps, seed=0)
  3. Split train/held (same 3/4 : 1/4 ratio as collect protocol)
  4. Verify each candidate with warmup on train (handles stateful _step=[] functions)
  5. If best ≥ 0.99: run BFS via model, then EXECUTE in real Arcade env
  6. Only print "BANK" if levels_completed > 0 after real env execution
  7. If best < 0.99: write a refinement DM body (--refine-out) showing failure cases

Usage:
  python3 e87_prism_verify.py --game ar25 --dm /path/to/prism_reply.md
  python3 e87_prism_verify.py --game cn04 --steps 245 --dm /path/to/prism_reply.md
  # After first round fails, auto-generate refinement DM:
  python3 e87_prism_verify.py --game ar25 --dm reply1.md --refine-out /tmp/ar25_refine.md
"""
import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e86_arc3 import collect, verify_code, ACTS  # noqa: E402


# ---------------------------------------------------------------------------
# Reply parsing — extract Python code blocks from Prism's markdown response
# ---------------------------------------------------------------------------

def extract_candidates(body_text: str) -> list[str]:
    """Pull ```python ... ``` blocks from Prism DM body."""
    blocks = re.findall(r"```python\s*(.*?)```", body_text, re.S)
    if not blocks:
        blocks = re.findall(r"```\s*(.*?)```", body_text, re.S)
    return [b.strip() for b in blocks if "def predict" in b]


# ---------------------------------------------------------------------------
# BFS planner over in-process predict()
# ---------------------------------------------------------------------------

def bfs_levels_completed(predict_fn, init_frame, available_actions, max_steps=80):
    """BFS over model-predicted frames.
    Stops when predict_fn changes frame significantly (proxy for level trigger).
    Returns (action_seq, final_pred_frame) or (None, None).
    The real level-completion check is done in Arcade — this just finds a candidate seq.
    """
    queue = collections.deque([(init_frame.copy(), [])])
    visited = {init_frame.tobytes()}
    init_sum = int(init_frame.sum())

    while queue:
        frame, seq = queue.popleft()
        if len(seq) >= max_steps:
            continue
        for a in available_actions:
            try:
                nf = np.asarray(predict_fn(frame.copy(), a), dtype=np.int16).reshape(64, 64)
            except Exception:
                continue
            key = nf.tobytes()
            if key in visited:
                continue
            visited.add(key)
            new_seq = seq + [a]
            # Large frame delta = likely level transition
            if abs(int(nf.sum()) - init_sum) > 500:
                return new_seq, nf
            queue.append((nf, new_seq))
    return None, None


def build_refinement_dm(game_id, best_code, errs, best_frac, n_held):
    """Build a Prism DM body for multi-round refinement.

    Shows the best candidate's code + the failure transitions it got wrong,
    asks Prism to return 2-4 refined candidates that fix those cases.
    """
    from e86_arc3 import deltas

    failure_lines = []
    for t in errs[:8]:  # cap at 8 failure examples
        if "exc" in t:
            failure_lines.append(f"  EXCEPTION: {t['exc']!r} on action={t.get('action')}")
        else:
            frame = np.asarray(t["frame"])
            nxt = np.asarray(t["next"])
            d = deltas(frame, nxt)
            failure_lines.append(f"  action {t['action']} → expected delta: {d[:20]}")

    failures_block = "\n".join(failure_lines)
    return f"""# ARC-SYNTH refinement request — game: {game_id}

## Context

Previous synthesis round: best candidate exact-match = **{best_frac:.4f}** on {n_held} held-out transitions.
**Not yet ≥0.99.** Please refine.

## Best candidate so far

```python
{best_code}
```

## Failure cases (transitions the best candidate got wrong)

```
{failures_block}
```

## Request

Please return **2-4 refined Python candidates** that fix the failure cases above.
Focus on the specific patterns you see in the delta mismatches.
Each candidate should use a different fix strategy.
Same function signature: `def predict(frame: np.ndarray, action: int) -> np.ndarray`

Label them C1_refine, C2_refine, etc.

— Forge (A003)
"""


def execute_seq_in_env(game_id, seq, available_actions):
    """Execute action sequence in real Arcade env from fresh reset.
    Returns (levels_completed, steps_executed).
    Only BANK the solution if levels_completed > 0.
    """
    import arc_agi
    from arcengine import GameAction
    arc = arc_agi.Arcade()
    env = arc.make(game_id)
    obs = env.reset()
    levels = 0
    for i, a in enumerate(seq):
        ga = ACTS[a - 1]
        obs = env.step(ga)
        if obs is None:
            print(f"  real-env: step {i+1} action={a} → obs=None (invalid)")
            break
        levels = max(levels, obs.levels_completed)
        print(f"  real-env: step {i+1} action={a} levels_completed={obs.levels_completed}")
        if obs.levels_completed > 0:
            print(f"  *** LEVEL COMPLETED at step {i+1} — BANK THIS SEQUENCE ***")
            break
    sc = arc.get_scorecard()
    print(f"  scorecard: {sc}")
    return levels, len(seq)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True, help="Game id, e.g. ar25 or cn04")
    ap.add_argument("--dm", required=True, help="Path to Prism DM body text file")
    ap.add_argument("--steps", type=int, default=300, help="Transition collection budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="Optional path to write JSON summary")
    ap.add_argument("--refine-out", default="", dest="refine_out",
                    help="If best < 0.99, write refinement DM body to this path")
    args = ap.parse_args()

    body_text = Path(args.dm).read_text()
    candidates = extract_candidates(body_text)
    print(f"[prism-verify] game={args.game} extracted {len(candidates)} candidate(s)")
    if not candidates:
        print("ERROR: no predict() code blocks found in DM")
        sys.exit(1)

    print(f"[prism-verify] collecting {args.steps} transitions (seed={args.seed})...")
    t0 = time.time()
    trans, best_levels, win_levels = collect(args.game, args.steps, args.seed)
    print(f"[prism-verify] collected {len(trans)} transitions in {time.time()-t0:.1f}s "
          f"(best_levels={best_levels}, win_levels={win_levels})")

    n = len(trans)
    split = n * 3 // 4
    train = trans[:split]
    held = trans[split:]
    print(f"[prism-verify] train={len(train)} held={len(held)}")

    available_actions = sorted({t["action"] for t in trans})
    print(f"[prism-verify] actions in data: {available_actions}")

    results = []
    best_frac = 0.0
    best_code = None
    best_idx = -1

    for i, code in enumerate(candidates):
        label = f"C{i+1}"
        t0 = time.time()
        # warmup on train so stateful _step=[] counters reach the right position
        frac, errs = verify_code(code, held, warmup=train)
        elapsed = time.time() - t0
        print(f"  {label}: exact={frac:.4f} ({int(frac*len(held))}/{len(held)}) {elapsed:.1f}s"
              + (f"  [{len(errs)} errors]" if errs else ""))
        results.append({"label": label, "exact_frac": frac,
                        "n_correct": int(frac * len(held)), "n_total": len(held)})
        if frac > best_frac:
            best_frac = frac
            best_code = code
            best_idx = i

    print(f"\n[prism-verify] best: C{best_idx+1} exact={best_frac:.4f}")

    seq = None
    real_levels = 0

    if best_frac >= 0.99:
        print("[prism-verify] ≥0.99 gate PASSED — running BFS via model...")
        ns = {"np": np, "numpy": np}
        exec(compile(best_code, "<synth>", "exec"), ns)  # noqa: S102
        predict_fn = ns["predict"]

        # warm up to end of train so the in-memory predict_fn state is at step=len(train)
        for t in train:
            try:
                predict_fn(np.asarray(t["frame"]), t["action"])
            except Exception:
                pass

        # get a fresh init frame from the env
        import arc_agi
        arc = arc_agi.Arcade()
        env = arc.make(args.game)
        obs = env.reset()
        init_frame = np.asarray(obs.frame).reshape(64, 64) if np.asarray(obs.frame).ndim == 2 \
            else np.asarray(obs.frame[-1]).reshape(64, 64)

        # re-exec predict in a clean ns so step counter starts at 0 for planning
        ns2 = {"np": np, "numpy": np}
        exec(compile(best_code, "<synth>", "exec"), ns2)  # noqa: S102
        plan_predict = ns2["predict"]

        seq, _ = bfs_levels_completed(plan_predict, init_frame, available_actions)
        if seq:
            print(f"[prism-verify] BFS candidate sequence ({len(seq)} steps): {seq}")
            real_levels, _ = execute_seq_in_env(args.game, seq, available_actions)
        else:
            print("[prism-verify] BFS exhausted without finding level-transition candidate")
    else:
        print(f"[prism-verify] best={best_frac:.4f} < 0.99 — no BFS (needs refinement)")
        if args.refine_out and best_code:
            # Re-verify best candidate to collect failure cases for refinement DM
            _, errs = verify_code(best_code, held, warmup=train)
            dm_body = build_refinement_dm(args.game, best_code, errs, best_frac, len(held))
            Path(args.refine_out).write_text(dm_body)
            print(f"[prism-verify] refinement DM body written → {args.refine_out}")

    summary = {
        "game": args.game,
        "n_trans": len(trans),
        "n_train": len(train),
        "n_held": len(held),
        "best_exact_frac": best_frac,
        "best_candidate_idx": best_idx,
        "results": results,
        "bfs_seq": seq,
        "real_levels_completed": real_levels,
        "banked": real_levels > 0,
    }

    out_path = Path(args.out) if args.out else \
        HERE / "results" / f"e87_prism_verify_{args.game}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[prism-verify] summary → {out_path}")
    if real_levels > 0:
        print(f"*** BANK: {args.game} SOLVED — real_levels_completed={real_levels} ***")
    return summary


if __name__ == "__main__":
    main()
