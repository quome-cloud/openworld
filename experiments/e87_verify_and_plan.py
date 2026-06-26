"""E87 verify-and-plan: parse Prism's predict() candidates, run exact-match gate,
then BFS through the verified model to attempt level completion on the official scorecard.

Usage:
  python3 e87_verify_and_plan.py --reply-dm <msg_body_file> --state e87_20260626_001
  python3 e87_verify_and_plan.py --reply-body '<json_or_kv_text>' --state e87_20260626_001
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STATES_DIR = HERE / "arc_synthesizer" / "states"
CALLS_DIR = HERE / "arc_synthesizer" / "calls"
ENVFILES = HERE / "environment_files"


# ---------------------------------------------------------------------------
# Reply parsing (handles both JSON and Prism's KV format)
# ---------------------------------------------------------------------------

def parse_reply(body_text):
    """Parse Prism's reply — same dual-format handling as arc_synthesizer_harness."""
    if "CANDIDATES_JSON:" in body_text:
        return _parse_kv(body_text)
    start = body_text.find("[")
    end = body_text.rfind("]")
    if start != -1 and end > start:
        try:
            candidates = json.loads(body_text[start:end + 1])
            if isinstance(candidates, list):
                return {"candidates": candidates, "protocol": "raw-array"}
        except json.JSONDecodeError:
            pass
    start = body_text.find("{")
    end = body_text.rfind("}")
    if start != -1 and end > start:
        data = json.loads(body_text[start:end + 1])
        return data
    raise ValueError("Could not parse reply body")


def _parse_kv(body_text):
    data = {"candidates": [], "status": "ok"}
    for line in body_text.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip(); val = val.strip()
        if key == "CANDIDATES_JSON":
            arr_s = val.find("["); arr_e = val.rfind("]")
            if arr_s != -1 and arr_e > arr_s:
                try:
                    data["candidates"] = json.loads(val[arr_s:arr_e + 1])
                except json.JSONDecodeError:
                    pass
        elif key in ("run_id", "seq", "track", "agent_id", "model_id", "status"):
            data[key] = val
    return data


# ---------------------------------------------------------------------------
# Sandbox execution for predict(frame, action) -> next_frame
# ---------------------------------------------------------------------------

def sandbox_predict(code, frame, action, timeout=15):
    """Run predict(frame, action) in a subprocess. Returns np.ndarray(64,64) or None."""
    harness = textwrap.dedent(f"""
import json, sys, numpy as np
{code}
frame_list = json.loads(sys.argv[1])
action = int(sys.argv[2])
frame = np.array(frame_list, dtype=np.int16).reshape(64, 64)
try:
    result = predict(frame, action)
    result = np.asarray(result, dtype=np.int16).reshape(64, 64)
    print(json.dumps(result.tolist()))
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
""")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness)
        fname = f.name
    try:
        proc = subprocess.run(
            [sys.executable, fname, json.dumps(frame.tolist()), str(action)],
            capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            return None
        result = json.loads(proc.stdout.strip())
        return np.array(result, dtype=np.int16).reshape(64, 64)
    except Exception:
        return None
    finally:
        try:
            Path(fname).unlink()
        except OSError:
            pass


def verify_candidate(code, held_out):
    """Exact-match fraction on held-out transitions. Returns (frac, n_correct, n_total)."""
    n_correct = 0
    for t in held_out:
        frame = np.array(t["frame"], dtype=np.int16).reshape(64, 64)
        expected = np.array(t["next"], dtype=np.int16).reshape(64, 64)
        pred = sandbox_predict(code, frame, t["action"])
        if pred is not None and np.array_equal(pred, expected):
            n_correct += 1
    return n_correct / len(held_out) if held_out else 0.0, n_correct, len(held_out)


# ---------------------------------------------------------------------------
# BFS planner using verified predict() (runs in-memory, fast)
# ---------------------------------------------------------------------------

def bfs_with_model(predict_fn, init_frame, max_steps=60, available_actions=(1, 2, 3, 4)):
    """BFS over model-predicted frames to find sequence reaching a new 'state'.
    Returns action_seq that changes max frame value (proxy for level completion) or None.
    This is a heuristic — real level detection requires env execution.
    """
    import collections
    queue = collections.deque([(init_frame.copy(), [])])
    visited = {init_frame.tobytes()}
    init_sum = init_frame.sum()

    while queue:
        frame, seq = queue.popleft()
        if len(seq) > max_steps:
            break
        for a in available_actions:
            nf = predict_fn(frame, a)
            if nf is None:
                continue
            key = nf.tobytes()
            if key in visited:
                continue
            visited.add(key)
            new_seq = seq + [a]
            # Heuristic: large frame change suggests transition event
            if abs(int(nf.sum()) - int(init_sum)) > 500:
                return new_seq, nf
            queue.append((nf, new_seq))
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reply-dm", default="", help="path to DM body text file")
    ap.add_argument("--reply-body", default="", help="DM body text inline")
    ap.add_argument("--state", default="e87_20260626_001", help="run state id")
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # Load state
    state_path = STATES_DIR / f"{args.state}.json"
    if not state_path.exists():
        print(f"ERROR: state file not found: {state_path}")
        sys.exit(1)
    state = json.loads(state_path.read_text())
    held_out = state["held_out"]
    task_hash = state["task_hash"]
    print(f"[e87-verify] run={args.state} held_out={len(held_out)} transitions")

    # Parse reply
    body_text = ""
    if args.reply_dm:
        body_text = Path(args.reply_dm).read_text()
    elif args.reply_body:
        body_text = args.reply_body
    else:
        print("ERROR: provide --reply-dm or --reply-body")
        sys.exit(1)

    reply = parse_reply(body_text)
    candidates = reply.get("candidates", [])
    print(f"[e87-verify] {len(candidates)} candidate(s) received from Prism")

    # Verify each candidate
    results = []
    best_frac = 0.0
    best_code = None
    for i, code in enumerate(candidates):
        if not isinstance(code, str) or "def predict" not in code:
            print(f"  candidate {i+1}: SKIP (no predict fn)")
            results.append({"idx": i, "skipped": True})
            continue
        t0 = time.time()
        frac, n_ok, n_tot = verify_candidate(code, held_out)
        elapsed = time.time() - t0
        print(f"  candidate {i+1}: exact={frac:.3f} ({n_ok}/{n_tot}) {elapsed:.1f}s")
        results.append({"idx": i, "exact_frac": frac, "n_correct": n_ok, "n_total": n_tot})
        if frac > best_frac:
            best_frac = frac
            best_code = code

    print(f"[e87-verify] best exact-match: {best_frac:.3f}")

    # Write call record
    call_dir = CALLS_DIR / args.state
    call_dir.mkdir(parents=True, exist_ok=True)
    call_record = {
        "run_id": args.state,
        "task_hash": task_hash,
        "game": args.game,
        "n_candidates": len(candidates),
        "verify_results": results,
        "best_exact_frac": best_frac,
        "best_code": best_code,
    }
    call_path = call_dir / "0001.json"
    call_path.write_text(json.dumps(call_record, indent=2))
    print(f"[e87-verify] wrote call record: {call_path}")

    # If best candidate is good, run BFS via model then execute in real env
    if best_code and best_frac >= 0.5:
        print(f"[e87-verify] best candidate at {best_frac:.3f} — attempting model BFS...")
        import arc_agi
        from arcengine import GameAction
        arc = arc_agi.Arcade()
        env = arc.make(args.game)
        obs = env.reset()
        init_frame = np.asarray(obs.frame[-1]).reshape(64, 64)

        ns = {"np": np, "numpy": np}
        exec(compile(best_code, "<synth>", "exec"), ns)
        predict_fn = ns["predict"]

        def safe_predict(f, a):
            try:
                r = np.asarray(predict_fn(f, a), dtype=np.int16).reshape(64, 64)
                return r
            except Exception:
                return None

        seq, _ = bfs_with_model(safe_predict, init_frame)
        if seq:
            print(f"[e87-verify] BFS found candidate sequence: {seq}")
            ACTS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                    GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION6,
                    GameAction.ACTION7]
            for i, a in enumerate(seq):
                obs = env.step(ACTS[a - 1])
                print(f"  exec step {i+1} a={a} levels={obs.levels_completed}")
                if obs.levels_completed > 0:
                    print(f"  *** LEVEL COMPLETED via verified model at step {i+1}! ***")
                    break
            sc = arc.get_scorecard()
            call_record["bfs_seq"] = seq
            call_record["scorecard"] = str(sc)
            call_path.write_text(json.dumps(call_record, indent=2))
        else:
            print("[e87-verify] BFS found no candidate sequence within budget")
    else:
        print(f"[e87-verify] best candidate {best_frac:.3f} < 0.5 threshold — skip BFS")

    out = Path(args.out) if args.out else HERE / "results" / f"e87_verify_{args.state}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": args.state,
        "n_candidates": len(candidates),
        "best_exact_frac": best_frac,
        "verify_results": results,
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"[e87-verify] wrote summary: {out}")


if __name__ == "__main__":
    main()
