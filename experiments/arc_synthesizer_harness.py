"""ARC Synthesizer Harness — Forge (A003) side of the T362 wire protocol.

Protocol spec: docs/arc_synthesizer_protocol.md

Sends demo-pair DMs to Prism (A004), collects candidate Python programs,
runs exact-match verify gate, applies vote on test input, writes per-call disk records.

Usage:
  python3 arc_synthesizer_harness.py --round-trip-test  # T362 exit criterion
  python3 arc_synthesizer_harness.py --run e89_20260625_001 --data /tmp/arc-agi-2/data/evaluation --n 40
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORDS_BASE = HERE / "arc_synthesizer"
SANDBOX_TIMEOUT = 10   # seconds per candidate


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------

def grids_equal(a, b):
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(len(ra) == len(rb) and all(x == y for x, y in zip(ra, rb))
               for ra, rb in zip(a, b))


def task_hash(demos, test_input):
    """sha256 of serialized demo pairs + test input (canonical sort)."""
    payload = json.dumps({"demos": demos, "test_input": test_input}, sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Sandbox executor
# ---------------------------------------------------------------------------

def sandbox_run(code, input_grid, timeout=SANDBOX_TIMEOUT):
    """Execute LLM-written code in subprocess. Returns predicted grid or None."""
    harness = textwrap.dedent(f"""
import json, sys
{code}
grid = json.loads(sys.argv[1])
try:
    result = transform(grid)
    print(json.dumps(result))
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
""")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness)
        fname = f.name
    try:
        proc = subprocess.run(
            [sys.executable, fname, json.dumps(input_grid)],
            capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            return None
        result = json.loads(proc.stdout.strip())
        if isinstance(result, list) and all(isinstance(r, list) for r in result):
            return result
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass
    return None


def verify_candidates(candidates, demos):
    """Return list of candidates that pass exact-match on ALL demo pairs."""
    verified = []
    for code in candidates:
        if not isinstance(code, str) or "def transform" not in code:
            continue
        ok = True
        for demo in demos:
            pred = sandbox_run(code, demo["input"])
            if pred is None or not grids_equal(pred, demo["output"]):
                ok = False
                break
        if ok:
            verified.append(code)
    return verified


def vote(verified, test_input):
    """Majority vote over verified candidates applied to test_input. Returns (grid, votes_dict)."""
    preds = []
    for code in verified:
        pred = sandbox_run(code, test_input)
        if pred is not None:
            preds.append(pred)
    if not preds:
        return None, {}
    votes = {}
    for p in preds:
        k = json.dumps(p)
        votes[k] = votes.get(k, 0) + 1
    winner_key = max(votes, key=votes.__getitem__)
    return json.loads(winner_key), votes


# ---------------------------------------------------------------------------
# DOH DM I/O
# ---------------------------------------------------------------------------

def doh_send(to, subject, body):
    """Send a DOH DM via mcp tool. Returns msg_id or None."""
    # Implemented via doh MCP — called from the main process using subprocess
    # when not running inside claude-code. In-session calls use the MCP directly.
    raise NotImplementedError("Call mcp__doh__doh_send_message directly from the agent session")


def build_request_dm(run_id, seq, track, demos, test_input, n_candidates=8):
    """Construct the protocol-compliant request body."""
    th = task_hash(demos, test_input)
    body = {
        "protocol": "arc-synth-v1",
        "run_id": run_id,
        "seq": seq,
        "track": track,
        "task_hash": th,
        "n_candidates": n_candidates,
        "demos": demos,
        "test_input": test_input,
        "sandbox_timeout": SANDBOX_TIMEOUT,
    }
    return json.dumps(body, indent=2), th


def parse_reply_dm(body_text):
    """Parse a [ARC-SYNTH-REPLY] DM body. Returns dict or raises ValueError."""
    start = body_text.find("{")
    end = body_text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in reply body")
    data = json.loads(body_text[start:end + 1])
    if data.get("protocol") != "arc-synth-v1":
        raise ValueError(f"Unexpected protocol: {data.get('protocol')}")
    return data


# ---------------------------------------------------------------------------
# Disk recording
# ---------------------------------------------------------------------------

def write_call_record(run_id, seq, track, task_hash_val, demos, test_input,
                      candidates_raw, candidates_verified, verify_result,
                      tokens_in, tokens_out, latency_ms, dm_request_id, dm_reply_id,
                      agent_id, model_id, status, ts):
    """Write one per-call record to disk."""
    call_dir = RECORDS_BASE / "calls" / run_id
    call_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": ts,
        "run_id": run_id,
        "seq": seq,
        "track": track,
        "task_hash": task_hash_val,
        "agent_id": agent_id,
        "model_id": model_id,
        "demo_pairs_serialized": json.dumps(demos),
        "test_input_serialized": json.dumps(test_input),
        "candidates_raw": candidates_raw,
        "candidates_verified": candidates_verified,
        "verify_result": verify_result,
        "tokens_est_in": tokens_in,
        "tokens_est_out": tokens_out,
        "latency_ms": latency_ms,
        "dm_msg_id_request": dm_request_id,
        "dm_msg_id_reply": dm_reply_id,
        "status": status,
    }
    path = call_dir / f"{seq:04d}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def write_run_manifest(run_id, track, eval_split, start_ts, end_ts, n_tasks,
                       n_calls, n_solved, total_in, total_out, hyperparams,
                       synthesizer_agent="A004", harness_agent="A003", model_id="claude-sonnet-4-6"):
    """Write run manifest."""
    runs_dir = RECORDS_BASE / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "track": track,
        "eval_split": eval_split,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "n_tasks": n_tasks,
        "n_calls": n_calls,
        "n_solved": n_solved,
        "pct_solved": round(100 * n_solved / n_tasks, 2) if n_tasks else 0.0,
        "total_tokens_est_in": total_in,
        "total_tokens_est_out": total_out,
        "contamination_boundary_proof": "demo pairs only; task_hash = sha256(demos_json + test_input_json)",
        "synthesizer_agent": synthesizer_agent,
        "harness_agent": harness_agent,
        "model_id": model_id,
        "hyperparams": hyperparams,
    }
    path = runs_dir / f"{run_id}.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Round-trip test helper (T362 exit criterion)
# ---------------------------------------------------------------------------

ROUNDTRIP_DEMOS = [
    {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},
    {"input": [[4, 5, 6], [7, 8, 9]], "output": [[6, 5, 4], [9, 8, 7]]},
]
ROUNDTRIP_TEST_INPUT = [[0, 1, 2, 3]]
ROUNDTRIP_TEST_ANSWER = [[3, 2, 1, 0]]


def complete_roundtrip(reply_body_text, dm_request_id, dm_reply_id):
    """Given Prism's reply body text, run verify gate + write records. Returns (acc, paths)."""
    import datetime
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    t0 = time.time()

    data = parse_reply_dm(reply_body_text)
    candidates_raw = data.get("candidates", [])
    candidates_verified = verify_candidates(candidates_raw, ROUNDTRIP_DEMOS)

    pred, votes = vote(candidates_verified, ROUNDTRIP_TEST_INPUT)
    acc = 1.0 if pred is not None and grids_equal(pred, ROUNDTRIP_TEST_ANSWER) else 0.0

    latency_ms = int((time.time() - t0) * 1000)
    th = task_hash(ROUNDTRIP_DEMOS, ROUNDTRIP_TEST_INPUT)

    verify_result = {
        "n_candidates_received": len(candidates_raw),
        "n_candidates_verified": len(candidates_verified),
        "passed": len(candidates_verified) > 0,
        "acc": acc,
    }

    call_path = write_call_record(
        run_id="test_roundtrip",
        seq=1,
        track="E89",
        task_hash_val=th,
        demos=ROUNDTRIP_DEMOS,
        test_input=ROUNDTRIP_TEST_INPUT,
        candidates_raw=candidates_raw,
        candidates_verified=candidates_verified,
        verify_result=verify_result,
        tokens_in=data.get("tokens_est_in", 0),
        tokens_out=data.get("tokens_est_out", 0),
        latency_ms=latency_ms,
        dm_request_id=dm_request_id,
        dm_reply_id=dm_reply_id,
        agent_id=data.get("agent_id", "A004"),
        model_id=data.get("model_id", "unknown"),
        status=data.get("status", "ok"),
        ts=ts,
    )

    manifest_path = write_run_manifest(
        run_id="test_roundtrip",
        track="E89",
        eval_split="synthetic",
        start_ts=ts,
        end_ts=ts,
        n_tasks=1,
        n_calls=1,
        n_solved=int(acc),
        total_in=data.get("tokens_est_in", 0),
        total_out=data.get("tokens_est_out", 0),
        hyperparams={"n_candidates": 8, "sandbox_timeout": SANDBOX_TIMEOUT, "burst_limit": 30},
    )

    return acc, call_path, manifest_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-trip-test", action="store_true",
                    help="Print the DM body to send to Prism (manual step)")
    args = ap.parse_args()

    if args.round_trip_test:
        body, th = build_request_dm(
            run_id="test_roundtrip",
            seq=1,
            track="E89",
            demos=ROUNDTRIP_DEMOS,
            test_input=ROUNDTRIP_TEST_INPUT,
        )
        print("=== DM to A004 ===")
        print(f"Subject: [ARC-SYNTH] v1 run=test_roundtrip seq=0001 track=E89")
        print("Body:")
        print(body)
        print(f"\ntask_hash: {th}")
