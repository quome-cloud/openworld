"""E89 Run Manager — stateful multi-cycle orchestrator for the ARC-AGI-2 synthesis eval.

The E89 eval sends tasks to Prism (A004) via DOH DMs across multiple agent cycles:
  Cycle 1: Initialize run state, send first 30 task DMs to Prism
  Cycle 2+: Collect Prism replies, verify candidates, write call records, send next burst
  Final: Write run manifest, report results

State is persisted to disk so each cycle picks up where the previous left off.

Usage (called from within the Forge agent session each cycle):
  from e89_run_manager import RunManager
  mgr = RunManager.load_or_init("e89_20260625_001", ARC2_DATA_DIR, n_tasks=120)
  mgr.send_burst(doh_send_fn, n=30)    # send next N pending tasks as DMs
  mgr.collect_replies(doh_list_fn)      # match A004 replies, verify, record
  mgr.print_status()
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import e80_arc as A
from arc_synthesizer_harness import (
    task_hash, build_request_dm, parse_reply_dm,
    verify_candidates, vote, grids_equal,
    write_call_record, write_run_manifest,
    SANDBOX_TIMEOUT,
)

ARC2_DEFAULT = "/tmp/arc-agi-2/data/evaluation"
STATES_DIR = HERE / "arc_synthesizer" / "states"
SEED = 89


def _task_entry(seq, task_id, demos, test_input, test_answer):
    return {
        "seq": seq,
        "task_id": task_id,
        "task_hash": task_hash(demos, test_input),
        "demos": demos,
        "test_input": test_input,
        "test_answer": test_answer,
        "status": "pending",   # pending | sent | done | error | timeout
        "dm_request_id": None,
        "dm_reply_id": None,
        "acc": None,
        "n_verified": None,
        "retry": 0,
    }


class RunManager:
    BURST_LIMIT = 30

    def __init__(self, run_id, track, tasks_list, hyperparams=None):
        self.run_id = run_id
        self.track = track
        self.tasks = tasks_list  # list of _task_entry dicts
        self.hyperparams = hyperparams or {
            "n_candidates": 8, "sandbox_timeout": SANDBOX_TIMEOUT, "burst_limit": self.BURST_LIMIT
        }
        self.start_ts = None

    @classmethod
    def load_or_init(cls, run_id, data_dir=ARC2_DEFAULT, n_tasks=120, track="E89"):
        state_path = STATES_DIR / f"{run_id}.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            mgr = cls(run_id, state["track"], state["tasks"], state.get("hyperparams"))
            mgr.start_ts = state.get("start_ts")
            print(f"[RunManager] Loaded state: {state_path} ({mgr.count_by_status()})")
            return mgr

        # Initialize from ARC data
        all_tasks = A.load_tasks(data_dir)
        task_ids = sorted(all_tasks)[:n_tasks]
        tasks_list = []
        for seq, tid in enumerate(task_ids, start=1):
            task = all_tasks[tid]
            demos = [{"input": ex["input"], "output": ex["output"]}
                     for ex in task.get("train", []) if "input" in ex and "output" in ex]
            test_input = task.get("test", [{}])[0].get("input")
            test_answer = task.get("test", [{}])[0].get("output")
            if demos and test_input is not None and test_answer is not None:
                tasks_list.append(_task_entry(seq, tid, demos, test_input, test_answer))

        import datetime
        mgr = cls(run_id, track, tasks_list)
        mgr.start_ts = datetime.datetime.utcnow().isoformat() + "Z"
        mgr.save()
        print(f"[RunManager] Initialized: {len(tasks_list)} tasks for run {run_id}")
        return mgr

    def save(self):
        STATES_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "run_id": self.run_id,
            "track": self.track,
            "start_ts": self.start_ts,
            "hyperparams": self.hyperparams,
            "tasks": self.tasks,
        }
        with open(STATES_DIR / f"{self.run_id}.json", "w") as f:
            json.dump(state, f, indent=2)

    def count_by_status(self):
        counts = {}
        for t in self.tasks:
            counts[t["status"]] = counts.get(t["status"], 0) + 1
        return counts

    def pending_tasks(self):
        return [t for t in self.tasks if t["status"] == "pending"]

    def sent_tasks(self):
        return [t for t in self.tasks if t["status"] == "sent"]

    def done_tasks(self):
        return [t for t in self.tasks if t["status"] in ("done", "error", "timeout")]

    def is_complete(self):
        return all(t["status"] in ("done", "error", "timeout") for t in self.tasks)

    def send_burst(self, doh_send_fn, n=None):
        """Send next N pending tasks as DMs to Prism. doh_send_fn(to, subject, body) -> msg_id."""
        n = n or self.BURST_LIMIT
        pending = self.pending_tasks()[:n]
        if not pending:
            print("[RunManager] No pending tasks to send.")
            return 0

        sent = 0
        for t in pending:
            body, _ = build_request_dm(
                run_id=self.run_id,
                seq=t["seq"],
                track=self.track,
                demos=t["demos"],
                test_input=t["test_input"],
                n_candidates=self.hyperparams.get("n_candidates", 8),
            )
            subject = f"[ARC-SYNTH] v1 run={self.run_id} seq={t['seq']:04d} track={self.track}"
            msg_id = doh_send_fn("A004", subject, body)
            t["status"] = "sent"
            t["dm_request_id"] = msg_id
            sent += 1
            print(f"  [send] seq={t['seq']:04d} task={t['task_id']} dm={msg_id}")

        self.save()
        print(f"[RunManager] Sent {sent} task DMs. Pending: {len(self.pending_tasks())}")
        return sent

    def collect_replies(self, doh_list_messages_fn):
        """Fetch A004 reply DMs, match to sent tasks, verify, record. Returns n_collected."""
        # Build lookup: (run_id, seq) -> task entry
        sent_lookup = {(t["seq"],): t for t in self.sent_tasks()}
        if not sent_lookup:
            print("[RunManager] No sent tasks awaiting replies.")
            return 0

        # Fetch recent messages from A004
        msgs = doh_list_messages_fn(from_agent="A004", unread_for="A003")
        if not msgs:
            print("[RunManager] No new replies from A004.")
            return 0

        collected = 0
        for msg in msgs:
            subj = msg.get("subject", "")
            if "[ARC-SYNTH-REPLY]" not in subj:
                continue
            # Parse run_id and seq from subject
            try:
                parts = {kv.split("=")[0].strip(): kv.split("=")[1].strip()
                         for kv in subj.split("v1")[1].split() if "=" in kv}
                reply_run_id = parts.get("run")
                reply_seq = int(parts.get("seq", 0))
            except (IndexError, ValueError, AttributeError):
                continue

            if reply_run_id != self.run_id:
                continue

            # Find matching task
            task_entry = next((t for t in self.sent_tasks() if t["seq"] == reply_seq), None)
            if task_entry is None:
                continue

            # Read the body
            body_path = msg.get("body_path", "")
            # Resolve path relative to team docs
            body_full = Path("/data/doh/teams/researchy/agents/A003") / body_path
            if not body_full.exists():
                body_full = Path("/data/doh/teams/researchy") / body_path
            try:
                body_text = body_full.read_text()
            except Exception as e:
                print(f"  [collect] seq={reply_seq} body read failed: {e}")
                task_entry["status"] = "error"
                continue

            try:
                data = parse_reply_dm(body_text)
            except Exception as e:
                print(f"  [collect] seq={reply_seq} parse failed: {e}")
                task_entry["status"] = "error"
                continue

            t0 = time.time()
            candidates_raw = data.get("candidates", [])
            candidates_verified = verify_candidates(candidates_raw, task_entry["demos"])
            pred, _ = vote(candidates_verified, task_entry["test_input"])
            acc = 1.0 if pred is not None and grids_equal(pred, task_entry["test_answer"]) else 0.0
            latency_ms = int((time.time() - t0) * 1000)

            import datetime
            ts = datetime.datetime.utcnow().isoformat() + "Z"
            verify_result = {
                "n_candidates_received": len(candidates_raw),
                "n_candidates_verified": len(candidates_verified),
                "passed": len(candidates_verified) > 0,
                "acc": acc,
            }
            write_call_record(
                run_id=self.run_id,
                seq=reply_seq,
                track=self.track,
                task_hash_val=task_entry["task_hash"],
                demos=task_entry["demos"],
                test_input=task_entry["test_input"],
                candidates_raw=candidates_raw,
                candidates_verified=candidates_verified,
                verify_result=verify_result,
                tokens_in=data.get("tokens_est_in", 0),
                tokens_out=data.get("tokens_est_out", 0),
                latency_ms=latency_ms,
                dm_request_id=task_entry["dm_request_id"],
                dm_reply_id=msg.get("msg_id"),
                agent_id=data.get("agent_id", "A004"),
                model_id=data.get("model_id", "unknown"),
                status=data.get("status", "ok"),
                ts=ts,
            )

            task_entry["status"] = "done"
            task_entry["dm_reply_id"] = msg.get("msg_id")
            task_entry["acc"] = acc
            task_entry["n_verified"] = len(candidates_verified)
            collected += 1
            print(f"  [collect] seq={reply_seq:04d} task={task_entry['task_id']} acc={acc:.1f} "
                  f"verified={len(candidates_verified)}/{len(candidates_raw)}")

        self.save()
        print(f"[RunManager] Collected {collected} replies. Status: {self.count_by_status()}")
        return collected

    def write_final_manifest(self):
        """Write run manifest when eval is complete."""
        import datetime
        done = self.done_tasks()
        scored = [t for t in done if t["acc"] is not None]
        n_solved = sum(1 for t in scored if t["acc"] > 0.5)
        end_ts = datetime.datetime.utcnow().isoformat() + "Z"
        return write_run_manifest(
            run_id=self.run_id,
            track=self.track,
            eval_split="public",
            start_ts=self.start_ts or end_ts,
            end_ts=end_ts,
            n_tasks=len(self.tasks),
            n_calls=len(done),
            n_solved=n_solved,
            total_in=0,  # summed from call records
            total_out=0,
            hyperparams=self.hyperparams,
        )

    def print_status(self):
        counts = self.count_by_status()
        total = len(self.tasks)
        done = self.done_tasks()
        scored = [t for t in done if t["acc"] is not None]
        n_solved = sum(1 for t in scored if t["acc"] > 0.5)
        pct = 100 * n_solved / len(scored) if scored else 0
        print(f"[E89 Run {self.run_id}] {counts} | solved {n_solved}/{len(scored)} = {pct:.1f}%")
