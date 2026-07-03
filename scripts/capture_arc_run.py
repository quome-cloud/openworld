"""Capture one completed ARC-3 agent run dir (fl_ focused-final-level, su_ SHU-cycle, ...) into the
HF-ready dataset experiments/results/arc3_traces/ by REUSING scripts/capture_lib.py. Callable by a
runner at finish, or in a backfill loop over existing run dirs. Stdlib-only (any python works).

  capture_arc_run.py <game> <workdir> [tier] [runner_script]
"""
import sys, os, json, shutil
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture_lib as C


def _mtime_iso(p):
    return datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture(game, wd, tier="focused-final-level", runner="run_arc_agent_final_level.sh"):
    wd = os.path.abspath(wd)
    log = os.path.join(wd, "agent.log")
    task = os.path.join(wd, "TASK.md")
    solp = os.path.join(wd, "solved.json")
    has_log = os.path.exists(log)                              # LLM runs (fl_) have a stream-json log
    sol = json.load(open(solp)) if os.path.exists(solp) else {}
    if not has_log and not sol:                               # nothing produced -> nothing to capture
        print(f"[capture] {game}: no agent.log and no solved.json in {wd} -- skip", flush=True)
        return None
    anchor = task if os.path.exists(task) else (log if has_log else solp)
    started = _mtime_iso(anchor)
    ended = _mtime_iso(log if has_log else solp)
    rid = f"{game}__{tier}__{started.replace(':', '-')}"
    C.ensure_dirs()
    prompt = open(task).read() if os.path.exists(task) else ""
    if prompt:
        (C.PROMPTS / f"{rid}.md").write_text(prompt)
    if has_log:
        shutil.copy(log, C.TRANSCRIPTS / f"{rid}.jsonl")      # large; gitignored -> HF/object storage
    if sol:
        (C.SOLUTIONS / f"{rid}.json").write_text(json.dumps(sol))
    summary = C.summarize_transcript(log) if has_log else None     # deterministic runs have no transcript
    solver_meta = {}                                          # cycle telemetry for deterministic (su_) runs
    mp = os.path.join(wd, "run_meta.json")
    if os.path.exists(mp):
        try:
            solver_meta = json.load(open(mp))
        except Exception:
            pass
    actions = sol.get("actions") or []
    record = {
        "run_id": rid, "game": game, "tier": tier, "method": tier,
        "source_free": True, "fairness": "source-free (process-isolated SandboxGame; audit-gated)",
        "runner": runner, "started_at": started, "ended_at": ended,
        "dataset_version": C.DATASET_VERSION, "benchmark": C.BENCHMARK,
        "outcome": {"levels": sol.get("levels"), "win": sol.get("win"),
                    "full": bool(sol.get("win") and sol.get("levels", 0) >= sol.get("win", 0)),
                    "wrote_solution": bool(sol)},
        "action_stats": C.action_stats(actions),
        "transcript_summary": summary,
        "solver_meta": solver_meta or None,                  # deterministic-solver telemetry (seed/cycles/...)
        "model_config": (C.model_config(requested_model="claude-opus-4-8", effort="high", summary=summary)
                         if has_log else {"solver": solver_meta.get("solver", "deterministic"),
                                          "requested_model": None, "resolved_model": None,
                                          "seed": solver_meta.get("seed")}),
        "prompt_stats": C.prompt_stats(prompt) if prompt else None,
        "prompt_file": f"prompts/{rid}.md" if prompt else None,
        "solution_file": f"solutions/{rid}.json" if sol else None,
        "transcript_file": (f"transcripts/{rid}.jsonl" if has_log else None),
        "provenance": {"host": C.host_info(), "git": C.git_info(),
                       "scripts": C.file_provenance([os.path.join(C.ROOT, "scripts", runner)])},
        "captured_at": C.iso_now(),
    }
    C.write_meta(rid, record)
    n = C.append_run(record)
    print(f"[capture] {game}: {tier} -> runs.jsonl ({n} rows) | "
          f"levels {sol.get('levels')}/{sol.get('win')} | rid {rid}", flush=True)
    return rid


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: capture_arc_run.py <game> <workdir> [tier] [runner]")
        sys.exit(1)
    capture(sys.argv[1], sys.argv[2],
            sys.argv[3] if len(sys.argv) > 3 else "focused-final-level",
            sys.argv[4] if len(sys.argv) > 4 else "run_arc_agent_final_level.sh")
