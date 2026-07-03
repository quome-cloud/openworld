"""Shared helpers for capturing ARC-AGI-3 source-free solving runs as a reproducible, HuggingFace-ready
dataset (prompts + full structured transcripts + timestamps + verified outcomes).

The dataset lives under experiments/results/arc3_traces/:
  runs.jsonl            one record per run (metadata + prompt + outcome + audit + verification); COMMITTED.
  prompts/<rid>.md      the exact prompt given to the agent (small; COMMITTED for reproducibility).
  transcripts/<rid>.jsonl  the raw claude -p stream-json transcript (LARGE; gitignored -> HF/object storage).
  meta/<rid>.json       per-run sidecar written at launch time (game, tier, timestamps, paths); COMMITTED.

Every record is timestamped in UTC ISO-8601 so the dataset is auditable and orderable. Deterministic
(cheap) runs have no prompt/transcript; their record carries the solver name + seed instead.
"""
import json, os, hashlib, glob
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/jim/Desktop/openworld")
TRACES = ROOT / "experiments" / "results" / "arc3_traces"
PROMPTS = TRACES / "prompts"
TRANSCRIPTS = TRACES / "transcripts"
META = TRACES / "meta"
SOLUTIONS = TRACES / "solutions"          # per-run action-trace snapshot (the thing the run produced)
RUNS = TRACES / "runs.jsonl"

DATASET_VERSION = "1.0"
BENCHMARK = {"name": "ARC-AGI-3", "grid": "64x64", "colors": 16,
             "actions": "directional 1-5,7 + click ACTION6(x,y)"}


def iso_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ts_slug():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def run_id(game, tier):
    return f"{game}__{tier}__{ts_slug()}"


def ensure_dirs():
    for d in (TRACES, PROMPTS, TRANSCRIPTS, META, SOLUTIONS):
        d.mkdir(parents=True, exist_ok=True)


def sha256_file(p):
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def write_meta(rid, record):
    """Sidecar written at launch (and updated at finish). One source of truth per run."""
    ensure_dirs()
    (META / f"{rid}.json").write_text(json.dumps(record, indent=1))


def read_meta(rid):
    p = META / f"{rid}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def summarize_transcript(jsonl_path):
    """Pull reproducibility metadata out of a claude -p stream-json transcript.
    Returns {model, claude_code_version, session_id, num_turns, n_messages, n_tool_calls,
             cost_usd, usage, is_error, result_text, started_hint}."""
    p = Path(jsonl_path)
    out = {"model": None, "claude_code_version": None, "session_id": None, "num_turns": None,
           "n_messages": 0, "n_tool_calls": 0, "cost_usd": None, "usage": None,
           "is_error": None, "result_text": None, "fast_mode_state": None, "permission_mode": None}
    if not p.exists():
        return out
    for ln in open(p, errors="ignore"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
        except Exception:
            continue
        out["n_messages"] += 1
        t = o.get("type")
        if t == "system" and o.get("subtype") == "init":
            out["model"] = o.get("model")                       # RESOLVED model id, e.g. claude-opus-4-8[1m]
            out["claude_code_version"] = o.get("claude_code_version")
            out["fast_mode_state"] = o.get("fast_mode_state")
            out["permission_mode"] = o.get("permissionMode")
            out["session_id"] = out["session_id"] or o.get("session_id")
        elif t == "assistant":
            msg = o.get("message", {})
            u = msg.get("usage") or {}
            if u:                                              # aggregate per-message usage (each = 1 API call)
                a = out.setdefault("_agg_usage", {"input_tokens": 0, "output_tokens": 0,
                                                  "cache_creation_input_tokens": 0,
                                                  "cache_read_input_tokens": 0, "n_calls": 0})
                a["input_tokens"] += u.get("input_tokens", 0) or 0
                a["output_tokens"] += u.get("output_tokens", 0) or 0
                a["cache_creation_input_tokens"] += u.get("cache_creation_input_tokens", 0) or 0
                a["cache_read_input_tokens"] += u.get("cache_read_input_tokens", 0) or 0
                a["n_calls"] += 1
            for blk in (msg.get("content") or []):
                if isinstance(blk, dict) and blk.get("type") == "tool_use":
                    out["n_tool_calls"] += 1
                    name = blk.get("name", "?")
                    out.setdefault("tool_calls_by_name", {})
                    out["tool_calls_by_name"][name] = out["tool_calls_by_name"].get(name, 0) + 1
                elif isinstance(blk, dict) and blk.get("type") == "text":
                    out["n_text_blocks"] = out.get("n_text_blocks", 0) + 1
                elif isinstance(blk, dict) and blk.get("type") == "thinking":
                    out["n_thinking_blocks"] = out.get("n_thinking_blocks", 0) + 1
        elif t == "user":
            out["n_user_msgs"] = out.get("n_user_msgs", 0) + 1
        elif t == "result":
            out["num_turns"] = o.get("num_turns")
            out["cost_usd"] = o.get("total_cost_usd")
            out["usage"] = o.get("usage")
            out["is_error"] = o.get("is_error")
            out["api_error_status"] = o.get("api_error_status")
            out["duration_ms"] = o.get("duration_ms")
            out["duration_api_ms"] = o.get("duration_api_ms")
            out["ttft_ms"] = o.get("ttft_ms")
            out["result_text"] = str(o.get("result"))[:2000] if o.get("result") is not None else None
            out["session_id"] = out["session_id"] or o.get("session_id")
    out.setdefault("tool_calls_by_name", {})
    # Token usage: prefer the authoritative result-block usage; else reconstruct from per-message usage
    # (for runs cut mid-stream that never emitted a result block). Always record the source.
    u = out.get("usage") or {}
    agg = out.pop("_agg_usage", None)
    if u:
        src = "result_block"; tu = u
    elif agg:
        src = "reconstructed_from_messages"; tu = agg
        out["usage"] = agg
    else:
        src = "none"; tu = {}
    out["tokens"] = {
        "input": tu.get("input_tokens"), "output": tu.get("output_tokens"),
        "cache_creation": tu.get("cache_creation_input_tokens"),
        "cache_read": tu.get("cache_read_input_tokens"),
        "total": (None if not tu else sum(v for k, v in tu.items()
                  if k.endswith("_tokens") and isinstance(v, int))),
        "source": src,
    }
    # Cost: authoritative if the result block reported it; else estimate from tokens at recorded rates.
    out["cost_basis"] = "result_block" if out.get("cost_usd") is not None else (
        "estimated_from_tokens" if tu else "unknown")
    if out.get("cost_usd") is None and tu:
        r = PRICING_OPUS_PER_MTOK
        out["cost_usd_estimated"] = round(
            (tu.get("input_tokens", 0) * r["input"]
             + tu.get("output_tokens", 0) * r["output"]
             + tu.get("cache_creation_input_tokens", 0) * r["cache_write"]
             + tu.get("cache_read_input_tokens", 0) * r["cache_read"]) / 1_000_000, 4)
        out["pricing_assumed"] = r
    return out


# Assumed Anthropic Opus pricing (USD per million tokens) for ESTIMATED costs on runs that lack an
# authoritative result block. Recorded in the record so estimates are transparent and correctable.
PRICING_OPUS_PER_MTOK = {"model": "claude-opus-4-x", "input": 15.0, "output": 75.0,
                         "cache_write": 18.75, "cache_read": 1.5}


# ----- provenance / environment / stats helpers (maximal metadata for the dataset) -----

def host_info():
    import platform as _pf
    return {"hostname": _pf.node(), "platform": _pf.platform(), "system": _pf.system(),
            "machine": _pf.machine(), "python": _pf.python_version()}


def git_info(root=ROOT):
    import subprocess
    def _q(args, default=""):
        try:
            return subprocess.run(["git", "-C", str(root)] + args, capture_output=True,
                                  text=True, timeout=10).stdout.strip() or default
        except Exception:
            return default
    dirty = bool(_q(["status", "--porcelain"]))
    return {"commit": _q(["rev-parse", "HEAD"]), "branch": _q(["rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": dirty, "remote": _q(["config", "--get", "remote.origin.url"])}


def env_version(game, root=ROOT):
    """The ARC-AGI-3 engine version hash for `game` -- the `environment_files/<game>/<hash>/` directory
    name the engine loaded this run. HARNESS METADATA ONLY: reads the directory NAME, never the game
    source (source-free is preserved). Recording it per run lets a later 'did the game change?' question
    be answered by a hash diff instead of memory (the env source is regenerable/reversionable upstream)."""
    import glob
    r = str(root)
    for base in (os.path.join(r, "experiments", ".sandbox_env", game, "environment_files", game),
                 os.path.join(r, "experiments", "environment_files", game),
                 os.path.join(r, "environment_files", game)):
        dirs = sorted(d for d in glob.glob(base + "/*/") if os.path.isdir(d))
        if dirs:
            return os.path.basename(dirs[-1].rstrip("/"))
    dirs = sorted(glob.glob(os.path.join(r, "scratch_arc", "*", "environment_files", game, "*/")))
    return os.path.basename(dirs[-1].rstrip("/")) if dirs else None


def file_provenance(paths):
    """sha256 + size + mtime for the scripts that produced a run (pipeline reproducibility)."""
    out = {}
    for p in paths:
        p = Path(p)
        if p.exists():
            st = p.stat()
            out[p.name] = {"sha256": sha256_file(p), "bytes": st.st_size,
                           "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%SZ")}
    return out


def action_stats(actions):
    actions = actions or []
    n_click = sum(1 for a in actions if isinstance(a, (list, tuple)) and a and a[0] == 6)
    return {"n_actions": len(actions), "n_click": n_click, "n_directional": len(actions) - n_click}


def prompt_stats(text):
    return {"chars": len(text), "lines": text.count("\n") + 1, "sha256": sha256_text(text),
            "approx_tokens": max(1, len(text) // 4)}


def model_config(requested_model=None, effort=None, fallback_model=None, summary=None):
    """The (model, version, effort, fast-mode) tuple that isolates an artifact. `summary` is the
    summarize_transcript() output (for the RESOLVED model id + version + fast/permission mode)."""
    s = summary or {}
    return {"requested_model": requested_model, "resolved_model": s.get("model"),
            "effort": effort, "fallback_model": fallback_model,
            "claude_code_version": s.get("claude_code_version"),
            "fast_mode_state": s.get("fast_mode_state"),
            "permission_mode": s.get("permission_mode")}


def append_run(record):
    """Append (or replace by run_id) a record into runs.jsonl, kept sorted by started_at."""
    ensure_dirs()
    recs = {}
    if RUNS.exists():
        for ln in open(RUNS, errors="ignore"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                recs[r["run_id"]] = r
            except Exception:
                pass
    recs[record["run_id"]] = record
    ordered = sorted(recs.values(), key=lambda r: (r.get("started_at") or "", r.get("run_id")))
    with open(RUNS, "w") as f:
        for r in ordered:
            f.write(json.dumps(r) + "\n")
    return len(ordered)


# --- E124 codex telemetry (reproducibility) ---
def codex_record(traces_dir, rec):
    """Write one codex-call record (JSONL) + prompt/transcript sidecars. rec carries game/level/regime,
    model+version, prompt, raw response, events, parsed goal, decision, tainted, tokens, latency, timings."""
    import os, json, hashlib
    os.makedirs(os.path.join(traces_dir, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(traces_dir, "transcripts"), exist_ok=True)
    base = f"{rec.get('game','?')}__{rec.get('level',0)}_{rec.get('regime',0)}"
    h = hashlib.blake2b((base + str(rec.get('prompt',''))).encode(), digest_size=5).hexdigest()
    rid = f"{base}__{h}"
    open(os.path.join(traces_dir, "prompts", rid + ".txt"), "w").write(rec.get("prompt", ""))
    json.dump({"raw": rec.get("raw", ""), "events": rec.get("events", [])},
              open(os.path.join(traces_dir, "transcripts", rid + ".json"), "w"))
    line = {k: rec.get(k) for k in ("game", "level", "regime", "model", "model_version",
            "decision", "tainted", "tokens", "latency", "started", "finished")}
    line["run_id"] = rid
    line["parsed_summary"] = {"n_subgoals": len((rec.get("parsed") or {}).get("subgoals", [])),
                              "n_macros": len((rec.get("parsed") or {}).get("macros", []))}
    line["hash"] = h
    with open(os.path.join(traces_dir, "calls.jsonl"), "a") as fh:
        fh.write(json.dumps(line) + "\n")
    return rid
