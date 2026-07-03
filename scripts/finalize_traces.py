"""Finalize the arc3_traces dataset: join each captured run's meta sidecar + its solution snapshot with a
freshly-computed VERIFIED outcome (source-free audit + real-engine replay + OpenWorld-World round-trip), and
write the complete record into runs.jsonl. Idempotent -- safe to re-run; it recomputes every outcome.

Run with the arc venv python (needs arc_agi + openworld):
    <arcv>/bin/python scripts/finalize_traces.py
"""
import json, sys, glob, os
from pathlib import Path

ROOT = Path("/Users/jim/Desktop/openworld")
sys.path.insert(0, str(ROOT / "scripts"))
import capture_lib as c
from audit_sandbox import audit, audit_files
from autobank_sourcefree import openworld_roundtrip   # E121 World round-trip (shared with the banker)


def verified_outcome(rec):
    """Recompute the run's outcome from its solution snapshot: audit + replay + OpenWorld round-trip."""
    sol_path = ROOT / "experiments" / "results" / "arc3_traces" / rec.get("solution_file", "")
    game = rec["game"]
    out = {"levels": 0, "win": 0, "full_solve": False, "audit": None,
           "replay_verified": False, "openworld_roundtrip": None, "actions": None}
    # source-free audit: AGENT runs audit a working dir (by-construction); CHEAP runs audit the fixed
    # solver script files (by-audit -- the algorithm provably reads only frames).
    mode = rec.get("audit_mode", "strict")
    if rec.get("audit_files"):
        paths = [str(ROOT / p) for p in rec["audit_files"]]
        findings = audit_files(paths, mode=mode)
        out["audit"] = {"mode": mode, "files": rec["audit_files"], "clean": (findings == []),
                        "findings": findings}
    else:
        adir = ROOT / rec.get("audit_dir", f"scratch_arc/sb_{game}")
        findings = audit(str(adir), mode=mode) if adir.exists() else ["audit_dir missing"]
        out["audit"] = {"mode": mode, "dir": rec.get("audit_dir"), "clean": (findings == []),
                        "findings": findings}
    if not sol_path.exists():
        return out
    try:
        d = json.loads(sol_path.read_text())
    except Exception as ex:
        out["audit"]["findings"].append(f"unreadable solution: {ex}")
        return out
    actions = d.get("actions") or []
    out["actions"] = actions
    out["win"] = int(d.get("win", 0))
    out["action_stats"] = c.action_stats(actions)
    if not actions:
        return out
    try:
        rt = openworld_roundtrip(game, actions)
        out["levels"] = int(rt.get("depth_real", 0))
        out["replay_verified"] = (rt.get("depth_real", 0) >= int(d.get("levels", 0)))
        out["openworld_roundtrip"] = rt
        out["full_solve"] = bool(out["win"] and out["levels"] >= out["win"]
                                 and out["audit"]["clean"] and rt.get("pass"))
    except Exception as ex:
        out["openworld_roundtrip"] = {"error": str(ex)[:200], "pass": False}
    return out


def main():
    c.ensure_dirs()
    force = "--force" in sys.argv
    # incremental: skip run_ids already finalized (immutable once the run is done), unless --force
    done = set()
    if c.RUNS.exists() and not force:
        for l in open(c.RUNS, errors="ignore"):
            if l.strip():
                try:
                    r = json.loads(l)
                    if r.get("outcome") is not None:
                        done.add(r["run_id"])
                except Exception:
                    pass
    metas = sorted(glob.glob(str(c.META / "*.json")))
    n = 0
    for mp in metas:
        try:
            rec = json.loads(open(mp).read())
        except Exception:
            continue
        if rec.get("run_id") in done:
            continue
        rec["outcome"] = verified_outcome(rec)
        rec["finalized_at"] = c.iso_now()
        c.append_run(rec)
        n += 1
        o = rec["outcome"]
        rt = o.get("openworld_roundtrip") or {}
        print(f"[finalize] {rec['run_id']}: {o['levels']}/{o['win']} "
              f"{'FULL' if o['full_solve'] else 'partial'} "
              f"audit={'clean' if o['audit']['clean'] else 'TAINT'} "
              f"world_pass={rt.get('pass')} "
              f"model={rec.get('model_config',{}).get('resolved_model')} "
              f"effort={rec.get('model_config',{}).get('effort')}", flush=True)
    # dataset-level summary
    runs = [json.loads(l) for l in open(c.RUNS) if l.strip()] if c.RUNS.exists() else []
    games = {}
    for r in runs:
        o = r.get("outcome") or {}
        g = r["game"]
        if g not in games or (o.get("levels", 0) > games[g]):
            games[g] = o.get("levels", 0)
    full = sum(1 for r in runs if (r.get("outcome") or {}).get("full_solve"))
    print(f"\n[finalize] {n} runs -> runs.jsonl ({len(runs)} total); "
          f"{full} full-solve runs across {len(games)} games", flush=True)


if __name__ == "__main__":
    main()
