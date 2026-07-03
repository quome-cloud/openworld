"""Backfill the ARC-AGI-3 engine version hash into trace-run meta sidecars captured BEFORE per-run
env_version capture existed (PR #173).

Those older runs never observed the hash, so we JOIN it from the baseline snapshot
(experiments/results/arc3_env_versions.json) by game. This is honest only because ARC-AGI-3 games are
FIXED templates whose engine hash is a property of the game, not the run -- and we mark every backfilled
value with `env_version_source` so a reader can always tell an observed hash from a joined one.

  - Runs that already carry a non-null `env_version` (written by the runner) are left untouched.
  - Runs missing it get `env_version` = baseline[game] and `env_version_source` = the provenance string.
  - Idempotent: re-running never double-stamps and never overwrites an observed hash. Safe to re-run as
    the live arms append new meta files.
  - HARNESS METADATA ONLY -- the hash is a directory name (engine version), never game source; the
    source-free guarantee is untouched.

  python scripts/backfill_env_version.py            # stamp in place
  python scripts/backfill_env_version.py --dry-run  # report only, write nothing
"""
import os, sys, json, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(ROOT, "experiments/results/arc3_env_versions.json")
META_GLOB = os.path.join(ROOT, "experiments/results/arc3_traces/meta/*.json")


def load_baseline():
    d = json.load(open(BASELINE))
    return d["env_versions"], d.get("captured_at", "unknown")


def reorder(rec, hashval, source):
    """Insert env_version + env_version_source right after dataset_version (where the runner puts the
    hash), preserving all other key order so the diff stays minimal."""
    out = {}
    for k, v in rec.items():
        if k in ("env_version", "env_version_source"):
            continue                      # drop any stale copy; we re-place it canonically
        out[k] = v
        if k == "dataset_version":
            out["env_version"] = hashval
            out["env_version_source"] = source
    if "env_version" not in out:          # meta lacked dataset_version -> append at end
        out["env_version"] = hashval
        out["env_version_source"] = source
    return out


def main():
    dry = "--dry-run" in sys.argv[1:]
    versions, captured_at = load_baseline()
    source = f"backfill:arc3_env_versions.json (baseline {captured_at})"
    metas = sorted(glob.glob(META_GLOB))
    stamped = skipped_observed = no_game = 0
    for m in metas:
        try:
            rec = json.load(open(m))
        except Exception as e:
            print(f"  !! unreadable {os.path.basename(m)}: {e}")
            continue
        if rec.get("env_version"):        # already has a hash (observed OR previously backfilled)
            skipped_observed += 1
            continue
        game = rec.get("game")
        hashval = versions.get(game)
        if not hashval:
            no_game += 1
            print(f"  ?? no baseline hash for game={game} ({os.path.basename(m)})")
            continue
        if not dry:
            json.dump(reorder(rec, hashval, source), open(m, "w"), indent=1)
        stamped += 1
    print(f"meta files: {len(metas)}")
    print(f"  stamped (backfilled): {stamped}" + ("  [DRY-RUN, not written]" if dry else ""))
    print(f"  left untouched (already had env_version): {skipped_observed}")
    if no_game:
        print(f"  UNRESOLVED (game not in baseline): {no_game}")


if __name__ == "__main__":
    main()
