"""Snapshot the ARC-AGI-3 engine version hash of every game -> experiments/results/arc3_env_versions.json.

Each game's dynamics live in environment_files/<game>/<hash>/<game>.py; the <hash> is the engine's
version of that game and can change if the benchmark reversions/regenerates it upstream. We record the
hash per run (capture_lib.env_version, in run meta) AND keep this baseline map, so a later "did the game
change?" question is answered by a hash diff rather than memory. HARNESS METADATA ONLY -- reads the
directory NAME, never the game source; source-free is preserved.

  python scripts/snapshot_env_versions.py
"""
import os, sys, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import capture_lib as c

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES = ("ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 "
         "sp80 su15 tn36 tr87 tu93 vc33 wa30").split()


def main():
    vers = {g: c.env_version(g, root=ROOT) for g in GAMES}
    missing = [g for g, v in vers.items() if v is None]
    payload = {
        "note": "ARC-AGI-3 engine version hash per game (environment_files/<game>/<hash>). Diff a later "
                "snapshot against this to detect a game being reversioned/regenerated upstream. Harness "
                "metadata -- the hash is a directory name, not game source; source-free is preserved.",
        "captured_at": c.iso_now(),
        "env_versions": vers,
        "resolved": sum(1 for v in vers.values() if v),
        "missing": missing,
    }
    out = os.path.join(ROOT, "experiments/results/arc3_env_versions.json")
    json.dump(payload, open(out, "w"), indent=1)
    print(f"wrote {out}: {payload['resolved']}/25 resolved" + (f", missing {missing}" if missing else ""))


if __name__ == "__main__":
    main()
