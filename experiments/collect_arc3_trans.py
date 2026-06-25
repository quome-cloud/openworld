"""Collect (frame,action,next) transitions per ARC-3 game -> JSON (for E91 GRPO training).
Run in a py3.12 venv with arc-agi==0.9.9. Decouples collection (arc-agi) from training (torch)."""
import argparse, json, logging, contextlib, io
from pathlib import Path
import e86_arc3 as E

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/arc3_trans")
ap.add_argument("--steps", type=int, default=300)
a = ap.parse_args()
Path(a.out).mkdir(parents=True, exist_ok=True)
logging.disable(logging.CRITICAL)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import arc_agi
    envs = arc_agi.Arcade().available_environments
games = sorted({(e if isinstance(e, str) else getattr(e, "game_id", str(e))).split("-")[0] for e in envs})
for g in games:
    try:
        trans, _, _ = E.collect(g, a.steps, 0)
        if trans:
            json.dump(trans, open(f"{a.out}/{g}.json", "w"))
            print(f"{g}: {len(trans)} transitions", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"{g}: collect failed {ex}", flush=True)
