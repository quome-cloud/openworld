"""Re-verify every banked E119 solve: replay its action sequence on a FRESH env and confirm a
level completes. The deterministic anchor — a banked solve is genuine regardless of how the
(stochastic) SLM arm reached it."""
import json, pathlib
from e119 import planner


def reverify_solves(logdir, make):
    d = pathlib.Path(logdir); ok = 0; n = 0; fail = []
    for f in sorted(d.glob("*_solved.json")):
        rec = json.loads(f.read_text())
        gid = rec["game"]; actions = [tuple(a) for a in rec["actions"]]
        n += 1
        try:
            reached, _ = planner.replay_levels(make(gid), actions)
            if reached >= 1:
                ok += 1
            else:
                fail.append(gid)
        except Exception:
            fail.append(gid)
    return {"ok": ok, "n": n, "fail": fail}
