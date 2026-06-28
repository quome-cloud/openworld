# experiments/e127/sandbox.py
"""Source-free ARC-AGI-3 sandbox for E127 (vendored, self-contained). The real arc_agi game runs in
an isolated worker process whose cwd holds the downloaded source; the agent imports only SandboxGame
-- a pipe client exposing ONLY {frame, levels, win, avail, done}. The agent process never holds the
game object and its cwd has no source => discovery is by acting only (source-free by construction)."""
import sys, os, json, subprocess

ARC_VENV = os.environ.get("ARC_VENV", os.path.expanduser("~/.arcv/bin/python"))
WORKER_ROOT = os.path.join(os.path.dirname(__file__), ".sandbox_env")


def _worker(gid):
    import logging
    logging.disable(logging.CRITICAL)
    proto_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    proto = os.fdopen(proto_fd, "w", buffering=1)

    def send(d):
        proto.write(json.dumps(d) + "\n"); proto.flush()

    os.makedirs(os.path.join(WORKER_ROOT, gid), exist_ok=True)
    os.chdir(os.path.join(WORKER_ROOT, gid))
    import numpy as np, arc_agi
    from arcengine import GameAction
    A = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
         4: GameAction.ACTION4, 5: GameAction.ACTION5, 7: GameAction.ACTION7}
    arc = arc_agi.Arcade(); env = arc.make(gid)
    last = {"levels": 0, "win": 0}

    def obs(o):
        if o is None or getattr(o, "frame", None) is None:
            return {"frame": None, "levels": last["levels"], "win": last["win"], "avail": [], "done": True}
        f = np.asarray(o.frame); f = (f[-1] if f.ndim == 3 else f).reshape(64, 64)
        last["levels"] = int(o.levels_completed); last["win"] = int(o.win_levels)
        return {"frame": f.astype(int).tolist(), "levels": last["levels"], "win": last["win"],
                "avail": list(o.available_actions), "done": str(o.state) != "GameState.NOT_FINISHED"}

    env.reset(); send({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        try:
            c = req.get("cmd")
            if c == "reset":
                o = env.reset()
            elif c == "step":
                a = req["a"]
                o = (env.step(GameAction.ACTION6, {"x": int(req["x"]), "y": int(req["y"])})
                     if a == 6 else env.step(A[a]))
            elif c == "close":
                break
            else:
                raise ValueError("bad cmd")
            r = obs(o)
        except Exception as e:
            r = {"error": str(e)[:200]}
        send(r)


class SandboxGame:
    def __init__(self, gid, venv=ARC_VENV):
        import numpy as np
        self._np = np; self.gid = gid; self.frame = None
        self.p = subprocess.Popen([venv, os.path.abspath(__file__), "--worker", gid],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        try:
            while True:
                line = self.p.stdout.readline()
                if not line:
                    raise RuntimeError("worker died before ready")
                if json.loads(line).get("ready"):
                    break
            self.reset()
        except BaseException:
            # Reap the worker on ANY failure after Popen so a half-built
            # SandboxGame never orphans the arc_agi engine subprocess.
            self.p.terminate()
            try:
                self.p.wait(timeout=2)
            except Exception:
                pass
            raise

    def _rpc(self, req):
        self.p.stdin.write(json.dumps(req) + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("worker died")
        r = json.loads(line)
        if "error" in r:
            raise RuntimeError(r["error"])
        if r["frame"] is not None:
            self.frame = self._np.array(r["frame"])
        self.levels = r["levels"]; self.win = r["win"]; self.avail = r["avail"]; self.done = r["done"]
        return self.frame

    def reset(self):
        return self._rpc({"cmd": "reset"})

    def step(self, a, x=None, y=None):
        return self._rpc({"cmd": "step", "a": a, "x": x, "y": y})

    def close(self):
        try:
            self.p.stdin.write(json.dumps({"cmd": "close"}) + "\n"); self.p.stdin.flush()
        except Exception:
            pass
        self.p.terminate()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker(sys.argv[2])
    else:
        print("run as: python sandbox.py --worker <gid>")
