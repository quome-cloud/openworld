"""Source-free ARC-AGI-3 sandbox (fair by construction).

The real `arc_agi` game runs in an ISOLATED worker process whose working dir holds the downloaded
game source. The agent imports only `SandboxGame` -- a thin client that talks to the worker over a
pipe and exposes ONLY {frame, levels, win, avail, done}. The agent's process never holds the game
object (so `inspect.getsource(env._game)` has no target) and its working dir has no game source (so
`importlib`-loading `<gid>.py` has nothing to load). => the agent must discover dynamics by acting
and reason the win from observed frames, which is the real ARC-AGI-3 task.

  worker (needs arc_agi; run with the arc venv):   python arc3_sandbox.py --worker <gid>
  client (agent side; plain python, NO arc_agi):   from arc3_sandbox import SandboxGame
"""
import sys, os, json, subprocess

# Durable arc venv; override with ARC_VENV, default in $HOME (survives across Claude sessions).
ARC_VENV = os.environ.get("ARC_VENV", os.path.expanduser("~/.arcv/bin/python"))
# Source downloads here -- a tree SEPARATE from any agent working dir.
WORKER_ROOT = "/Users/jim/Desktop/openworld/experiments/.sandbox_env"


def _worker(gid):
    import logging
    logging.disable(logging.CRITICAL)                 # silence arc_agi logging
    # Dedicate a clean channel for the JSON protocol; send ALL env stdout noise to the void so it
    # cannot corrupt the protocol. proto = the original stdout pipe; fd1 -> /dev/null.
    proto_fd = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    proto = os.fdopen(proto_fd, "w", buffering=1)

    def send(d):
        proto.write(json.dumps(d) + "\n"); proto.flush()

    os.makedirs(os.path.join(WORKER_ROOT, gid), exist_ok=True)
    os.chdir(os.path.join(WORKER_ROOT, gid))          # arc_agi downloads source HERE, not in agent cwd
    import numpy as np, arc_agi
    from arcengine import GameAction
    A = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
         4: GameAction.ACTION4, 5: GameAction.ACTION5, 7: GameAction.ACTION7}
    arc = arc_agi.Arcade(); env = arc.make(gid)
    last = {"levels": 0, "win": 0}

    def obs(o):
        levels = int(getattr(o, "levels_completed", last["levels"]))
        win = int(getattr(o, "win_levels", last["win"]))
        last["levels"] = levels
        last["win"] = win
        done = str(getattr(o, "state", "")) != "GameState.NOT_FINISHED"
        if o is None or getattr(o, "frame", None) is None:
            return {"frame": None, "levels": levels, "win": win, "avail": [], "done": True}
        f = np.asarray(o.frame)
        if f.size == 0:
            return {"frame": None, "levels": levels, "win": win, "avail": [], "done": True}
        if f.ndim == 3:
            f = f[-1]
        if f.size != 64 * 64:
            return {"frame": None, "levels": levels, "win": win, "avail": [], "done": True}
        f = f.reshape(64, 64)
        return {"frame": f.astype(int).tolist(), "levels": levels, "win": win,
                "avail": list(getattr(o, "available_actions", [])), "done": done}

    env.reset()
    send({"ready": True})
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
    """Agent-side client: holds ONLY a pipe to the worker -- no game object, no source on its fs."""

    def __init__(self, gid, venv=ARC_VENV):
        import numpy as np
        self._np = np
        self.gid = gid
        self.p = subprocess.Popen([venv, os.path.abspath(__file__), "--worker", gid],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        while True:                                   # wait for the worker to finish arc.make()
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("worker died before ready")
            if json.loads(line).get("ready"):
                break
        self.reset()

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
        print("run as:  python arc3_sandbox.py --worker <gid>   (worker)\n"
              "or import SandboxGame in the agent process (client).")
