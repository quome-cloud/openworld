"""Streaming transition logging for the source-blind induction leg.

NetHack episodes run to 10^4-10^5 steps, so unlike the MiniHack arm we
stream gzip JSONL instead of holding the episode in RAM:

results/transitions/<label>/<tag>__ep<k>.jsonl.gz
  line 0: header {task, episode, seed, condition, policy, action_space}
  line 1: {"obs": obs_0}
  line i: {"action", "reward", "done", "info", "obs": obs_i}

obs entries hold the served observation verbatim (glyphs/blstats/tty/inv/
misc/message + the wrapper's rendered text). To keep files tractable the
glyph/tty planes are delta-encoded against the previous step (full frame
every 100 steps); the decoder in this module restores exact frames.
"""

import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSDIR = os.path.join(HERE, "results", "transitions")


def _full_obs(raw, obs):
    inv_letters = "".join(chr(c) for c in raw["inv_letters"] if c != 0)
    inv_strs = []
    for i in range(len(inv_letters)):
        inv_strs.append(bytes(raw["inv_strs"][i]).partition(b"\x00")[0]
                        .decode("latin-1"))
    return {
        "glyphs": [int(v) for row in raw["glyphs"] for v in row],
        "tty_chars": ["".join(chr(c) for c in row) for row in raw["tty_chars"]],
        "tty_colors": [int(v) for row in raw["tty_colors"] for v in row],
        "blstats": [int(v) for v in raw["blstats"]],
        "tty_cursor": [int(v) for v in raw["tty_cursor"]],
        "misc": [int(v) for v in raw["misc"]],
        "message": raw.get("text_message", ""),
        "inv_letters": inv_letters,
        "inv_strs": inv_strs,
        "text_long": obs["text"]["long_term_context"],
        "text_short": obs["text"]["short_term_context"],
    }


class TransitionLogger:
    FULL_EVERY = 100

    def __init__(self, tag, episode, seed, condition, policy, action_space,
                 label):
        d = os.path.join(TRANSDIR, label)
        os.makedirs(d, exist_ok=True)
        self.fn = os.path.join(d, f"{tag}__ep{episode}.jsonl.gz")
        self.f = gzip.open(self.fn, "wt")
        self.f.write(json.dumps({
            "task": tag, "episode": episode, "seed": seed,
            "condition": condition, "policy": policy,
            "action_space": list(action_space)}) + "\n")
        self._prev_glyphs = None
        self._prev_tty = None
        self._prev_colors = None
        self.n = 0

    def _encode_obs(self, obs):
        raw = obs["obs"]
        full = _full_obs(raw, obs)
        out = dict(full)
        if self._prev_glyphs is not None and self.n % self.FULL_EVERY != 0:
            out["glyphs_delta"] = [
                [i, v] for i, v in enumerate(full["glyphs"])
                if v != self._prev_glyphs[i]]
            del out["glyphs"]
            out["tty_delta"] = [
                [i, r] for i, r in enumerate(full["tty_chars"])
                if r != self._prev_tty[i]]
            del out["tty_chars"]
            out["colors_delta"] = [
                [i, v] for i, v in enumerate(full["tty_colors"])
                if v != self._prev_colors[i]]
            del out["tty_colors"]
        self._prev_glyphs = full["glyphs"]
        self._prev_tty = full["tty_chars"]
        self._prev_colors = full["tty_colors"]
        return out

    def log_reset(self, obs):
        self.f.write(json.dumps({"obs": self._encode_obs(obs)}) + "\n")
        self.n += 1

    def log_step(self, action, obs, reward, done, info):
        self.f.write(json.dumps({
            "action": action,
            "reward": round(float(reward), 5),
            "done": bool(done),
            "info": {k: str(v) for k, v in (info or {}).items()},
            "obs": self._encode_obs(obs)}) + "\n")
        self.n += 1

    def close(self):
        self.f.close()
        return self.fn


def read_episode(fn):
    """Decoder: yields (header) then fully-restored step dicts."""
    prev_g = prev_t = prev_c = None
    with gzip.open(fn, "rt") as f:
        header = json.loads(f.readline())
        yield header
        for line in f:
            rec = json.loads(line)
            o = rec["obs"]
            if "glyphs" not in o:
                g = list(prev_g)
                for i, v in o.pop("glyphs_delta"):
                    g[i] = v
                o["glyphs"] = g
                t = list(prev_t)
                for i, r in o.pop("tty_delta"):
                    t[i] = r
                o["tty_chars"] = t
                c = list(prev_c)
                for i, v in o.pop("colors_delta"):
                    c[i] = v
                o["tty_colors"] = c
            prev_g, prev_t, prev_c = o["glyphs"], o["tty_chars"], o["tty_colors"]
            yield rec
