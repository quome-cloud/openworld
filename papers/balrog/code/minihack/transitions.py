"""Replayable transition logging for the source-blind induction leg.

Every scored or exploratory episode logs (obs_t, action, obs_{t+1}, reward,
done, info) with observations EXACTLY as served by the BALROG wrapper stack
(the same dict an agent receives). Stored as gzip JSON, one file per
episode: results/transitions/<label>/<task>__ep<k>.json.gz

Format:
{
  "task", "episode", "seed", "condition", "policy",
  "action_space": [...],
  "obs": [obs_0, obs_1, ..., obs_T],        # T+1 entries
  "steps": [{"action", "reward", "done", "info"}, ...]   # T entries
}
obs_i = {"glyphs": [21*79 ints], "blstats": [...], "tty_chars": [24 strings],
         "tty_colors": [24*80 ints], "tty_cursor": [y, x],
         "inv_letters": "abc...", "inv_strs": [...], "message": str,
         "text_long": str, "text_short": str}
"""

import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSDIR = os.path.join(HERE, "results", "transitions")


def serialize_obs(obs):
    raw = obs["obs"]
    inv_letters = "".join(chr(c) for c in raw["inv_letters"] if c != 0)
    inv_strs = []
    for i in range(len(inv_letters)):
        inv_strs.append(bytes(raw["inv_strs"][i]).partition(b"\x00")[0]
                        .decode("latin-1"))
    return {
        "glyphs": [int(v) for row in raw["glyphs"] for v in row],
        "blstats": [int(v) for v in raw["blstats"]],
        "tty_chars": ["".join(chr(c) for c in row) for row in raw["tty_chars"]],
        "tty_colors": [int(v) for row in raw["tty_colors"] for v in row],
        "tty_cursor": [int(v) for v in raw["tty_cursor"]],
        "inv_letters": inv_letters,
        "inv_strs": inv_strs,
        "text_long": obs["text"]["long_term_context"],
        "text_short": obs["text"]["short_term_context"],
    }


class TransitionLogger:
    def __init__(self, task, episode, seed, condition, policy, action_space,
                 label):
        self.dir = os.path.join(TRANSDIR, label)
        os.makedirs(self.dir, exist_ok=True)
        self.doc = {"task": task, "episode": episode, "seed": seed,
                    "condition": condition, "policy": policy,
                    "action_space": list(action_space),
                    "obs": [], "steps": []}
        self.fn = os.path.join(self.dir, f"{task}__ep{episode}.json.gz")

    def log_reset(self, obs):
        self.doc["obs"].append(serialize_obs(obs))

    def log_step(self, action, obs, reward, done, info):
        self.doc["obs"].append(serialize_obs(obs))
        self.doc["steps"].append({
            "action": action,
            "reward": round(float(reward), 5),
            "done": bool(done),
            "info": {k: str(v) for k, v in (info or {}).items()},
        })

    def close(self):
        with gzip.open(self.fn, "wt") as f:
            json.dump(self.doc, f)
        return self.fn
