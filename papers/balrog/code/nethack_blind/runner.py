"""Episode runner with predict-before-observe verification loop and JSONL.gz logging."""

import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from blind_env import make_env, blstats_dict  # noqa: E402
from world_model import WorldModel  # noqa: E402

RESULTS = os.path.join(HERE, "results")
TRANS = os.path.join(RESULTS, "transitions")


def tty_to_str(tty_chars):
    return "\n".join(bytes(row).decode("latin-1") for row in tty_chars)


def view(obs):
    raw = obs["obs"]
    return {
        "bl": blstats_dict(raw["blstats"]),
        "tty": raw["tty_chars"],
        "cursor": [int(raw["tty_cursor"][0]), int(raw["tty_cursor"][1])],
        "msg": raw.get("text_message", b"")
        if isinstance(raw.get("text_message"), str)
        else str(raw.get("text_message", "")),
        "text_long": obs["text"]["long_term_context"],
        "text_short": obs["text"]["short_term_context"],
    }


def run_episode(ep_id, policy, seed=None, max_steps=100000, frame_every=0,
                log_name=None, model=None, quiet=True):
    """frame_every: 0 = frames only at start/end + msg events; N = every N steps."""
    env = make_env(seed=seed)
    obs, info = env.reset(seed=seed)
    if model is None:
        model = WorldModel()
    anomalies = []
    n_pred = 0
    n_viol = 0

    log_path = os.path.join(TRANS, (log_name or f"ep{ep_id}") + ".jsonl.gz")
    f = gzip.open(log_path, "wt")

    pre = view(obs)
    f.write(json.dumps({"t": 0, "kind": "reset", "seed": seed, "bl": pre["bl"],
                        "msg": pre["msg"], "frame": tty_to_str(pre["tty"])}) + "\n")
    policy.reset(pre)

    t = 0
    done = False
    ep_start = time.time()
    try:
        while not done and t < max_steps:
            action, why = policy.act(pre)
            pred = model.predict(pre, action)
            t += 1
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            post = view(obs)
            post["done"] = done
            results = model.verify(pred, post)
            model.record(results, ep_id, t, anomalies)
            n_pred += len(results)
            viols = [r for r in results if not r[2]]
            n_viol += len(viols)

            rec = {"t": t, "a": action, "why": why, "bl": post["bl"],
                   "msg": post["msg"], "rew": float(reward), "done": done}
            if viols:
                rec["viol"] = [[d, rid, det] for d, rid, ok, det in viols]
                rec["frame"] = tty_to_str(post["tty"])
            if frame_every and t % frame_every == 0:
                rec["frame"] = tty_to_str(post["tty"])
            f.write(json.dumps(rec) + "\n")
            policy.observe(pre, action, post, reward, done)
            pre = post
    finally:
        stats = env.get_stats()
        stats.pop("dlvl_list", None), stats.pop("xplvl_list", None)
        summary = {"t": t, "kind": "end", "stats": {k: v for k, v in stats.items()},
                   "frame": tty_to_str(pre["tty"]), "n_pred": n_pred, "n_viol": n_viol,
                   "wallclock": round(time.time() - ep_start, 1)}
        f.write(json.dumps(summary) + "\n")
        f.close()
        env.close()

    return {"ep": ep_id, "seed": seed, "steps": t, "stats": stats,
            "n_pred": n_pred, "n_viol": n_viol,
            "viol_rate": (n_viol / n_pred) if n_pred else None,
            "anomalies": anomalies,
            "wallclock": round(time.time() - ep_start, 1)}
