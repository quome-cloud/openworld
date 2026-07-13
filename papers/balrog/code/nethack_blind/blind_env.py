"""Source-blind env shim for the BALROG NetHackChallenge arm.

QUARANTINE NOTE (see FABLE_NETHACK_BLIND_REPORT.md):
This module touches ONLY the balrog wrapper interface:
  - balrog.environments.nle.nle_env.make_nle_env  (constructor)
  - env.language_action_space                      (served action strings)
  - env.reset()/env.step()/env.get_stats()         (served obs/metric)
It reads NO NetHack/NLE source. blstats field names come from
balrog/environments/nle/progress.py (metric definition, explicitly allowed).
"""

import os
import sys
from types import SimpleNamespace

FABLE_NH = "/data/doh/teams/researchy/work/fable_nethack"
sys.path.insert(0, os.path.join(FABLE_NH, "pylib"))
sys.path.insert(0, FABLE_NH)  # for `balrog` package

# blstats index -> name, from balrog/environments/nle/progress.py (allowed: metric def)
BLSTATS_NAMES = [
    "x_pos", "y_pos", "strength_percentage", "strength", "dexterity",
    "constitution", "intelligence", "wisdom", "charisma", "score",
    "hitpoints", "max_hitpoints", "depth", "gold", "energy", "max_energy",
    "armor_class", "monster_level", "experience_level", "experience_points",
    "time", "hunger_state", "carrying_capacity", "dungeon_number", "level_number",
]


def _make_config(max_episode_steps=100_000, no_progress_timeout=150, seed=None):
    """Mirror of the official BALROG config.yaml nle_kwargs (protocol, allowed)."""
    nle_kwargs = {
        "character": "@",
        "max_episode_steps": max_episode_steps,
        "no_progress_timeout": no_progress_timeout,
        "savedir": None,
        "save_ttyrec_every": 0,
        "skip_more": True,
    }
    return SimpleNamespace(
        envs=SimpleNamespace(nle_kwargs=nle_kwargs, env_kwargs={"seed": seed}),
        agent=SimpleNamespace(max_image_history=0),
    )


def make_env(seed=None, max_episode_steps=100_000, no_progress_timeout=150):
    from balrog.environments.nle.nle_env import make_nle_env

    cfg = _make_config(max_episode_steps, no_progress_timeout, seed)
    env = make_nle_env("nle", "NetHackChallenge-v0", cfg)
    return env


def blstats_dict(blstats):
    return {name: int(v) for name, v in zip(BLSTATS_NAMES, blstats)}


def obs_view(obs):
    """Extract the served pieces we log/use: text contexts + raw arrays we cite."""
    raw = obs["obs"]
    return {
        "blstats": blstats_dict(raw["blstats"]),
        "tty_chars": raw["tty_chars"],   # ndarray (24,80) uint8
        "tty_colors": raw.get("tty_colors"),
        "tty_cursor": raw["tty_cursor"],
        "message": obs["text"]["long_term_context"].split("\n", 2),  # unused; text kept in logs
        "text_long": obs["text"]["long_term_context"],
        "text_short": obs["text"]["short_term_context"],
    }


if __name__ == "__main__":
    env = make_env(seed=1)
    obs, info = env.reset(seed=1)
    print("obs keys:", list(obs.keys()))
    print("raw obs keys:", list(obs["obs"].keys()))
    acts = list(env.language_action_space)
    print("n_actions:", len(acts))
    print("actions:", acts[:120])
    v = obs_view(obs)
    print("blstats:", v["blstats"])
    print("stats:", env.get_stats())
    env.close()
