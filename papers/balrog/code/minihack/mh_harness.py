"""BALROG-identical MiniHack env harness.

Builds the exact env stack BALROG uses for its MiniHack suite:
  gym.make(task, observation_keys=[...], **minihack_kwargs)
  -> AutoMore -> NLELanguageWrapper -> NLETimeLimit -> GymV21CompatibilityV0 -> EnvWrapper

Config values are taken verbatim from BALROG balrog/config/config.yaml
(minihack_kwargs + agent.max_image_history=0 + eval.num_episodes.minihack=5).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(_file_ := __file__))
sys.path.insert(0, os.path.join(_HERE, "pylib"))
sys.path.insert(0, _HERE)  # vendored `balrog` package

from types import SimpleNamespace

MINIHACK_TASKS = [
    "MiniHack-Boxoban-Hard-v0",
    "MiniHack-Boxoban-Medium-v0",
    "MiniHack-MazeWalk-9x9-v0",
    "MiniHack-MazeWalk-15x15-v0",
    "MiniHack-Corridor-R3-v0",
    "MiniHack-CorridorBattle-Dark-v0",
    "MiniHack-Quest-Easy-v0",
    "MiniHack-Quest-Medium-v0",
]

EPISODES_PER_TASK = 5  # eval.num_episodes.minihack

BALROG_CONFIG = SimpleNamespace(
    agent=SimpleNamespace(max_image_history=0),
    envs=SimpleNamespace(
        minihack_kwargs=dict(
            character="@",
            max_episode_steps=100,
            penalty_step=-0.01,
            penalty_time=0.0,
            penalty_mode="constant",
            savedir=None,
            save_ttyrec_every=0,
            autopickup=False,
            skip_more=True,
        ),
    ),
)


def make_env(task: str):
    from balrog.environments import make_env as balrog_make_env

    return balrog_make_env("minihack", task, BALROG_CONFIG)
