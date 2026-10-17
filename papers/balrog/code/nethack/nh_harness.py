"""BALROG-identical NetHack (full NLE) env harness.

Builds the exact env stack BALROG uses for its 'nle' environment:
  gym.make("NetHackChallenge-v0", **nle_kwargs)
  -> AutoMore -> NLELanguageWrapper -> NLETimeLimit -> GymV21CompatibilityV0 -> EnvWrapper

Config values verbatim from BALROG balrog/config/config.yaml:
  nle_kwargs: character '@', max_episode_steps 100_000, no_progress_timeout 150,
  savedir null, save_ttyrec_every 0, skip_more True; eval.num_episodes.nle = 5.

Clean-run contract: agents interact via reset(seed)/step(action_string) only and
consume only the observation dict this stack serves to every BALROG agent.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "pylib"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from types import SimpleNamespace

NLE_TASK = "NetHackChallenge-v0"   # tasks.nle_tasks
EPISODES = 5                        # eval.num_episodes.nle

BALROG_CONFIG = SimpleNamespace(
    agent=SimpleNamespace(max_image_history=0),
    envs=SimpleNamespace(
        nle_kwargs=dict(
            character="@",
            max_episode_steps=100_000,
            no_progress_timeout=150,
            savedir=None,
            save_ttyrec_every=0,
            skip_more=True,
        ),
    ),
)


def make_env():
    from balrog.environments import make_env as balrog_make_env

    return balrog_make_env("nle", NLE_TASK, BALROG_CONFIG)
