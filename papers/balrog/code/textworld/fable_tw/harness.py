"""BALROG-faithful TextWorld harness.

Replicates balrog/environments/textworld/base.py (TextWorldFactory + TextWorldWrapper)
without importing the balrog package (whose top-level __init__ drags in the full env zoo).
Semantics are copied verbatim from BALROG-main @ main:
  - games registered from tw_games/<task>/*.{ulx,z8}, sorted, max_episode_steps=80
  - EnvInfos(objective, description, score, max_score, won) - exactly what BALROG requests
  - observation filter: everything up to and including the objective string is stripped
  - progression = max(score/max_score, 1.0 if won else 0.0), computed when done
  - official episode selection: factory cycles count[task] += 1; index = count % n_games
"""
import glob
import os
from collections import defaultdict
from pathlib import Path

import textworld
import textworld.gym

TW_GAMES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "BALROG-main", "tw_games")
TASKS = ["treasure_hunter", "the_cooking_game", "coin_collector"]
MAX_EPISODE_STEPS = 80  # config.yaml envs.textworld_kwargs.max_episode_steps
NUM_EPISODES_OFFICIAL = 10  # config.yaml eval.num_episodes.textworld

# BALROG's textworld_kwargs (config.yaml): the exact info channels agents get.
BALROG_ENV_INFOS = dict(objective=True, description=True, score=True, max_score=True, won=True)


class TextWorldFactory:
    def __init__(self, tasks=TASKS, max_episode_steps=MAX_EPISODE_STEPS, extra_infos=None):
        self.max_steps = max_episode_steps
        self.count = defaultdict(int)
        kw = dict(BALROG_ENV_INFOS)
        if extra_infos:  # only used by dev-time validation harnesses, never at test time
            kw.update(extra_infos)
        self.request_infos = textworld.EnvInfos(**kw)
        self.env_ids = defaultdict(list)
        self.gamefiles = defaultdict(list)
        for pattern in ["*.ulx", "*.z8"]:
            for entry in sorted(glob.glob(os.path.join(TW_GAMES_PATH, f"**/{pattern}"), recursive=True)):
                task = Path(entry).parent.name
                if task in tasks:
                    env_id = textworld.gym.register_game(entry, self.request_infos, max_episode_steps=max_episode_steps)
                    self.env_ids[task].append(env_id)
                    self.gamefiles[task].append(entry)

    def get_env(self, task, seed=None):
        """seed=None cycles (official evaluator path); seed=i selects game i % n."""
        if seed is not None:
            idx = seed % len(self.env_ids[task])
        else:
            self.count[task] += 1
            idx = self.count[task] % len(self.env_ids[task])
        env = textworld.gym.make(self.env_ids[task][idx])
        return TextWorldWrapper(env, max_steps=self.max_steps), self.gamefiles[task][idx]


class TextWorldWrapper:
    """Verbatim port of BALROG's TextWorldWrapper (gym.Wrapper shell dropped; same behavior)."""

    def __init__(self, env, max_steps=MAX_EPISODE_STEPS):
        self.env = env
        self.progression = 0.0
        self.max_steps = max_steps

    def filter_objective(self, obs, info):
        objective = info["objective"]
        parts = obs.split(objective)
        if len(parts) == 1:
            return parts[0].strip()
        else:
            return parts[-1].strip()

    def _process(self, obs):
        return {"text": {"long_term_context": obs, "short_term_context": ""}, "image": None}

    def reset(self):
        obs, info = self.env.reset()
        obs = self.filter_objective(obs, info)
        self.progression = 0.0
        self._last_info = info
        return self._process(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.filter_objective(obs, info)
        if done:
            self.progression = max(info["score"] / info["max_score"], 1.0 if info["won"] else 0.0)
        self._last_info = info
        return self._process(obs), reward, done, info

    def get_stats(self):
        return {"progression": self.progression}

    def close(self):
        self.env.close()
