"""Exact replication of BALROG's BabyAI env construction + episode protocol.

Source: balrog/environments/babyai_text/babyai_env.py (make_babyai_env) and
balrog/evaluator.py (run_episode): per episode a fresh env is constructed by
looping gym.make('BabyAI-MixedTrainLocal-v0', num_dists=0) until
env.unwrapped.action_kinds[0] matches the task's goal, then env.reset(seed=s).
Success metric: progression = 1.0 iff any step returns reward > 0
(BabyAITextCleanLangWrapper.step). max steps = env.max_steps.

BALROG draws s from a hash of pid/time (get_unique_seed) - i.e. random,
unrecorded-in-advance seeds, 10 episodes per task. We use a fixed, recorded
seed list (base + episode index) for reproducibility; this matches the
official protocol in distribution (seeds are arbitrary ints either way).
"""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401  (registers envs on import in fork? be explicit:)

minigrid.register_minigrid_envs()

TASKS = ["goto", "pickup", "open", "putnext", "pick_up_seq_go_to"]
NUM_EPISODES = 10          # config.eval.num_episodes.babyai
BABYAI_KWARGS = {"num_dists": 0}   # config.envs.babyai_kwargs


def make_task_env(goal: str, max_tries=1000):
    """Mirror make_babyai_env's construction loop (task type is drawn at
    construction time; reset(seed) then determines the layout)."""
    for _ in range(max_tries):
        env = gym.make("BabyAI-MixedTrainLocal-v0", **BABYAI_KWARGS)
        if env.unwrapped.action_kinds[0].replace(" ", "_") == goal:
            return env
        env.close()
    raise RuntimeError(f"could not draw task {goal} in {max_tries} tries")
