"""Gymnasium adapter: expose any OpenWorld ``World`` as a standard ``gym.Env`` so a verified
code world model drops into existing planners / RL libraries (Stable-Baselines3, CleanRL, ...).

This is an OPTIONAL integration layer: it imports ``gymnasium`` (a third-party package) and is
NOT imported by ``openworld/__init__.py``, so the core stays zero-dependency. Install with
``pip install openworld[gym]`` (or just ``pip install gymnasium``) and:

    from openworld.gym_env import OpenWorldEnv
    env = OpenWorldEnv(world, objectives=suite)      # suite optional -> 0 reward
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

The observation is a flat ``Box`` over the world's scalar numeric state fields (the full symbolic
state is always in ``info["state"]``); the action space is ``Discrete`` over the world's declared
actions. Rollouts run at native code speed (the world's verified transition), not LLM inference.
"""

from typing import Any, Callable, Dict, List, Optional

try:
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover - optional dependency
    raise ImportError(
        "OpenWorldEnv needs gymnasium + numpy: pip install 'openworld[gym]' (or pip install "
        "gymnasium numpy). The OpenWorld core itself remains zero-dependency."
    ) from e

from .state import Action, WorldState


def _numeric_keys(state: Dict[str, Any]) -> List[str]:
    """Sorted scalar numeric fields of a state (bools excluded), the Box observation vector."""
    return sorted(k for k, v in state.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool))


class OpenWorldEnv(gym.Env):
    """A Gymnasium environment backed by an OpenWorld verified ``World``.

    world:       an OpenWorld World with a runnable transition.
    objectives:  optional ObjectiveSuite/Objective (or callable(prev, action, next)->float)
                 providing the reward; absent -> reward 0.0.
    is_terminal: optional callable(state)->bool for episode termination; absent -> never
                 terminates (episodes end by ``max_steps`` truncation).
    max_steps:   truncation horizon.
    """

    metadata = {"render_modes": []}

    def __init__(self, world, objectives=None, is_terminal: Optional[Callable] = None,
                 max_steps: int = 100):
        super().__init__()
        if getattr(world, "transition", None) is None:
            raise ValueError(f"World {getattr(world, 'name', '?')!r} has no dynamics to step.")
        self.world = world
        self._reward = self._make_reward(objectives)
        self._is_terminal = is_terminal
        self.max_steps = int(max_steps)
        self._actions = list(world.actions)
        self.action_space = spaces.Discrete(len(self._actions))
        self._keys = _numeric_keys(world.initial_state)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(self._keys),), dtype=np.float64)
        self._steps = 0

    def _make_reward(self, objectives) -> Callable:
        if objectives is None:
            return lambda prev, a, nxt: 0.0
        # A plain reward callable(prev, action, next) -> float.
        if callable(objectives) and not hasattr(objectives, "score"):
            return lambda prev, a, nxt: float(objectives(prev, a, nxt))

        # Objective.score -> float; ObjectiveSuite.score -> {name: float, "aggregate": float}.
        def reward(prev, a, nxt):
            out = objectives.score(prev, a, nxt)
            return float(out["aggregate"] if isinstance(out, dict) else out)
        return reward

    def _obs(self, state: Dict[str, Any]):
        return np.array([float(state.get(k, 0.0)) for k in self._keys], dtype=np.float64)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        state = self.world.reset()
        self._steps = 0
        return self._obs(state), {"state": dict(state)}

    def step(self, action: int):
        a = Action(name=self._actions[int(action)])
        prev = WorldState(dict(self.world.state))
        nxt = self.world.step(a)
        self._steps += 1
        reward = float(self._reward(prev, a, nxt))
        terminated = bool(self._is_terminal(nxt)) if self._is_terminal else False
        truncated = self._steps >= self.max_steps
        return self._obs(nxt), reward, terminated, truncated, {"state": dict(nxt), "action": a.name}

    def action_name(self, index: int) -> str:
        return self._actions[int(index)]
