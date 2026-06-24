"""Gymnasium wrapper for ARC-AGI-3 games, so DreamerV3 (and any gym RL agent) can train on them --
the learned-world-model + model-based-RL baseline for the verified-code approach (E86/E88).

ARC-AGI-3's `arc-agi` toolkit is gym-LIKE but not a registered gymnasium env. This wraps one game:
  obs    : 64x64x3 uint8 image (the 16-color grid mapped to grayscale-ish RGB for the CNN encoder)
  action : Discrete(7) (ACTION1..7; unavailable actions are no-ops with a tiny penalty)
  reward : delta in levels_completed (sparse) -- the official progress signal
  done   : env reaches a terminal (win/over) state

Registers ids like "Arc3-ls20-v0". Used by bench/competitors/run_dreamerv3_arc.sh.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import arc_agi
from arcengine import GameAction

ACTS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4,
        GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7]


def _grid(obs):
    a = np.asarray(obs.frame)
    return a[-1].reshape(64, 64) if a.ndim == 3 else a.reshape(64, 64)


class Arc3Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, game="ls20", max_steps=1000):
        super().__init__()
        self.game = game
        self.max_steps = max_steps
        self._arc = arc_agi.Arcade()
        self.observation_space = spaces.Box(0, 255, (64, 64, 3), np.uint8)
        self.action_space = spaces.Discrete(7)
        self._env = None
        self._best = 0
        self._avail = list(range(1, 8))
        self._t = 0
        self._last = np.zeros((64, 64, 3), np.uint8)

    def _obs(self, o):
        g = _grid(o).astype(np.uint8) * 16  # 0..15 -> 0..240, spread for the CNN
        self._last = np.repeat(g[:, :, None], 3, axis=2)
        return self._last

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env = self._arc.make(self.game)
        o = self._env.reset()
        self._best = o.levels_completed
        self._avail = list(o.available_actions)
        self._t = 0
        return self._obs(o), {"levels": o.levels_completed}

    def step(self, action):
        self._t += 1
        a1 = int(action) + 1  # Discrete(0..6) -> ACTION1..7
        if a1 not in self._avail:                       # unavailable action -> no-op + tiny penalty
            return self._last, -0.001, False, self._t >= self.max_steps, {"noop": True}
        try:
            o = self._env.step(ACTS[a1 - 1])
        except Exception:  # noqa: BLE001
            o = None
        if o is None or getattr(o, "frame", None) is None:
            return self._last, -0.01, False, True, {"bad_step": True}
        lvl = o.levels_completed
        reward = float(lvl - self._best)
        self._best = max(self._best, lvl)
        self._avail = list(o.available_actions)
        terminated = str(o.state) != "GameState.NOT_FINISHED"
        truncated = self._t >= self.max_steps
        return self._obs(o), reward, terminated, truncated, {"levels": lvl, "win_levels": o.win_levels}


def register_all(games=None):
    """Register Arc3-<game>-v0 ids. Default: all public games (suppress arc_agi import logs)."""
    import contextlib
    import io
    import logging
    if games is None:
        logging.disable(logging.CRITICAL)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            envs = arc_agi.Arcade().available_environments
        games = sorted({(e if isinstance(e, str) else getattr(e, "game_id", str(e))).split("-")[0]
                        for e in envs})
    for g in games:
        gym.register(id=f"Arc3-{g}-v0", entry_point="arc3_gym:Arc3Env", kwargs={"game": g})
    return games


# register on import so `gymnasium.make("Arc3-ls20-v0")` works in the dreamerv3 subprocess
try:
    register_all()
except Exception:  # noqa: BLE001
    pass
