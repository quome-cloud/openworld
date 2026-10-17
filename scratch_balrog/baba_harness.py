"""Baba Is AI harness for BALROG - Game interface mirroring arc3_harness.py.

Usage:
    from baba_harness import Game
    g = Game("env/goto_win")
    g.reset()                # returns obs (8x8x3 numpy array)
    g.frame                  # numpy 8x8x3 grid obs
    g.avail                  # ['up', 'right', 'down', 'left']
    g.levels                 # episodes solved this session
    g.done                   # True when episode ended
    g.step("up")             # directional action -> next frame
    g.clone()                # deep copy for tree search
    g.get_ruleset_text()     # "key is win\\nbaba is you"
    g.get_objects()          # {type: [(x,y), ...]}
    g.get_win_positions()    # [(x,y)] of WIN objects
"""
import copy
import numpy as np
import baba as _baba
from baba.world_object import name_mapping

ACTIONS = ['up', 'right', 'down', 'left']
_ACTION_INT = {'idle': 0, 'up': 1, 'right': 2, 'down': 3, 'left': 4}

# Baba Is AI property string -> human name (extend baba's own mapping)
_PROP_NAME = dict(name_mapping)


def _get_rule_property(rule, key="property"):
    val = rule.get(key, "")
    return _PROP_NAME.get(val, val)


class Game:
    def __init__(self, task_id, seed=None):
        self.task_id = task_id
        self.seed = seed
        kwargs = {} if seed is None else {"seed": seed}
        self._env = _baba.make(task_id, **kwargs)
        self.levels = 0
        self.done = False
        self.avail = ACTIONS[:]
        self.frame = None

    def reset(self):
        obs = self._env.reset()
        self.done = False
        self.frame = obs
        return obs

    @property
    def agent_pos(self):
        return tuple(self._env.agent_pos)

    @property
    def width(self):
        return self._env.width

    @property
    def height(self):
        return self._env.height

    def get_ruleset_text(self):
        rules = []
        for rule in self._env.grid._ruleset["_rule_"]:
            if "object" not in rule or "property" not in rule:
                continue
            name = rule["object"].removeprefix("f")
            prop = _get_rule_property(rule)
            rules.append(f"{name} is {prop}")
        return "\n".join(rules)

    def get_objects(self):
        """Returns {cell_type: [(x,y), ...]} for all non-None cells."""
        objects = {}
        for j in range(self._env.height):
            for i in range(self._env.width):
                cell = self._env.grid.get(i, j)
                if cell is not None:
                    objects.setdefault(cell.type, []).append((i, j))
        return objects

    def _win_object_types(self):
        types = set()
        for rule in self._env.grid._ruleset["_rule_"]:
            if rule.get("property") == "is_goal":
                types.add(rule["object"])
        return types

    def _stop_object_types(self):
        types = set()
        for rule in self._env.grid._ruleset["_rule_"]:
            if rule.get("property") == "is_stop":
                types.add(rule["object"])
        return types

    def get_win_positions(self):
        win_types = self._win_object_types()
        return [
            (i, j)
            for j in range(self._env.height)
            for i in range(self._env.width)
            if self._env.grid.get(i, j) is not None
            and self._env.grid.get(i, j).type in win_types
        ]

    def get_stop_positions(self):
        stop_types = self._stop_object_types()
        return [
            (i, j)
            for j in range(self._env.height)
            for i in range(self._env.width)
            if self._env.grid.get(i, j) is not None
            and self._env.grid.get(i, j).type in stop_types
        ]

    def get_wall_positions(self):
        return [
            (i, j)
            for j in range(self._env.height)
            for i in range(self._env.width)
            if self._env.grid.get(i, j) is not None
            and self._env.grid.get(i, j).type == "wall"
        ]

    def state_key(self):
        """Bytes hash of full game state for BFS visited-set."""
        return self._env.gen_obs().tobytes()

    def step(self, action):
        """action: str 'up'/'right'/'down'/'left' or int 0-4."""
        action_int = _ACTION_INT[action] if isinstance(action, str) else action
        obs, reward, done, _ = self._env.step(action_int)
        self.done = done
        self.frame = obs
        if done and reward > 0:
            self.levels += 1
        return obs

    def clone(self):
        """Deep copy for tree search without re-creating the env."""
        g = object.__new__(Game)
        g.task_id = self.task_id
        g.seed = self.seed
        g._env = copy.deepcopy(self._env)
        g.levels = self.levels
        g.done = self.done
        g.avail = self.avail[:]
        g.frame = self.frame.copy() if self.frame is not None else None
        return g
