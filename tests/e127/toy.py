"""Deterministic hidden-state ground-truth game for E127 offline tests (GameLike)."""
import numpy as np

_GEMS = [(1, 1), (1, 6), (6, 3)]
TOY_ACTIONS = [1, 2, 3, 4, 5, 7]
ACTION_API = ("Actions: 1=up,2=down,3=left,4=right move a cursor (color 8) one cell, clamped to "
              "rows 1..7 / cols 0..7. 5=interact and 7=noop do not move. Row 0 is a status bar. "
              "Grid 8x8, colors 0-15. No clicks (action 6 unavailable).")


def _draw(cursor, gems, t):
    f = np.zeros((8, 8), dtype=int)
    for (gy, gx) in gems:
        f[gy, gx] = 4
    f[cursor[0], cursor[1]] = 8
    f[0, 0] = (t % 15) + 1
    return f


class ToyGame:
    def __init__(self):
        self.win = 1
        self._reset_fields()

    def _reset_fields(self):
        self.cursor = [4, 4]
        self.gems = list(_GEMS)
        self.collected = 0          # HIDDEN
        self.t = 0
        self.levels = 0
        self.done = False
        self.avail = [1, 2, 3, 4, 5, 7]

    def reset(self):
        self._reset_fields()
        self.frame = _draw(self.cursor, self.gems, self.t)
        return self.frame

    def step(self, a, x=None, y=None):
        if not self.done:
            self.t += 1
            ny, nx = self.cursor
            if a == 1:
                ny = max(1, ny - 1)
            elif a == 2:
                ny = min(7, ny + 1)
            elif a == 3:
                nx = max(0, nx - 1)
            elif a == 4:
                nx = min(7, nx + 1)
            self.cursor = [ny, nx]
            if (ny, nx) in self.gems:
                self.gems.remove((ny, nx))
                self.collected += 1
                if self.collected == 3:
                    self.levels += 1
                    self.collected = 0
                    self.gems = list(_GEMS)
                    self.done = self.levels >= self.win
        self.frame = _draw(self.cursor, self.gems, self.t)
        return self.frame


def toy_factory():
    return ToyGame()


# A FAITHFUL reconstruction (what a perfect model would author). Mirrors ToyGame exactly.
TOY_ENGINE_SRC = '''
_GEMS = [(1, 1), (1, 6), (6, 3)]
class Engine:
    def __init__(self):
        self.state = {"levels": 0, "collected": 0, "t": 0, "cursor": [4, 4], "gems": list(_GEMS), "done": False}
    def _draw(self):
        f = np.zeros((8, 8), dtype=int)
        for (gy, gx) in self.state["gems"]:
            f[gy, gx] = 4
        c = self.state["cursor"]; f[c[0], c[1]] = 8
        f[0, 0] = (self.state["t"] % 15) + 1
        return f
    def reset(self):
        self.state = {"levels": 0, "collected": 0, "t": 0, "cursor": [4, 4], "gems": list(_GEMS), "done": False}
        return self._draw()
    def step(self, action):
        a = action[0]; s = self.state
        if not s["done"]:
            s["t"] += 1
            ny, nx = s["cursor"]
            if a == 1: ny = max(1, ny - 1)
            elif a == 2: ny = min(7, ny + 1)
            elif a == 3: nx = max(0, nx - 1)
            elif a == 4: nx = min(7, nx + 1)
            s["cursor"] = [ny, nx]
            if (ny, nx) in s["gems"]:
                s["gems"].remove((ny, nx)); s["collected"] += 1
                if s["collected"] == 3:
                    s["levels"] += 1; s["collected"] = 0; s["gems"] = list(_GEMS)
                    s["done"] = s["levels"] >= 1
        return self._draw()
    def is_win(self, prev_frame):
        return self.state["levels"] >= 1
'''

# A PLAUSIBLE-BUT-WRONG reconstruction: moves the cursor correctly but never collects gems / levels up.
# Bug: draws gems AFTER cursor (wrong z-order), so when cursor sits on a gem cell the gem (4)
# overwrites the cursor (8) — diverges from ToyGame the moment cursor reaches a gem.
TOY_WRONG_SRC = '''
_GEMS = [(1, 1), (1, 6), (6, 3)]
class Engine:
    def __init__(self):
        self.state = {"levels": 0, "t": 0, "cursor": [4, 4]}
    def _draw(self):
        f = np.zeros((8, 8), dtype=int)
        c = self.state["cursor"]; f[c[0], c[1]] = 8
        for (gy, gx) in _GEMS:
            f[gy, gx] = 4
        f[0, 0] = (self.state["t"] % 15) + 1
        return f
    def reset(self):
        self.state = {"levels": 0, "t": 0, "cursor": [4, 4]}
        return self._draw()
    def step(self, action):
        a = action[0]; s = self.state; s["t"] += 1
        ny, nx = s["cursor"]
        if a == 1: ny = max(1, ny - 1)
        elif a == 2: ny = min(7, ny + 1)
        elif a == 3: nx = max(0, nx - 1)
        elif a == 4: nx = min(7, nx + 1)
        s["cursor"] = [ny, nx]
        return self._draw()
    def is_win(self, prev_frame):
        return False
'''
