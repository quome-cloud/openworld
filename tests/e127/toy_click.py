# tests/e127/toy_click.py
"""Click-only (mouse) ordered-protocol ground-truth game for E127 (GameLike). Win = press buttons
A->B->C in order. Hidden state = phase. Invalid/out-of-order clicks are no-ops."""
import numpy as np

A, B, C = (2, 2), (4, 5), (6, 1)            # (row, col)
_BTN = [(A, 5), (B, 6), (C, 7)]             # in required press order
CLICK_ACTION_API = ("Click-only game (avail=[6]). A click is step(6, x, y) with x=col, y=row, 0-63. "
                    "Small colored buttons must be pressed in a fixed order; pressing the correct next "
                    "button recolors it; wrong/empty clicks are no-ops. Row 0 is a status bar. 8x8, colors 0-15.")


def _draw(pressed, t):
    f = np.zeros((8, 8), dtype=int)
    for i, ((ry, cx), col) in enumerate(_BTN):
        f[ry, cx] = 3 if i < pressed else col
    f[0, 0] = (t % 15) + 1
    return f


class ToyClickGame:
    def __init__(self):
        self.win = 1
        self._reset_fields()

    def _reset_fields(self):
        self.phase = 0; self.t = 0; self.levels = 0; self.done = False; self.avail = [6]

    def reset(self):
        self._reset_fields()
        self.frame = _draw(self.phase, self.t)
        return self.frame

    def step(self, a, x=None, y=None):
        if not self.done and a == 6 and self.phase < 3:
            (ry, cx), _col = _BTN[self.phase]
            if x == cx and y == ry:                 # correct next button
                self.phase += 1; self.t += 1
                if self.phase == 3:
                    self.levels += 1; self.done = self.levels >= self.win
        self.frame = _draw(self.phase, self.t)
        return self.frame


def toy_click_factory():
    return ToyClickGame()


TOY_CLICK_ENGINE_SRC = '''
_BTN = [((2, 2), 5), ((4, 5), 6), ((6, 1), 7)]
class Engine:
    def __init__(self):
        self.state = {"levels": 0, "phase": 0, "t": 0, "done": False}
    def _draw(self):
        f = np.zeros((8, 8), dtype=int)
        for i, ((ry, cx), col) in enumerate(_BTN):
            f[ry, cx] = 3 if i < self.state["phase"] else col
        f[0, 0] = (self.state["t"] % 15) + 1
        return f
    def reset(self):
        self.state = {"levels": 0, "phase": 0, "t": 0, "done": False}
        return self._draw()
    def step(self, action):
        k, x, y = action; s = self.state
        if not s["done"] and k == 6 and s["phase"] < 3:
            (ry, cx), _col = _BTN[s["phase"]]
            if x == cx and y == ry:
                s["phase"] += 1; s["t"] += 1
                if s["phase"] == 3:
                    s["levels"] += 1; s["done"] = s["levels"] >= 1
        return self._draw()
    def is_win(self, prev_frame):
        return self.state["levels"] >= 1
'''
