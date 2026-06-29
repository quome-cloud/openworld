import numpy as np
from experiments.e127.safe_exec import compile_engine

GOOD = '''
class Engine:
    def __init__(self):
        self.state = {"levels": 0, "t": 0}
    def reset(self):
        self.state = {"levels": 0, "t": 0}
        return np.zeros((4, 4), dtype=int)
    def step(self, action):
        self.state["t"] += 1
        f = np.zeros((4, 4), dtype=int); f[0, 0] = self.state["t"] % 16
        return f
    def is_win(self, prev_frame):
        return self.state["levels"] >= 1
'''

def test_compiles_and_runs():
    factory = compile_engine(GOOD)
    assert factory is not None
    e = factory()
    f0 = e.reset()
    assert f0.shape == (4, 4) and f0.sum() == 0
    f1 = e.step((7, None, None))
    assert f1[0, 0] == 1 and e.state["t"] == 1

def test_fresh_instances_are_independent():
    factory = compile_engine(GOOD)
    a, b = factory(), factory()
    a.reset(); a.step((7, None, None))
    b.reset()
    assert a.state["t"] == 1 and b.state["t"] == 0

def test_syntax_error_returns_none():
    assert compile_engine("class Engine(:\n  pass") is None

def test_missing_engine_class_returns_none():
    assert compile_engine("x = 1") is None

def test_numpy_available_but_imports_blocked():
    # numpy usable via the injected `np`; but `import os` must fail at runtime
    bad = "class Engine:\n    def reset(self):\n        import os\n        return np.zeros((2,2),dtype=int)\n"
    factory = compile_engine(bad)
    # compiles (def body not executed yet) but reset() raises -> factory ok, reset raises
    e = factory()
    raised = False
    try:
        e.reset()
    except Exception:
        raised = True
    assert raised
