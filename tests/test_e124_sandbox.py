import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import numpy as np
from e124 import sandbox_exec

FR = np.zeros((64, 64), dtype=int)

def test_valid_predicate_returns_value():
    src = "def f(frame):\n    return float((frame==0).sum())"
    assert sandbox_exec.eval_fn(src, "f", FR) == 4096.0

def test_broken_code_returns_none():
    assert sandbox_exec.eval_fn("def f(frame):\n    return undefined_name", "f", FR) is None

def test_timeout_returns_none():
    assert sandbox_exec.eval_fn("def f(frame):\n    while True: pass", "f", FR, timeout=1.0) is None

def test_bool_coerces_to_float():
    assert sandbox_exec.eval_fn("def f(frame):\n    return (frame.sum()==0)", "f", FR) == 1.0
