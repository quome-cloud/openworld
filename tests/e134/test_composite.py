import numpy as np
from experiments.e134.composite import composite_key, fidelity, select_lens


def test_composite_key_distinguishes_when_any_lens_does():
    a = np.zeros((8, 8), dtype=int); b = a.copy(); b[0, 0] = 7   # differs only in one cell
    assert composite_key(a) != composite_key(b)                  # some lens catches it


def test_select_prefers_the_markov_nonaliasing_lens():
    # synthetic: 'objects' aliases (drops the deciding top-row counter) so its transitions are
    # inconsistent; 'meter' is Markov. SELECT must pick the consistent, non-degenerate lens.
    frames = []
    for t in range(4):
        f = np.zeros((8, 8), dtype=int); f[4, 4] = 5; f[0, 0] = t   # only the meter cell advances
        frames.append(f)
    trans = [(frames[i], 1, frames[i + 1]) for i in range(3)]
    name, fn, score = select_lens(trans)
    assert name == 'meter' and score > 0.0
