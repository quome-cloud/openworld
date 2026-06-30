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


def test_select_rejects_a_genuinely_inconsistent_lens():
    # The invariant SELECT must enforce: a lens whose (key,action) maps to >1 next-key
    # (it ALIASES the deciding cell) scores <1 and must LOSE to a Markov lens at 1.0 --
    # not be picked by tie-break. Two hand-built lenses prove this directly.
    sharp = lambda f: {"k": (int(f[0, 0]), int(f[0, 1]))}   # keeps the deciding cell -> Markov
    alias = lambda f: {"k": int(f[0, 1])}                   # DROPS f[0,0] -> collisions

    def F(a, b):
        f = np.zeros((4, 4), dtype=int); f[0, 0] = a; f[0, 1] = b; return f

    A, B, C, D, E = F(0, 0), F(1, 1), F(2, 0), F(3, 2), F(4, 9)
    trans = [(A, 1, B), (C, 1, D), (B, 1, E)]
    lenses = {"sharp": sharp, "alias": alias}

    # alias collides A & C to key 0: (0,act1) -> {1, 2} inconsistent; (1,act1)->{9} ok -> 1/2 = 0.5
    assert 0.0 < fidelity(trans, alias) < 1.0
    assert fidelity(trans, sharp) == 1.0                    # every (key,act) -> single next-key

    name, fn, score = select_lens(trans, lenses=lenses)
    assert name == "sharp" and score == 1.0                 # SELECT rejects the aliasing lens
