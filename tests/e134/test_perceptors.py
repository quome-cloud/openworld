import numpy as np
from experiments.e134.perceptors import LENSES, key_of


def _frame():
    f = np.zeros((16, 16), dtype=int)
    f[1, 1] = 3                      # rare singleton
    f[0, :] = (np.arange(16) % 9)    # a changing top "status bar"
    f[8:12, 4:8] = 5                 # a block (for symmetry/region)
    return f


def test_all_lenses_produce_keys():
    f = _frame()
    for name, fn in LENSES.items():
        s = fn(f)
        assert isinstance(s, dict)
        k = key_of(s)
        assert isinstance(k, tuple) and k == key_of(fn(f))   # deterministic


def test_meter_lens_captures_the_status_row_values():
    # the 'meter' lens must expose the top-row values (a timer/counter) -- the feature object-state drops
    f = _frame()
    m = LENSES['meter'](f)
    assert 'meter' in m and len(m['meter']) > 0


def test_salience_ranks_small_rare_first():
    f = _frame()
    s = LENSES['salience'](f)
    assert s['targets'] and s['targets'][0][:2] == [1, 1]   # the rare singleton at (y,x)=(1,1)
