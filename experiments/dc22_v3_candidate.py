import numpy as np

# dc22 world model v3 — 5 bugs fixed vs Prism's v2 (see M96889 for full list)
#
# Action semantics (match training data labels):
#   action=6  → button y=19: RECT_A+RECT_B toggle 4↔8 (+ BOX fill + counter)
#   action=7  → button y=36: TOGGLE_B toggle 4↔9   (+ BOX fill + counter)
#   action=1-4 → directional movement (2-step, 2×2 entity)
#
# BFS usage note: real clicks [6,48,19] → model action=6, [6,48,36] → model action=7

_BOX_POS = [
    (16,44),(16,45),(16,46),(16,47),(16,48),(16,49),(16,50),(16,51),(16,52),
    (17,44),(17,52),(18,44),(18,52),
    (19,41),(19,42),(19,43),(19,44),(19,52),(19,53),(19,54),(19,55),
    (20,41),(20,55),(21,41),(21,55),
    (22,41),(22,42),(22,43),(22,44),(22,45),(22,46),(22,47),(22,48),
    (22,49),(22,50),(22,51),(22,52),(22,53),(22,54),(22,55),
    (33,44),(33,45),(33,46),(33,47),(33,48),(33,49),(33,50),(33,51),(33,52),
    (34,44),(34,52),(35,44),(35,52),
    (36,41),(36,42),(36,43),(36,44),(36,52),(36,53),(36,54),(36,55),
    (37,41),(37,55),(38,41),(38,55),
    (39,41),(39,42),(39,43),(39,44),(39,45),(39,46),(39,47),(39,48),
    (39,49),(39,50),(39,51),(39,52),(39,53),(39,54),(39,55),
]

# button y=36 toggles these (4↔9)
_TOGGLE_B = [
    (20,18),(20,20),(21,19),(21,21),(22,18),(22,20),(23,19),(23,21),
    (34,8),(34,10),(35,9),(35,11),(36,8),(36,10),(37,9),(37,11),
]

# button y=19 toggles these (4↔8)
_RECT_A = [(r,c) for r in range(24,30) for c in range(18,22)]
_RECT_B = [(r,c) for r in range(30,34) for c in range(12,18)]

# Fix 2: 8 (open RECT) and 13 (sprite zone) are also traversable; 4 is always a barrier
_TRAVERSABLE = {2, 8, 9, 13}

# Fix 3: parity=1 so first move of each episode increments the row-63 counter (matches real game)
# Fix 4: bg is 2×2 array to correctly restore mixed-value cells after entity leaves
_state = {'parity': 1, 'prev_n3': -1, 'bg': np.full((2, 2), 2, dtype=np.uint8)}


def predict(frame: np.ndarray, action: int) -> np.ndarray:
    result = frame.copy()

    n3 = int(np.sum(frame[63, :] == 3))
    if _state['prev_n3'] >= 0 and n3 < _state['prev_n3']:
        # Episode reset: counter decreased — new game started
        _state['parity'] = 1  # Fix 3: reset to 1, not 0
        _state['bg'] = np.full((2, 2), 2, dtype=np.uint8)

    if action == 6:  # Fix 1: action=6 = button y=19 = RECT toggle
        for r, c in _RECT_A:
            v = result[r, c]
            if v == 4: result[r, c] = 8
            elif v == 8: result[r, c] = 4
        for r, c in _RECT_B:
            v = result[r, c]
            if v == 4: result[r, c] = 8
            elif v == 8: result[r, c] = 4
        for r, c in _BOX_POS:
            if result[r, c] == 0:
                result[r, c] = 5
        filled = int(np.sum(frame[63, :] == 3))
        if filled < 64:
            result[63, filled] = 3
        _state['prev_n3'] = filled + 1

    elif action == 7:  # Fix 1: action=7 = button y=36 = TOGGLE_B
        for r, c in _TOGGLE_B:
            v = result[r, c]
            if v == 9: result[r, c] = 4
            elif v == 4: result[r, c] = 9
        for r, c in _BOX_POS:
            if result[r, c] == 0:
                result[r, c] = 5
        filled = int(np.sum(frame[63, :] == 3))
        if filled < 64:
            result[63, filled] = 3
        _state['prev_n3'] = filled + 1

    else:  # directional movement (1-4)
        positions = np.argwhere(frame == 14)
        if len(positions) > 0:
            r0 = int(positions[:, 0].min())
            c0 = int(positions[:, 1].min())
            dr = {1: -2, 2: 2, 3: 0, 4: 0}.get(action, 0)
            dc = {1: 0, 2: 0, 3: -2, 4: 2}.get(action, 0)
            nr, nc = r0 + dr, c0 + dc
            if 0 <= nr <= 62 and 0 <= nc <= 62:
                dest = frame[nr:nr+2, nc:nc+2]
                if dest.shape == (2, 2) and np.all(np.isin(dest, list(_TRAVERSABLE))):
                    result[r0:r0+2, c0:c0+2] = _state['bg']  # Fix 4: restore full 2×2 bg
                    result[nr:nr+2, nc:nc+2] = 14
                    _state['bg'] = dest.copy()                 # Fix 4: save new 2×2 bg

        if _state['parity'] == 1:  # Fix 3: parity=1 means increment this step
            filled = int(np.sum(frame[63, :] == 3))
            if filled < 64:
                result[63, filled] = 3
            _state['prev_n3'] = filled + 1
        else:
            _state['prev_n3'] = n3
        _state['parity'] = 1 - _state['parity']

    return result
