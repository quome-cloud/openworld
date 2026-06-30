"""WorldSim: a pure, side-effect-free learned simulator. predict(state, action) is a table lookup over
OBSERVED transitions -- no real env, so the lookahead planner gets perfect free backtracking and an
arbitrary horizon (the fix for E131's reset()+replay fragility and short horizon). Unknown transitions
return None (the knowledge frontier); solve_hybrid fills them by bounded real-env exploration + the
E130 object-relative rules, then re-plans."""


class WorldSim:
    def __init__(self):
        self.trans = {}          # (state_key, action_key) -> (next_key, levels)
        self.seen = set()

    def learn(self, state_key, action, next_key, next_levels):
        self.trans[(state_key, tuple(action))] = (next_key, int(next_levels))
        self.seen.add(state_key); self.seen.add(next_key)

    def predict(self, state_key, action):
        return self.trans.get((state_key, tuple(action)))

    def known(self, state_key, action):
        return (state_key, tuple(action)) in self.trans
