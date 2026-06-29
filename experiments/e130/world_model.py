"""Introspective stereotype sigma_I: a learned deterministic dynamics table over masked
object-state keys, plus the TEIE subroutine database (Thm 4.10 -- O(1) replay of banked plans).
simulate() rolls a plan forward and reports the knowledge frontier (known=False at the first
unseen transition -- exactly where tension will be measured and the model must learn by acting)."""


class WorldModel:
    def __init__(self):
        self.table = {}        # (state_key, action) -> next_state_key
        self.db = {}           # subroutine name -> plan (list of actions)
        self.conflicts = 0     # contradictions at a seen (state, action): a regime-change signal

    def predict(self, state_key, action):
        key = (state_key, action)
        if key in self.table:
            return self.table[key], True
        return None, False

    def simulate(self, state_key, plan):
        preds, s = [], state_key
        for a in plan:
            nxt, known = self.predict(s, a)
            if not known:
                return preds, False
            preds.append(nxt); s = nxt
        return preds, True

    def update(self, state_key, action, observed_next_key):
        key = (state_key, action)
        if key in self.table and self.table[key] != observed_next_key:
            self.conflicts += 1
        self.table[key] = observed_next_key

    def bank_subroutine(self, name, plan):
        self.db[name] = list(plan)

    def lookup(self, name):
        return self.db.get(name)

    # --- object-relative (lifted) dynamics: transfer across configs/levels (RL-review fix) ---
    def learn_rule(self, avatar_color, action, prev_pos, next_pos):
        self.rules = getattr(self, "rules", {})
        dy, dx = next_pos[0] - prev_pos[0], next_pos[1] - prev_pos[1]
        self.rules[(avatar_color, action)] = (dy, dx)

    def predict_rel(self, avatar_color, action):
        return getattr(self, "rules", {}).get((avatar_color, action))
