"""Phase-1 probe policies: play to learn, not to score.

ProbeActions: cycles through the named action list once each, pressing 'esc'
after any action whose message looks like it opened a prompt (heuristic:
cursor jumped to top row, or message nonempty). Pure hypothesis generation.

ProbeMove: hammers the 8 direction actions in a fixed rotation to generate
movement evidence (position deltas, blocked cases, bump-into-monster cases).
"""

import json


class ProbeActions:
    # named actions from env.language_action_space (served interface),
    # dangerous-unknowns deliberately included -- we need the evidence.
    NAMED = ['north', 'east', 'south', 'west', 'northeast', 'southeast',
             'southwest', 'northwest', 'far north', 'far east', 'far south',
             'far west', 'up', 'down', 'wait', 'more', 'adjust', 'apply',
             'attributes', 'call', 'cast', 'chat', 'close', 'dip', 'drop',
             'droptype', 'eat', 'engrave', 'enhance', 'fire', 'fight',
             'force', 'inventory', 'inventtype', 'invoke', 'jump', 'kick',
             'look', 'loot', 'monster', 'move', 'movefar', 'offer', 'open',
             'pay', 'pickup', 'pray', 'puton', 'quaff', 'quiver', 'read',
             'remove', 'ride', 'rub', 'rush', 'search', 'seetrap', 'sit',
             'swap', 'takeoff', 'throw', 'tip', 'untrap', 'wear', 'wield',
             'wipe', 'zap', 'esc', 'space']

    def __init__(self):
        self.i = 0
        self.pending_esc = 0

    def reset(self, pre):
        self.i = 0
        self.pending_esc = 0

    def act(self, pre):
        if self.pending_esc > 0:
            self.pending_esc -= 1
            return "esc", "clear-prompt"
        if self.i >= len(self.NAMED):
            return "wait", "probe-done"
        a = self.NAMED[self.i]
        self.i += 1
        return a, f"probe#{self.i-1}"

    def observe(self, pre, action, post, reward, done):
        # if the screen looks like a prompt/menu (cursor on top rows or
        # message ends with '?'), schedule escapes to clear it.
        if action != "esc":
            msg = post["msg"] or ""
            if post["cursor"][0] == 0 or msg.strip().endswith("?"):
                self.pending_esc = 2


class ProbeMove:
    ROT = ["north", "east", "south", "west", "northeast", "southeast",
           "southwest", "northwest"]

    def __init__(self, reps=40):
        self.k = 0
        self.reps = reps

    def reset(self, pre):
        self.k = 0

    def act(self, pre):
        a = self.ROT[(self.k // 5) % 8]
        self.k += 1
        return a, "probe-move"

    def observe(self, *a):
        pass
