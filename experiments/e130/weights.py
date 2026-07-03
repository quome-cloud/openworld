"""Multiplicative-weights bandit over expert proposers (the RL review's adaptive EFEI weighting):
upweight whichever expert actually produced progress on THIS game. This is the real weighted average
the formalism allows (non-uniform w_j), and it counters expert bias by down-weighting wrong experts."""
import math


class ExpertWeights:
    def __init__(self, names, eta=0.5):
        self.eta = eta
        self._w = {n: 1.0 for n in names}

    def weight(self, name):
        tot = sum(self._w.values()) or 1.0
        return self._w.get(name, 0.0) / tot

    def reward(self, name, r):
        if name in self._w:
            self._w[name] *= math.exp(self.eta * float(r))

    def as_dict(self):
        tot = sum(self._w.values()) or 1.0
        return {n: v / tot for n, v in self._w.items()}
