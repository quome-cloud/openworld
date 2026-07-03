# experiments/e130/perception.py
"""Extrospective stereotype sigma_E: perceive a frame into object-state + a fixed-length vector
(for tension) + the click-target set (small/rare components -- the click modality). Reuses the
E125 object perceptor; adds the embedding and click-target extraction."""
import os, sys
from dataclasses import dataclass, field
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from experiments.e125 import objstate


@dataclass
class Stereotype:
    key: tuple
    vec: np.ndarray
    objects: list
    click_targets: list = field(default_factory=list)


def embed(objects, d=64):
    flat = []
    for o in sorted(objects, key=lambda o: (o["color"], o["y"], o["x"])):
        flat += [o["color"], o["y"], o["x"]]
    flat = (flat + [0] * d)[:d]
    return np.asarray(flat, dtype=float)


def extrospect(frame, avail=(), ignore_colors=()):
    grid = np.asarray(frame).astype(int).tolist()
    s = objstate.object_state(grid, ignore_colors=ignore_colors)
    objs = s["objects"]
    targets = [o for o in objs if o["size"] <= 16] if 6 in tuple(avail) else []
    return Stereotype(key=objstate.state_key(s), vec=embed(objs), objects=objs, click_targets=targets)
