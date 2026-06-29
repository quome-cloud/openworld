"""Moral filter (Def 2.8): expert proposers each suggest a candidate waypoint; the world model
scores each by simulated progress (valence V); pooled agreement gives phi; the selected behavior
maximizes S = phi*V. With amateur=True the rule degenerates to a uniform draw (the book's amateur
filter, Thm 4.4's Theta(M) regime) -- kept as the ablation baseline."""
from collections import Counter
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Waypoint:
    kind: str        # 'click' | 'reach'
    y: int
    x: int
    source: str


def reach_rare_color(stereotype, history, world_model):
    if not stereotype.objects:
        return []
    colors = Counter(o["color"] for o in stereotype.objects)
    rare = min(colors, key=colors.get)
    return [Waypoint("reach", o["y"], o["x"], "reach_rare_color")
            for o in stereotype.objects if o["color"] == rare]


def click_smallest(stereotype, history, world_model):
    if not stereotype.click_targets:
        return []
    o = min(stereotype.click_targets, key=lambda o: o["size"])
    return [Waypoint("click", o["y"], o["x"], "click_smallest")]


def reach_unseen_state(stereotype, history, world_model):
    # coverage proposer: target objects whose (key, candidate-action) is unseen in the model
    out = []
    for o in stereotype.objects:
        if (stereotype.key, ("click", o["y"], o["x"])) not in world_model.table:
            out.append(Waypoint("click", o["y"], o["x"], "reach_unseen_state"))
    return out


DEFAULT_EXPERTS = [reach_rare_color, click_smallest]


def valence(wp, stereotype, world_model, potential=None):
    # simulated progress: novelty of the waypoint's induced transition under sigma_I.
    seen = (stereotype.key, (wp.kind, wp.y, wp.x)) in world_model.table
    base = 0.5 if seen else 1.0       # unseen transitions carry more expected progress
    if potential is None:
        return base
    # potential-based shaping (Ng-Harada-Russell): gamma*Phi(s') - Phi(s)
    # proxy: unseen next-state has potential 1.0 (novel), seen has 0.0.
    phi_cur = potential(stereotype.key)
    phi_next = 0.0 if seen else 1.0
    return base + phi_next - phi_cur


def realize(wp, stereotype, dir_map=None, avatar=None):
    if wp.kind == "reach" and dir_map and avatar is not None:
        # use navigation to produce a directional action plan toward the target
        from experiments.e130 import navigation  # lazy import to avoid circular dependency
        return navigation.plan_reach(stereotype, avatar, dir_map, (wp.y, wp.x))
    if wp.kind == "click":
        return [(6, wp.x, wp.y)]
    return [(6, wp.x, wp.y)]          # reach degenerates to click without dir_map+avatar


def select(stereotype, history, world_model, experts, rng, dir_map=None, amateur=False, weights=None):
    pool = []
    for e in experts:
        pool += e(stereotype, history, world_model)
    if not pool:
        return Waypoint("noop", 0, 0, "none"), [], 0.0
    if amateur:
        wp = pool[int(rng.integers(0, len(pool)))]
        return wp, realize(wp, stereotype, dir_map), 0.0
    # phi = pooled agreement (how many experts proposed this (kind,y,x)); S = phi * V
    agree = Counter((w.kind, w.y, w.x) for w in pool)
    best, best_s = None, float("-inf")   # -inf (not -1.0): a pool member always wins, so best is never None
    for w in pool:
        V = valence(w, stereotype, world_model)
        if weights is not None:
            s = weights.weight(w.source) * agree[(w.kind, w.y, w.x)] * V
        else:
            s = agree[(w.kind, w.y, w.x)] * V
        if s > best_s:
            best, best_s = w, s
    return best, realize(best, stereotype, dir_map), float(best_s)
