"""Greedy action policy shared by all coordination conditions.

Action selection heuristic: enumerate available build actions, score each by
expected-VP-per-resource-cost, pick the highest-scoring sequence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .board import BOARD, HEX_TOKEN, RESOURCES, STONE, WOOD, GRAIN
from .personas import Persona
from .state import (
    CITY_COST, ROAD_COST, SETTLEMENT_COST, BANK_TRADE_RATE,
    GameState,
    build_city, build_road, build_settlement,
    can_afford,
    valid_city_vertices, valid_road_edges, valid_settlement_vertices,
)


# ── Hex production value ──────────────────────────────────────────────────────

# Probability of rolling each number on 2d6 (×36 to avoid floats)
_ROLL_PROB_36 = {2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:5, 9:4, 10:3, 11:2, 12:1}

def hex_production_value(hex_id: int) -> float:
    """Expected resources per turn from one settlement on this hex."""
    token = HEX_TOKEN.get(hex_id, 0)
    return _ROLL_PROB_36.get(token, 0) / 36.0


def vertex_production_value(vid: int) -> float:
    """Sum of production values across all hexes adjacent to this vertex."""
    return sum(hex_production_value(hid) for hid in BOARD.vert_to_hexes[vid])


# ── Action scoring ────────────────────────────────────────────────────────────

@dataclass
class BuildAction:
    action_type: str     # "road" | "settlement" | "city"
    target_id: int       # vertex_id or edge_id
    score: float


def score_settlement(vid: int, persona: Persona, state: GameState) -> float:
    prod = vertex_production_value(vid)
    # Settlement opens expansion; weight by expansion_preference
    return prod * (0.5 + 0.5 * persona.expansion_preference) + 1.0  # +1 for VP


def score_city(vid: int, persona: Persona, state: GameState) -> float:
    prod = vertex_production_value(vid)
    # City doubles production; weight by city-preference (1 - expansion_preference)
    return prod * (0.5 + 0.5 * (1.0 - persona.expansion_preference)) + 1.0  # +1 VP


def score_road(eid: int, persona: Persona, state: GameState) -> float:
    # Road value = max settlement value accessible from its endpoints
    best = 0.0
    for vid in BOARD.edges[eid]:
        for nb in BOARD.vert_neighbors[vid]:
            if nb not in state.settlements and nb not in state.cities:
                best = max(best, vertex_production_value(nb))
    return best * persona.expansion_preference * 0.3  # roads have lower direct VP payoff


def available_build_actions(state: GameState, player: str, persona: Persona) -> List[BuildAction]:
    """Return all affordable build actions, scored."""
    actions: List[BuildAction] = []

    if can_afford(state, player, CITY_COST):
        for vid in valid_city_vertices(state, player):
            actions.append(BuildAction("city", vid, score_city(vid, persona, state)))

    if can_afford(state, player, SETTLEMENT_COST):
        for vid in valid_settlement_vertices(state, player):
            actions.append(BuildAction("settlement", vid, score_settlement(vid, persona, state)))

    if can_afford(state, player, ROAD_COST):
        for eid in valid_road_edges(state, player):
            actions.append(BuildAction("road", eid, score_road(eid, persona, state)))

    return sorted(actions, key=lambda a: a.score, reverse=True)


# ── Robber placement ──────────────────────────────────────────────────────────

def choose_robber_hex(state: GameState, active_player: str) -> int:
    """Place robber on the highest-production hex of the leading opponent."""
    # Find leading opponent by VP
    opponents = [p for p in state.vp if p != active_player]
    if not opponents:
        return 0
    leader = max(opponents, key=lambda p: state.vp[p])

    # Find their highest-production hex
    leader_verts = (
        set(v for v, p in state.settlements.items() if p == leader) |
        set(v for v, p in state.cities.items() if p == leader)
    )
    best_hex, best_val = 0, -1.0
    for hid in BOARD.hex_coords:
        if not any(vid in leader_verts for vid in BOARD.hex_to_verts[hid]):
            continue
        val = hex_production_value(hid)
        if val > best_val:
            best_val, best_hex = val, hid
    return best_hex


# ── Greedy turn executor ──────────────────────────────────────────────────────

def greedy_build_phase(state: GameState, player: str, persona: Persona) -> int:
    """Execute greedy builds until no affordable action remains. Returns build count."""
    built = 0
    while True:
        actions = available_build_actions(state, player, persona)
        if not actions:
            break
        best = actions[0]
        if best.action_type == "city":
            ok = build_city(state, player, best.target_id)
        elif best.action_type == "settlement":
            ok = build_settlement(state, player, best.target_id)
        else:
            ok = build_road(state, player, best.target_id)
        if ok:
            built += 1
        else:
            break  # safety: avoid infinite loop if build unexpectedly fails
    return built


def greedy_bank_trade(state: GameState, player: str, persona: Persona) -> bool:
    """Execute one bank trade if it enables an otherwise-unaffordable build. Returns True if traded."""
    # Check what we can't afford but want
    want_city = bool(valid_city_vertices(state, player)) and not can_afford(state, player, CITY_COST)
    want_settlement = bool(valid_settlement_vertices(state, player)) and not can_afford(state, player, SETTLEMENT_COST)

    if not (want_city or want_settlement):
        return False

    target_cost = CITY_COST if want_city else SETTLEMENT_COST
    needed = {r: max(0, amt - state.resources[player].get(r, 0)) for r, amt in target_cost.items()}
    needed = {r: n for r, n in needed.items() if n > 0}
    if not needed:
        return False

    # Find a resource we have >= 3 of that isn't needed
    surplus = {r: state.resources[player].get(r, 0) for r in RESOURCES
               if state.resources[player].get(r, 0) >= BANK_TRADE_RATE and r not in needed}
    if not surplus:
        return False

    give_res = max(surplus, key=surplus.get)
    receive_res = next(iter(needed))
    state.resources[player][give_res] -= BANK_TRADE_RATE
    state.resources[player][receive_res] += 1
    return True
