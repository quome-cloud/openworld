"""Adversarial counter-alliance condition for the Catan simulator.

P3 and P4 form a rule-based counter-alliance targeting P1+P2 when the alliance
is leading (combined VP >= P3+P4 combined VP after turn 3).

Counter-strategies (per design doc §4):
1. Territorial block: build roads/settlements to deny P1+P2 high-value spots.
2. Wedge trade: offer P1 a trade that starves P2 of a needed resource.
3. Robber targeting: place robber on P1/P2's highest-production hex.
P3 handles blocking, P4 handles robber; both may wedge-trade.

This module wraps any base TurnRunner and adds counter-alliance logic for P3/P4.
"""

from __future__ import annotations

import random
from typing import Dict

from .board import BOARD, HEX_TOKEN
from .game import TurnRunner, run_turn_greedy
from .personas import ALLIANCE_PLAYERS, Persona
from .policy import (
    choose_robber_hex,
    greedy_bank_trade,
    greedy_build_phase,
    vertex_production_value,
)
from .state import (
    PLAYERS,
    GameState,
    apply_robber,
    build_road,
    build_settlement,
    bilateral_trade,
    can_afford,
    roll_and_produce,
    valid_road_edges,
    valid_settlement_vertices,
    ROAD_COST,
    SETTLEMENT_COST,
)

COUNTER_ALLIANCE = ("P3", "P4")
_ACTIVATION_TURN = 3 * len(PLAYERS)  # after round 3


def _counter_alliance_active(state: GameState) -> bool:
    """Activate when alliance leads on combined VP after turn 3."""
    if state.turn < _ACTIVATION_TURN:
        return False
    alliance_vp = sum(state.vp[p] for p in ALLIANCE_PLAYERS)
    counter_vp = sum(state.vp[p] for p in COUNTER_ALLIANCE)
    return alliance_vp >= counter_vp


def _block_best_alliance_vertex(state: GameState, player: str) -> bool:
    """P3: place road/settlement to deny high-value vertex to alliance."""
    # Find the highest-value free vertex adjacent to alliance road network
    alliance_verts: set = set()
    for vid, owner in {**state.settlements, **state.cities}.items():
        if owner in ALLIANCE_PLAYERS:
            alliance_verts.add(vid)
    for eid, owner in state.roads.items():
        if owner in ALLIANCE_PLAYERS:
            for vid in BOARD.edges[eid]:
                alliance_verts.add(vid)

    # Target: vertices reachable by alliance (neighbors of their network)
    targets = set()
    for vid in alliance_verts:
        for nb in BOARD.vert_neighbors[vid]:
            if nb not in state.settlements and nb not in state.cities:
                targets.add(nb)

    if not targets:
        return False

    best_target = max(targets, key=vertex_production_value)

    # Can player build a road toward that target?
    if can_afford(state, player, ROAD_COST):
        player_verts = (
            {v for v, p in state.settlements.items() if p == player}
            | {v for v, p in state.cities.items() if p == player}
        )
        for eid in state.roads:
            if state.roads[eid] == player:
                for vid in BOARD.edges[eid]:
                    player_verts.add(vid)
        # Find an edge leading toward best_target
        for eid in valid_road_edges(state, player):
            for vid in BOARD.edges[eid]:
                if vid == best_target or best_target in BOARD.vert_neighbors.get(vid, []):
                    build_road(state, player, eid)
                    return True
    return False


def _wedge_trade(state: GameState, player: str, turn: int) -> bool:
    """Offer P1 a trade favorable to P1 but draining P2's needed resource.

    P2 needs GRAIN+STONE for cities. Offer P1 a 1:1 Wood→Stone trade
    (good for P1's roads) to drain P2's Stone availability via opportunity.
    Simplified: drain P2's most-stocked resource via a trade with P1.
    """
    p1, p2 = ALLIANCE_PLAYERS
    # Find P2's most-stocked resource
    p2_res = state.resources[p2]
    if not any(v > 0 for v in p2_res.values()):
        return False

    drain_res = max(p2_res, key=p2_res.get)
    # Offer P1 a favorable trade: give 1 of drain_res for 1 of what player has
    if state.resources[player].get(drain_res, 0) == 0:
        return False
    if state.resources[p1].get(drain_res, 0) == 0:
        return False  # P1 has nothing to drain

    from .board import WOOD, STONE, GRAIN
    # Offer P1: give 1 Wood → receive 1 of drain_res from P1
    # This reduces P2's potential alliance-trade pool
    give_res = drain_res
    for offer_res in (WOOD, STONE, GRAIN):
        if offer_res == give_res:
            continue
        if state.resources[player].get(offer_res, 0) >= 1 and state.resources[p1].get(give_res, 0) >= 1:
            # P1 accepts any positive-EV offer (trade_openness_adversary=0.5 by default)
            bilateral_trade(state, player, p1,
                            give={offer_res: 1}, receive={give_res: 1},
                            turn=turn, accepted=True)
            return True
    return False


def _counter_robber(state: GameState, player: str, rng: random.Random) -> None:
    """P4: place robber on P1/P2's highest-production hex."""
    # Find highest-production hex occupied by alliance
    best_hex, best_val = 0, -1.0
    for hid in BOARD.hex_coords:
        alliance_present = any(
            vid in state.settlements or vid in state.cities
            for vid in BOARD.hex_to_verts[hid]
            if (state.settlements.get(vid) or state.cities.get(vid)) in ALLIANCE_PLAYERS
        )
        # Simpler check
        has_alliance = False
        for vid in BOARD.hex_to_verts[hid]:
            if state.settlements.get(vid) in ALLIANCE_PLAYERS or \
               state.cities.get(vid) in ALLIANCE_PLAYERS:
                has_alliance = True
                break
        if not has_alliance:
            continue
        from .policy import hex_production_value
        val = hex_production_value(hid)
        if val > best_val:
            best_val, best_hex = val, hid
    apply_robber(state, player, best_hex, rng)


def make_adversarial_runner(base_runner: TurnRunner) -> TurnRunner:
    """Wrap a base TurnRunner to add counter-alliance logic for P3/P4."""

    def run_turn(
        state: GameState,
        player: str,
        personas: Dict[str, Persona],
        rng: random.Random,
    ) -> None:
        if player not in COUNTER_ALLIANCE or not _counter_alliance_active(state):
            base_runner(state, player, personas, rng)
            return

        roll = roll_and_produce(state, rng)

        if roll == 7:
            if player == "P4":
                _counter_robber(state, player, rng)
            else:
                # P3: standard greedy robber
                target_hex = choose_robber_hex(state, player)
                apply_robber(state, player, target_hex, rng)

        # Wedge trade attempt
        _wedge_trade(state, player, turn=state.turn)

        # P3: territorial blocking; P4: greedy build
        if player == "P3":
            _block_best_alliance_vertex(state, player)

        greedy_bank_trade(state, player, personas[player])
        greedy_build_phase(state, player, personas[player])

        state.turn += 1

    return run_turn
