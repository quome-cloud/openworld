"""Coordination conditions (b), (c), (d) for the Catan alliance.

Condition (a) is in game.py (run_turn_greedy).
Condition (b): shared pre-game strategy, fixed for entire game.
Condition (c): Polis-style per-turn ranked-proposal reconciliation.
Condition (d): Habermas LLM mediator with veto-and-fallback (stubs to real LLM).

All conditions expose a TurnRunner compatible with run_game(turn_runner=...).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .board import BOARD, HEX_RESOURCE, HEX_TOKEN, GRAIN, STONE, WOOD
from .game import TurnRunner, run_turn_greedy
from .personas import ALLIANCE_PLAYERS, Persona
from .policy import (
    available_build_actions,
    choose_robber_hex,
    greedy_bank_trade,
    greedy_build_phase,
    vertex_production_value,
)
from .state import (
    PLAYERS,
    GameState,
    apply_robber,
    build_city,
    build_road,
    build_settlement,
    can_afford,
    roll_and_produce,
    valid_city_vertices,
    valid_road_edges,
    valid_settlement_vertices,
    ROAD_COST,
    SETTLEMENT_COST,
    CITY_COST,
)


# ── Condition (b): shared pre-game strategy ───────────────────────────────────

@dataclass
class AllianceStrategy:
    """Pre-game strategy agreed by P1 and P2 before turn 1.

    Rule-based analogue of the single joint LLM call in the design doc.
    The strategy assigns roles based on starting production profiles and sets
    an intra-alliance trade protocol.
    """
    settler: str           # which alliance member focuses on roads+settlements
    builder: str           # which focuses on city upgrades
    trade_give: str        # resource settler offers to builder on request
    trade_receive: str     # resource builder offers to settler on request
    target_resource: Dict[str, str]  # player → preferred resource to accumulate


def compute_alliance_strategy(state: GameState) -> AllianceStrategy:
    """Derive a pre-game alliance strategy from the initial board state."""
    # Measure each alliance member's production profile
    production: Dict[str, Dict[str, float]] = {}
    for player in ALLIANCE_PLAYERS:
        prod = {STONE: 0.0, WOOD: 0.0, GRAIN: 0.0}
        for vid, owner in state.settlements.items():
            if owner != player:
                continue
            for hid in BOARD.vert_to_hexes[vid]:
                res = HEX_RESOURCE.get(hid)
                if res:
                    tok = HEX_TOKEN.get(hid, 0)
                    from .policy import _ROLL_PROB_36
                    prod[res] += _ROLL_PROB_36.get(tok, 0) / 36.0
        production[player] = prod

    # Assign roles: the player producing more WOOD+STONE is the settler (needs
    # those for roads/settlements); the other is the builder (needs GRAIN+STONE
    # for cities).
    def road_affinity(p: str) -> float:
        return production[p][WOOD] + production[p][STONE]

    p1_aff = road_affinity(ALLIANCE_PLAYERS[0])
    p2_aff = road_affinity(ALLIANCE_PLAYERS[1])

    settler = ALLIANCE_PLAYERS[0] if p1_aff >= p2_aff else ALLIANCE_PLAYERS[1]
    builder = ALLIANCE_PLAYERS[1] if settler == ALLIANCE_PLAYERS[0] else ALLIANCE_PLAYERS[0]

    # Trade protocol: settler offers surplus STONE to builder (both need it);
    # builder offers surplus GRAIN to settler (settler rarely produces grain).
    return AllianceStrategy(
        settler=settler,
        builder=builder,
        trade_give=STONE,    # settler trades STONE → builder when settler has 3+
        trade_receive=GRAIN, # builder trades GRAIN → settler when builder has 3+
        target_resource={
            settler: WOOD,
            builder: GRAIN,
        },
    )


def _execute_alliance_trade(
    state: GameState,
    active: str,
    strategy: AllianceStrategy,
    turn: int,
) -> None:
    """Attempt one intra-alliance trade per the pre-game agreement."""
    ally = ALLIANCE_PLAYERS[1] if active == ALLIANCE_PLAYERS[0] else ALLIANCE_PLAYERS[0]
    # Active player offers their trade_give resource to ally if they have 2+ surplus
    if active == strategy.settler:
        give, receive = strategy.trade_give, strategy.trade_receive
    else:
        give, receive = strategy.trade_receive, strategy.trade_give

    if state.resources[active].get(give, 0) >= 2 and state.resources[ally].get(receive, 0) >= 1:
        # Execute 1-for-1 trade
        from .state import bilateral_trade
        bilateral_trade(state, active, ally, {give: 1}, {receive: 1}, turn=turn, accepted=True)


def make_condition_b_runner(strategy_ref: List[Optional[AllianceStrategy]]) -> TurnRunner:
    """Return a TurnRunner for condition (b).

    strategy_ref is a 1-element list; the strategy is computed lazily on the
    first alliance turn and cached there.
    """
    def run_turn(
        state: GameState,
        player: str,
        personas: Dict[str, Persona],
        rng: random.Random,
    ) -> None:
        # Non-alliance players use the baseline greedy policy
        if player not in ALLIANCE_PLAYERS:
            run_turn_greedy(state, player, personas, rng)
            return

        # Compute strategy once on first alliance turn
        if strategy_ref[0] is None:
            strategy_ref[0] = compute_alliance_strategy(state)
        strategy = strategy_ref[0]

        roll = roll_and_produce(state, rng)
        if roll == 7:
            target_hex = choose_robber_hex(state, player)
            apply_robber(state, player, target_hex, rng)

        # Alliance trade before building (per pre-game agreement)
        _execute_alliance_trade(state, player, strategy, turn=state.turn)

        # Bank trades (2 attempts)
        greedy_bank_trade(state, player, personas[player])
        greedy_bank_trade(state, player, personas[player])

        # Build: use greedy policy (strategy shapes resources via trades)
        greedy_build_phase(state, player, personas[player])

        state.turn += 1

    return run_turn


def condition_b_runner() -> TurnRunner:
    """Factory: fresh condition-(b) runner with empty strategy cache."""
    return make_condition_b_runner([None])


# ── Condition (c): Polis-style per-turn reconciliation ────────────────────────

def _top_k_actions(state, player, persona, k=3):
    """Return top-k scored build actions for player."""
    return available_build_actions(state, player, persona)[:k]


def run_turn_condition_c(
    state: GameState,
    player: str,
    personas: Dict[str, Persona],
    rng: random.Random,
) -> None:
    """Condition (c): Polis-style reconciliation between alliance members."""
    if player not in ALLIANCE_PLAYERS:
        run_turn_greedy(state, player, personas, rng)
        return

    ally = ALLIANCE_PLAYERS[1] if player == ALLIANCE_PLAYERS[0] else ALLIANCE_PLAYERS[0]

    roll = roll_and_produce(state, rng)
    if roll == 7:
        target_hex = choose_robber_hex(state, player)
        apply_robber(state, player, target_hex, rng)

    greedy_bank_trade(state, player, personas[player])
    greedy_bank_trade(state, player, personas[player])

    # Both players propose their top-3 actions
    my_top3 = _top_k_actions(state, player, personas[player])
    ally_top3 = _top_k_actions(state, ally, personas[ally])

    my_ids = [(a.action_type, a.target_id) for a in my_top3]
    ally_ids = [(a.action_type, a.target_id) for a in ally_top3]

    # Intersection check: find actions in both top-3s
    intersection = [a for a in my_top3 if (a.action_type, a.target_id) in ally_ids]

    if intersection:
        # Execute highest-mutual-endorsement action first
        best = min(
            intersection,
            key=lambda a: my_ids.index((a.action_type, a.target_id))
                         + ally_ids.index((a.action_type, a.target_id))
        )
        _execute_action(state, player, best)

    # Then greedy-fill remaining build capacity
    greedy_build_phase(state, player, personas[player])

    state.turn += 1


def _execute_action(state: GameState, player: str, action) -> bool:
    """Execute a single BuildAction for player. Returns True on success."""
    if action.action_type == "city":
        return build_city(state, player, action.target_id)
    elif action.action_type == "settlement":
        return build_settlement(state, player, action.target_id)
    else:
        return build_road(state, player, action.target_id)


# ── Condition (d): Habermas LLM mediator stub ─────────────────────────────────

def run_turn_condition_d(
    state: GameState,
    player: str,
    personas: Dict[str, Persona],
    rng: random.Random,
    llm_call=None,
) -> None:
    """Condition (d): Habermas LLM mediator with veto-and-fallback.

    llm_call(prompt: str) -> str  — if None, falls back to condition (c).
    This keeps condition (d) testable without a live LLM.
    """
    if player not in ALLIANCE_PLAYERS or llm_call is None:
        # Fallback: condition (c) behaviour
        run_turn_condition_c(state, player, personas, rng)
        return

    ally = ALLIANCE_PLAYERS[1] if player == ALLIANCE_PLAYERS[0] else ALLIANCE_PLAYERS[0]

    roll = roll_and_produce(state, rng)
    if roll == 7:
        target_hex = choose_robber_hex(state, player)
        apply_robber(state, player, target_hex, rng)

    greedy_bank_trade(state, player, personas[player])
    greedy_bank_trade(state, player, personas[player])

    # Step 1: private submissions (top-1 preferred action + one-line rationale)
    my_top = _top_k_actions(state, player, personas[player], k=1)
    ally_top = _top_k_actions(state, ally, personas[ally], k=1)

    if not my_top or not ally_top:
        greedy_build_phase(state, player, personas[player])
        state.turn += 1
        return

    # Step 2: LLM mediator synthesis
    prompt = (
        f"Alliance turn {state.turn}. Active: {player}, Ally: {ally}.\n"
        f"VP: {dict(state.vp)}. Resources: {dict(state.resources)}.\n"
        f"{player} proposes: {my_top[0].action_type} on {my_top[0].target_id}.\n"
        f"{ally} recommends: {ally_top[0].action_type} on {ally_top[0].target_id}.\n"
        f"Reply with the single best action as: ACTION_TYPE TARGET_ID\n"
        f"(e.g. 'settlement 5' or 'city 3' or 'road 12')"
    )
    response = llm_call(prompt)

    # Parse mediator response
    mediator_action = None
    try:
        parts = response.strip().split()
        if len(parts) == 2:
            atype, tid = parts[0].lower(), int(parts[1])
            # Find matching action in my_top3
            all_top = _top_k_actions(state, player, personas[player], k=5)
            matches = [a for a in all_top if a.action_type == atype and a.target_id == tid]
            if matches:
                mediator_action = matches[0]
    except (ValueError, IndexError):
        pass

    # Step 3: Both vote (simplified: active player vetoes if mediator chose
    # something NOT in their top-3, ally vetoes if not in their top-3)
    my_top3_ids = {(a.action_type, a.target_id) for a in _top_k_actions(state, player, personas[player])}
    ally_top3_ids = {(a.action_type, a.target_id) for a in _top_k_actions(state, ally, personas[ally])}

    vetoed = mediator_action is None or (
        (mediator_action.action_type, mediator_action.target_id) not in my_top3_ids
        or (mediator_action.action_type, mediator_action.target_id) not in ally_top3_ids
    )

    from .state import CoordEvent
    if not vetoed and mediator_action:
        state.coord_log.append(CoordEvent(
            turn=state.turn, condition="d", active_player=player,
            event_type="accept", payload=f"{mediator_action.action_type} {mediator_action.target_id}"
        ))
        _execute_action(state, player, mediator_action)
    else:
        state.coord_log.append(CoordEvent(
            turn=state.turn, condition="d", active_player=player,
            event_type="veto", payload=None
        ))
        # Fallback to condition (c) reconciliation
        my_top3 = _top_k_actions(state, player, personas[player])
        ally_top3_list = _top_k_actions(state, ally, personas[ally])
        ally_ids = [(a.action_type, a.target_id) for a in ally_top3_list]
        intersection = [a for a in my_top3 if (a.action_type, a.target_id) in ally_ids]
        if intersection:
            _execute_action(state, player, intersection[0])

    greedy_build_phase(state, player, personas[player])
    state.turn += 1
