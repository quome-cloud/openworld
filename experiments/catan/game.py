"""Full game loop for the simplified Catan variant.

run_game() runs one complete game under a given coordination condition and
returns the final GameState.  Condition (a) = independent greedy (no alliance
communication) is implemented here.  Conditions (b)/(c)/(d) plug in via the
turn_runner parameter.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Optional

from .board import BOARD, HEX_RESOURCE, HEX_TOKEN
from .personas import Persona
from .policy import (
    _ROLL_PROB_36,
    choose_robber_hex,
    greedy_bank_trade,
    greedy_build_phase,
    vertex_production_value,
)
from .state import (
    PLAYERS,
    MAX_TURNS,
    GameState,
    apply_robber,
    build_road,
    build_settlement,
    check_win,
    initial_state,
    roll_and_produce,
)


# ── Setup helpers ─────────────────────────────────────────────────────────────

def _free_vertices_strict(state: GameState) -> list:
    """Vertices with no settlement/city and no occupied neighbor (distance rule)."""
    blocked: set = set()
    for vid in list(state.settlements) + list(state.cities):
        blocked.add(vid)
        blocked.update(BOARD.vert_neighbors[vid])
    return [v for v in BOARD.vert_to_hexes if v not in blocked]


def _free_vertices_relaxed(state: GameState) -> list:
    """Vertices with no settlement/city (distance rule waived — small board fallback)."""
    occupied = set(state.settlements) | set(state.cities)
    return [v for v in BOARD.vert_to_hexes if v not in occupied]


def _best_vertex(candidates: list) -> int:
    return max(candidates, key=vertex_production_value)


def _adjacent_free_edges(state: GameState, vid: int) -> list:
    """Edges adjacent to vid that have no road yet."""
    return [eid for eid in BOARD.vert_to_edges[vid] if eid not in state.roads]


def _setup_place(state: GameState, player: str, rng: random.Random, grant: bool) -> None:
    """Place one settlement + adjacent road during setup (no resource cost)."""
    candidates = _free_vertices_strict(state) or _free_vertices_relaxed(state)
    if not candidates:
        return

    vid = _best_vertex(candidates)
    # Direct placement — bypasses resource check and distance-rule re-check in build_settlement
    state.settlements[vid] = player
    state.vp[player] += 1

    # Road adjacent to the just-placed settlement (free, no resource cost)
    adj = _adjacent_free_edges(state, vid)
    if adj:
        eid = rng.choice(adj)
        state.roads[eid] = player

    if grant:
        for hid in BOARD.vert_to_hexes[vid]:
            res = HEX_RESOURCE.get(hid)
            if res:
                state.resources[player][res] += 1


def run_setup(state: GameState, rng: random.Random) -> None:
    """Reverse-snake setup: 1→2→3→4→4→3→2→1; second round grants starting resources."""
    order = list(PLAYERS) + list(reversed(PLAYERS))
    for i, player in enumerate(order):
        _setup_place(state, player, rng, grant=(i >= len(PLAYERS)))
    state.phase = "roll"
    state.active_player = PLAYERS[0]


# ── Greedy turn ───────────────────────────────────────────────────────────────

def run_turn_greedy(
    state: GameState,
    player: str,
    personas: Dict[str, Persona],
    rng: random.Random,
) -> None:
    """One greedy turn for player under condition (a): no inter-player communication."""
    persona = personas[player]

    roll = roll_and_produce(state, rng)

    if roll == 7:
        target_hex = choose_robber_hex(state, player)
        apply_robber(state, player, target_hex, rng)

    # Up to 2 bank trade attempts per turn (helps resource conversion)
    greedy_bank_trade(state, player, persona)
    greedy_bank_trade(state, player, persona)

    greedy_build_phase(state, player, persona)

    state.turn += 1


# ── Full game ─────────────────────────────────────────────────────────────────

TurnRunner = Callable[[GameState, str, Dict[str, Persona], random.Random], None]


def run_game(
    personas: Dict[str, Persona],
    rng: random.Random,
    turn_runner: TurnRunner = run_turn_greedy,
) -> GameState:
    """Run one complete game.  Returns the terminal GameState."""
    state = initial_state()
    run_setup(state, rng)

    n = len(PLAYERS)
    for _ in range(MAX_TURNS):
        player = PLAYERS[state.turn % n]
        state.active_player = player
        turn_runner(state, player, personas, rng)

        winner = check_win(state)
        if winner:
            state.game_over = True
            state.winner = winner
            state.phase = "done"
            return state

    # Safety cap: winner by most VP
    leader = max(PLAYERS, key=lambda p: state.vp[p])
    state.game_over = True
    state.winner = leader
    state.phase = "done"
    return state
