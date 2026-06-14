"""GameState and transition function for the simplified 7-hex Catan variant."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from .board import BOARD, GRAIN, RESOURCES, STONE, WOOD, HEX_RESOURCE, HEX_TOKEN

# ── Constants ─────────────────────────────────────────────────────────────────

PLAYERS = ("P1", "P2", "P3", "P4")
WIN_VP = 7
MAX_TURNS = 120  # safety cap to prevent infinite games

# Build costs
ROAD_COST = {WOOD: 1, STONE: 1}
SETTLEMENT_COST = {WOOD: 1, GRAIN: 1, STONE: 1}
CITY_COST = {STONE: 2, GRAIN: 2}

BANK_TRADE_RATE = 3  # 3:1 for any resource

# ── Trade / coord log entries ─────────────────────────────────────────────────

@dataclass
class Trade:
    turn: int
    from_player: str
    to_player: str           # "bank" for bank trades
    give: Dict[str, int]
    receive: Dict[str, int]
    accepted: bool


@dataclass
class CoordEvent:
    turn: int
    condition: str           # "b", "c", or "d"
    active_player: str
    event_type: str          # "proposal", "mediator_call", "vote", "accept", "veto", "fallback"
    payload: Optional[str] = None   # JSON string for proposals / mediator output


# ── Game state ────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    # Board occupancy
    settlements: Dict[int, str] = field(default_factory=dict)   # vertex_id → player_id
    cities: Dict[int, str] = field(default_factory=dict)         # vertex_id → player_id (upgrades)
    roads: Dict[int, str] = field(default_factory=dict)          # edge_id → player_id

    # Resources: player → resource → count
    resources: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # VP tracking
    vp: Dict[str, int] = field(default_factory=dict)

    # Robber
    robber_hex: Optional[int] = None   # None = no robber active

    # Turn tracking
    turn: int = 0
    active_player: str = "P1"
    phase: str = "setup"               # "setup" | "roll" | "trade" | "build" | "done"

    # Last dice roll (None before first roll this turn)
    dice_roll: Optional[int] = None

    # Logs (visible to all players)
    trade_log: List[Trade] = field(default_factory=list)
    coord_log: List[CoordEvent] = field(default_factory=list)

    # Terminal state
    game_over: bool = False
    winner: Optional[str] = None

    def copy(self) -> "GameState":
        import copy
        return copy.deepcopy(self)


def initial_state() -> GameState:
    """Return a blank pre-setup game state."""
    s = GameState()
    for p in PLAYERS:
        s.resources[p] = {r: 0 for r in RESOURCES}
        s.vp[p] = 0
    s.phase = "setup"
    s.active_player = PLAYERS[0]
    return s


# ── Placement helpers ─────────────────────────────────────────────────────────

def vertex_is_free(state: GameState, vid: int) -> bool:
    """True if vid has no settlement or city.

    Distance rule is omitted: the 7-hex board (18 vertices, 8 setup settlements)
    leaves 0 free vertices under the strict Catan distance rule, preventing any
    expansion.  For this research simulator the relevant mechanics are resource
    production and coordination decisions, not territory exclusion.
    """
    return vid not in state.settlements and vid not in state.cities


def player_road_network(state: GameState, player: str) -> set:
    """Return set of vertex IDs reachable via the player's road network."""
    road_verts: set = set()
    for eid, owner in state.roads.items():
        if owner == player:
            for vid in BOARD.edges[eid]:
                road_verts.add(vid)
    return road_verts


def valid_settlement_vertices(state: GameState, player: str, setup: bool = False) -> List[int]:
    """Return vertex IDs where player can legally place a settlement."""
    reachable = BOARD.vert_to_hexes.keys() if setup else player_road_network(state, player)
    return [
        vid for vid in reachable
        if vertex_is_free(state, vid)
    ]


def valid_road_edges(state: GameState, player: str, setup: bool = False) -> List[int]:
    """Return edge IDs where player can legally place a road."""
    if setup:
        # During setup, road must connect to a just-placed settlement.
        # Caller handles this externally; return all edges adjacent to player settlements.
        player_verts = {v for v, p in state.settlements.items() if p == player}
    else:
        player_verts = {v for v, p in state.settlements.items() if p == player}
        player_verts |= {v for v, p in state.cities.items() if p == player}
        player_verts |= player_road_network(state, player)

    valid = []
    for eid, edge in BOARD.edges.items():
        if eid in state.roads:
            continue
        if edge & player_verts:
            valid.append(eid)
    return valid


def valid_city_vertices(state: GameState, player: str) -> List[int]:
    """Return vertex IDs where player can upgrade a settlement to a city."""
    return [v for v, p in state.settlements.items() if p == player]


# ── Resource production ───────────────────────────────────────────────────────

def produce_resources(state: GameState, roll: int) -> Dict[str, Dict[str, int]]:
    """Compute resource grants for a given dice roll.  Does NOT modify state."""
    grants: Dict[str, Dict[str, int]] = {p: {r: 0 for r in RESOURCES} for p in PLAYERS}

    for hid, token in HEX_TOKEN.items():
        if token != roll:
            continue
        if state.robber_hex == hid:
            continue
        resource = HEX_RESOURCE[hid]
        for vid in BOARD.hex_to_verts[hid]:
            if vid in state.settlements:
                owner = state.settlements[vid]
                grants[owner][resource] += 1
            elif vid in state.cities:
                owner = state.cities[vid]
                grants[owner][resource] += 2

    return grants


def apply_production(state: GameState, grants: Dict[str, Dict[str, int]]) -> None:
    """Apply resource grants to state in-place."""
    for player, earned in grants.items():
        for resource, amount in earned.items():
            state.resources[player][resource] += amount


# ── Build actions ─────────────────────────────────────────────────────────────

def can_afford(state: GameState, player: str, cost: Dict[str, int]) -> bool:
    return all(state.resources[player].get(r, 0) >= amt for r, amt in cost.items())


def deduct_cost(state: GameState, player: str, cost: Dict[str, int]) -> None:
    for r, amt in cost.items():
        state.resources[player][r] -= amt


def build_road(state: GameState, player: str, edge_id: int) -> bool:
    """Place a road.  Returns True on success."""
    if edge_id in state.roads:
        return False
    if not can_afford(state, player, ROAD_COST):
        return False
    if edge_id not in valid_road_edges(state, player):
        return False
    deduct_cost(state, player, ROAD_COST)
    state.roads[edge_id] = player
    return True


def build_settlement(state: GameState, player: str, vertex_id: int, free: bool = False) -> bool:
    """Place a settlement.  Set free=True during setup (no resource cost)."""
    if vertex_id not in BOARD.vert_to_hexes:
        return False
    if not vertex_is_free(state, vertex_id):
        return False
    if not free and not can_afford(state, player, SETTLEMENT_COST):
        return False
    if not free and vertex_id not in valid_settlement_vertices(state, player):
        return False
    if not free:
        deduct_cost(state, player, SETTLEMENT_COST)
    state.settlements[vertex_id] = player
    state.vp[player] += 1
    return True


def build_city(state: GameState, player: str, vertex_id: int) -> bool:
    """Upgrade a settlement to a city."""
    if state.settlements.get(vertex_id) != player:
        return False
    if not can_afford(state, player, CITY_COST):
        return False
    deduct_cost(state, player, CITY_COST)
    del state.settlements[vertex_id]
    state.cities[vertex_id] = player
    state.vp[player] += 1  # net +1 (settlement was already +1)
    return True


# ── Bank trade ────────────────────────────────────────────────────────────────

def bank_trade(state: GameState, player: str, give_resource: str, receive_resource: str,
               turn: int) -> bool:
    """3:1 bank trade.  Returns True on success."""
    if state.resources[player].get(give_resource, 0) < BANK_TRADE_RATE:
        return False
    if give_resource == receive_resource:
        return False
    state.resources[player][give_resource] -= BANK_TRADE_RATE
    state.resources[player][receive_resource] += 1
    state.trade_log.append(Trade(
        turn=turn,
        from_player=player,
        to_player="bank",
        give={give_resource: BANK_TRADE_RATE},
        receive={receive_resource: 1},
        accepted=True,
    ))
    return True


# ── Bilateral trade ───────────────────────────────────────────────────────────

def bilateral_trade(
    state: GameState,
    from_player: str,
    to_player: str,
    give: Dict[str, int],
    receive: Dict[str, int],
    turn: int,
    accepted: bool,
) -> bool:
    """Record a bilateral trade offer; if accepted, apply the exchange."""
    state.trade_log.append(Trade(
        turn=turn,
        from_player=from_player,
        to_player=to_player,
        give=give,
        receive=receive,
        accepted=accepted,
    ))
    if not accepted:
        return False
    # Validate both sides can afford it
    for r, amt in give.items():
        if state.resources[from_player].get(r, 0) < amt:
            return False
    for r, amt in receive.items():
        if state.resources[to_player].get(r, 0) < amt:
            return False
    # Apply
    for r, amt in give.items():
        state.resources[from_player][r] -= amt
        state.resources[to_player][r] += amt
    for r, amt in receive.items():
        state.resources[to_player][r] -= amt
        state.resources[from_player][r] += amt
    return True


# ── Robber ────────────────────────────────────────────────────────────────────

def apply_robber(state: GameState, active_player: str, target_hex: int, rng: random.Random) -> None:
    """Move robber; steal a random card from an adjacent player."""
    state.robber_hex = target_hex
    # Find players with settlements/cities adjacent to target hex
    targets = []
    for vid in BOARD.hex_to_verts[target_hex]:
        owner = state.settlements.get(vid) or state.cities.get(vid)
        if owner and owner != active_player and owner not in targets:
            targets.append(owner)
    if not targets:
        return
    victim = rng.choice(targets)
    # Steal 1 random card
    hand = [(r, c) for r, c in state.resources[victim].items() if c > 0]
    if not hand:
        return
    stolen_res, _ = rng.choice(hand)
    state.resources[victim][stolen_res] -= 1
    state.resources[active_player][stolen_res] += 1


def apply_discard_rule(state: GameState, rng: random.Random) -> None:
    """On a 7 roll: players with >7 cards discard half (rounded down)."""
    for player in PLAYERS:
        total = sum(state.resources[player].values())
        if total > 7:
            n_discard = total // 2
            # Discard randomly from hand
            hand = []
            for r, c in state.resources[player].items():
                hand.extend([r] * c)
            rng.shuffle(hand)
            for r in hand[:n_discard]:
                state.resources[player][r] -= 1


# ── Win check ─────────────────────────────────────────────────────────────────

def check_win(state: GameState) -> Optional[str]:
    """Return winning player ID or None."""
    for player in PLAYERS:
        if state.vp[player] >= WIN_VP:
            return player
    return None


# ── Turn roll phase ───────────────────────────────────────────────────────────

def roll_and_produce(state: GameState, rng: random.Random) -> int:
    """Roll dice, apply production (or robber).  Returns the roll value."""
    d1 = rng.randint(1, 6)
    d2 = rng.randint(1, 6)
    roll = d1 + d2
    state.dice_roll = roll

    if roll == 7:
        apply_discard_rule(state, rng)
        # Robber placement is handled by the agent strategy; default: no-op
        # (agents call apply_robber explicitly)
    else:
        grants = produce_resources(state, roll)
        apply_production(state, grants)

    return roll
