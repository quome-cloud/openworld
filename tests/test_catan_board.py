"""Tests for Catan board topology and game state (Cycle 1)."""

import random

import pytest

from experiments.catan.board import BOARD, GRAIN, RESOURCES, STONE, WOOD, HEX_RESOURCE, HEX_TOKEN
from experiments.catan.state import (
    PLAYERS,
    WIN_VP,
    GameState,
    apply_production,
    bank_trade,
    bilateral_trade,
    build_city,
    build_road,
    build_settlement,
    can_afford,
    check_win,
    initial_state,
    produce_resources,
    roll_and_produce,
    vertex_is_free,
)


# ── Board topology tests ──────────────────────────────────────────────────────

class TestBoardTopology:
    def test_seven_hexes(self):
        assert len(BOARD.hex_coords) == 7

    def test_center_hex_exists(self):
        assert 6 in BOARD.hex_coords
        assert BOARD.hex_coords[6] == (0, 0)

    def test_each_hex_has_six_vertices(self):
        for hid, verts in BOARD.hex_to_verts.items():
            assert len(verts) == 6, f"hex {hid} has {len(verts)} vertices"

    def test_vertex_count_reasonable(self):
        # 7-hex board should have between 18 and 30 unique vertices
        n = BOARD.num_vertices
        assert 18 <= n <= 30, f"unexpected vertex count: {n}"

    def test_center_hex_shares_vertices_with_all_ring_hexes(self):
        center_verts = set(BOARD.hex_to_verts[6])
        for ring_hid in range(6):
            ring_verts = set(BOARD.hex_to_verts[ring_hid])
            shared = center_verts & ring_verts
            assert len(shared) >= 2, f"ring hex {ring_hid} shares <2 verts with center"

    def test_vert_to_hexes_consistent(self):
        for hid, verts in BOARD.hex_to_verts.items():
            for vid in verts:
                assert hid in BOARD.vert_to_hexes[vid], \
                    f"hex {hid} missing from vert_to_hexes[{vid}]"

    def test_edge_endpoints_are_valid_vertices(self):
        all_verts = set(BOARD.vert_to_hexes.keys())
        for eid, edge in BOARD.edges.items():
            for vid in edge:
                assert vid in all_verts

    def test_vert_neighbors_symmetric(self):
        for vid, neighbors in BOARD.vert_neighbors.items():
            for nb in neighbors:
                assert vid in BOARD.vert_neighbors[nb], \
                    f"asymmetric: {vid}→{nb} but not {nb}→{vid}"

    def test_resource_distribution(self):
        resources = list(HEX_RESOURCE.values())
        assert resources.count(STONE) == 3
        assert resources.count(WOOD) == 2
        assert resources.count(GRAIN) == 2

    def test_token_uniqueness_and_range(self):
        tokens = list(HEX_TOKEN.values())
        assert len(set(tokens)) == len(tokens), "duplicate tokens"
        for t in tokens:
            assert 2 <= t <= 12, f"token {t} out of valid 2d6 range"
        assert 7 not in tokens, "7 should not be a number token (robber roll)"


# ── Initial state tests ───────────────────────────────────────────────────────

class TestInitialState:
    def test_all_players_present(self):
        s = initial_state()
        assert set(s.resources.keys()) == set(PLAYERS)
        assert set(s.vp.keys()) == set(PLAYERS)

    def test_start_with_zero_resources(self):
        s = initial_state()
        for p in PLAYERS:
            for r in RESOURCES:
                assert s.resources[p][r] == 0

    def test_start_with_zero_vp(self):
        s = initial_state()
        for p in PLAYERS:
            assert s.vp[p] == 0

    def test_setup_phase(self):
        s = initial_state()
        assert s.phase == "setup"
        assert not s.game_over
        assert s.winner is None


# ── Placement tests ───────────────────────────────────────────────────────────

class TestPlacement:
    def _setup_state_with_settlement(self, vid: int, player: str = "P1") -> GameState:
        s = initial_state()
        s.settlements[vid] = player
        s.vp[player] += 1
        return s

    def test_vertex_free_empty_board(self):
        s = initial_state()
        for vid in BOARD.vert_to_hexes:
            assert vertex_is_free(s, vid)

    def test_vertex_occupied_not_free(self):
        s = initial_state()
        vid = list(BOARD.vert_to_hexes.keys())[0]
        s.settlements[vid] = "P1"
        assert not vertex_is_free(s, vid)

    def test_distance_rule_blocks_neighbors(self):
        s = initial_state()
        vid = list(BOARD.vert_to_hexes.keys())[0]
        s.settlements[vid] = "P1"
        for nb in BOARD.vert_neighbors[vid]:
            assert not vertex_is_free(s, nb), f"neighbor {nb} of {vid} should be blocked"

    def test_build_settlement_free_during_setup(self):
        s = initial_state()
        vid = list(BOARD.vert_to_hexes.keys())[0]
        ok = build_settlement(s, "P1", vid, free=True)
        assert ok
        assert s.settlements[vid] == "P1"
        assert s.vp["P1"] == 1

    def test_build_settlement_costs_resources(self):
        s = initial_state()
        # Give P1 enough resources
        s.resources["P1"][WOOD] = 1
        s.resources["P1"][GRAIN] = 1
        s.resources["P1"][STONE] = 1
        # Place a road first so the settlement is reachable
        vid = list(BOARD.vert_to_hexes.keys())[0]
        eid = BOARD.vert_to_edges[vid][0]
        s.roads[eid] = "P1"
        ok = build_settlement(s, "P1", vid, free=False)
        assert ok
        assert s.resources["P1"][WOOD] == 0
        assert s.resources["P1"][GRAIN] == 0
        assert s.resources["P1"][STONE] == 0

    def test_build_city_upgrades_settlement(self):
        s = initial_state()
        vid = list(BOARD.vert_to_hexes.keys())[0]
        s.settlements[vid] = "P1"
        s.vp["P1"] = 1
        s.resources["P1"][STONE] = 2
        s.resources["P1"][GRAIN] = 2
        ok = build_city(s, "P1", vid)
        assert ok
        assert vid not in s.settlements
        assert s.cities[vid] == "P1"
        assert s.vp["P1"] == 2

    def test_build_road_requires_connection(self):
        s = initial_state()
        # No settlements → no valid roads
        from experiments.catan.state import valid_road_edges
        edges = valid_road_edges(s, "P1")
        assert len(edges) == 0


# ── Production tests ──────────────────────────────────────────────────────────

class TestProduction:
    def test_no_production_without_settlements(self):
        s = initial_state()
        grants = produce_resources(s, 6)
        for p in PLAYERS:
            assert all(v == 0 for v in grants[p].values())

    def test_settlement_receives_production(self):
        s = initial_state()
        # Place P1 on a vertex adjacent to a hex with token 6 (GRAIN hex)
        # Find the hex with token 6
        grain_hex = next(hid for hid, tok in HEX_TOKEN.items() if tok == 6)
        vid = BOARD.hex_to_verts[grain_hex][0]
        s.settlements[vid] = "P1"
        grants = produce_resources(s, 6)
        assert grants["P1"][GRAIN] >= 1

    def test_city_receives_double_production(self):
        s = initial_state()
        grain_hex = next(hid for hid, tok in HEX_TOKEN.items() if tok == 6)
        vid = BOARD.hex_to_verts[grain_hex][0]
        s.cities[vid] = "P1"
        grants = produce_resources(s, 6)
        assert grants["P1"][GRAIN] == 2

    def test_robber_blocks_production(self):
        s = initial_state()
        grain_hex = next(hid for hid, tok in HEX_TOKEN.items() if tok == 6)
        vid = BOARD.hex_to_verts[grain_hex][0]
        s.settlements[vid] = "P1"
        s.robber_hex = grain_hex
        grants = produce_resources(s, 6)
        assert grants["P1"][GRAIN] == 0

    def test_no_production_on_seven(self):
        s = initial_state()
        # 7 triggers robber, not production — no token is 7
        grants = produce_resources(s, 7)
        for p in PLAYERS:
            assert all(v == 0 for v in grants[p].values())


# ── Trade tests ───────────────────────────────────────────────────────────────

class TestTrade:
    def test_bank_trade_3_for_1(self):
        s = initial_state()
        s.resources["P1"][STONE] = 3
        ok = bank_trade(s, "P1", STONE, WOOD, turn=1)
        assert ok
        assert s.resources["P1"][STONE] == 0
        assert s.resources["P1"][WOOD] == 1

    def test_bank_trade_insufficient_resources(self):
        s = initial_state()
        s.resources["P1"][STONE] = 2
        ok = bank_trade(s, "P1", STONE, WOOD, turn=1)
        assert not ok

    def test_bilateral_trade_accepted(self):
        s = initial_state()
        s.resources["P1"][STONE] = 2
        s.resources["P2"][WOOD] = 1
        ok = bilateral_trade(s, "P1", "P2", {STONE: 1}, {WOOD: 1}, turn=1, accepted=True)
        assert ok
        assert s.resources["P1"][STONE] == 1
        assert s.resources["P1"][WOOD] == 1
        assert s.resources["P2"][STONE] == 1
        assert s.resources["P2"][WOOD] == 0

    def test_bilateral_trade_rejected(self):
        s = initial_state()
        s.resources["P1"][STONE] = 2
        s.resources["P2"][WOOD] = 1
        ok = bilateral_trade(s, "P1", "P2", {STONE: 1}, {WOOD: 1}, turn=1, accepted=False)
        assert not ok
        # Resources unchanged
        assert s.resources["P1"][STONE] == 2
        assert s.resources["P2"][WOOD] == 1


# ── Win condition tests ───────────────────────────────────────────────────────

class TestWinCondition:
    def test_no_winner_at_start(self):
        s = initial_state()
        assert check_win(s) is None

    def test_winner_at_7_vp(self):
        s = initial_state()
        s.vp["P2"] = WIN_VP
        assert check_win(s) == "P2"

    def test_no_winner_below_7(self):
        s = initial_state()
        s.vp["P1"] = WIN_VP - 1
        assert check_win(s) is None


# ── Roll and produce integration test ─────────────────────────────────────────

class TestRollAndProduce:
    def test_roll_sets_dice(self):
        s = initial_state()
        rng = random.Random(42)
        roll = roll_and_produce(s, rng)
        assert s.dice_roll == roll
        assert 2 <= roll <= 12

    def test_7_roll_no_standard_production(self):
        s = initial_state()
        # Place settlement on every hex with token 6
        grain_hex = next(hid for hid, tok in HEX_TOKEN.items() if tok == 6)
        vid = BOARD.hex_to_verts[grain_hex][0]
        s.settlements[vid] = "P1"
        # Force a 7 roll by patching
        rng = random.Random(0)
        # We can't force a 7 easily, but we can test manually
        from experiments.catan.state import apply_discard_rule, produce_resources
        # 7 → no production from tokens
        grants = produce_resources(s, 7)
        assert all(v == 0 for v in grants["P1"].values())

    def test_discard_rule_triggers_at_8_cards(self):
        s = initial_state()
        s.resources["P1"][STONE] = 8
        rng = random.Random(42)
        from experiments.catan.state import apply_discard_rule
        apply_discard_rule(s, rng)
        total = sum(s.resources["P1"].values())
        assert total == 4  # 8 // 2 = 4 discarded → 4 remaining
