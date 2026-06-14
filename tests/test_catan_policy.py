"""Tests for Catan greedy policy."""

import pytest

from experiments.catan.board import BOARD, HEX_TOKEN, GRAIN, STONE, WOOD
from experiments.catan.personas import DEFAULT_PERSONAS
from experiments.catan.policy import (
    available_build_actions,
    choose_robber_hex,
    greedy_bank_trade,
    greedy_build_phase,
    hex_production_value,
    vertex_production_value,
)
from experiments.catan.state import initial_state, RESOURCES


class TestProductionValues:
    def test_high_prob_token_higher_value(self):
        # Token 6 (prob 5/36) should beat token 3 (prob 2/36)
        h6 = next(hid for hid, tok in HEX_TOKEN.items() if tok == 6)
        h3 = next(hid for hid, tok in HEX_TOKEN.items() if tok == 3)
        assert hex_production_value(h6) > hex_production_value(h3)

    def test_vertex_value_is_sum_of_hex_values(self):
        # Pick any vertex; its value >= each adjacent hex's value
        vid = list(BOARD.vert_to_hexes.keys())[0]
        total = vertex_production_value(vid)
        for hid in BOARD.vert_to_hexes[vid]:
            assert total >= hex_production_value(hid)


class TestAvailableActions:
    def test_no_actions_with_no_resources(self):
        s = initial_state()
        # Place a settlement so roads/cities are theoretically possible
        vid = list(BOARD.vert_to_hexes.keys())[0]
        s.settlements[vid] = "P1"
        s.vp["P1"] = 1
        actions = available_build_actions(s, "P1", DEFAULT_PERSONAS["P1"])
        assert len(actions) == 0

    def test_can_build_road_with_resources(self):
        s = initial_state()
        vid = list(BOARD.vert_to_hexes.keys())[0]
        s.settlements[vid] = "P1"
        s.vp["P1"] = 1
        s.resources["P1"][WOOD] = 1
        s.resources["P1"][STONE] = 1
        actions = available_build_actions(s, "P1", DEFAULT_PERSONAS["P1"])
        road_actions = [a for a in actions if a.action_type == "road"]
        assert len(road_actions) > 0

    def test_city_scored_above_road(self):
        s = initial_state()
        vid = list(BOARD.vert_to_hexes.keys())[0]
        s.settlements[vid] = "P1"
        s.vp["P1"] = 1
        # Give enough for both city and road
        s.resources["P1"][STONE] = 4
        s.resources["P1"][GRAIN] = 4
        s.resources["P1"][WOOD] = 2
        actions = available_build_actions(s, "P1", DEFAULT_PERSONAS["P1"])
        if any(a.action_type == "city" for a in actions):
            city_score = max(a.score for a in actions if a.action_type == "city")
            road_scores = [a.score for a in actions if a.action_type == "road"]
            if road_scores:
                assert city_score >= max(road_scores)


class TestRobberPlacement:
    def test_robber_targets_leading_opponent(self):
        s = initial_state()
        # P3 is leading
        s.vp["P3"] = 3
        s.vp["P1"] = 1
        # Give P3 a settlement somewhere
        grain_hex = next(hid for hid, tok in HEX_TOKEN.items() if tok == 6)
        vid = BOARD.hex_to_verts[grain_hex][0]
        s.settlements[vid] = "P3"
        target = choose_robber_hex(s, "P1")
        # Should target P3's hex
        assert target == grain_hex

    def test_robber_returns_valid_hex(self):
        s = initial_state()
        target = choose_robber_hex(s, "P1")
        assert target in BOARD.hex_coords


class TestGreedyBuild:
    def test_greedy_build_uses_resources(self):
        s = initial_state()
        vid = list(BOARD.vert_to_hexes.keys())[0]
        s.settlements[vid] = "P1"
        s.vp["P1"] = 1
        s.resources["P1"][WOOD] = 1
        s.resources["P1"][STONE] = 1
        built = greedy_build_phase(s, "P1", DEFAULT_PERSONAS["P1"])
        assert built >= 1
        assert s.resources["P1"][WOOD] == 0
        assert s.resources["P1"][STONE] == 0

    def test_greedy_build_zero_with_no_resources(self):
        s = initial_state()
        built = greedy_build_phase(s, "P1", DEFAULT_PERSONAS["P1"])
        assert built == 0


class TestBankTrade:
    def test_bank_trade_enables_city(self):
        s = initial_state()
        vid = list(BOARD.vert_to_hexes.keys())[0]
        s.settlements[vid] = "P1"
        s.vp["P1"] = 1
        # Have 3 Wood (surplus) + 1 Stone + 2 Grain; need 2 Stone + 2 Grain for city
        s.resources["P1"][WOOD] = 3
        s.resources["P1"][STONE] = 1
        s.resources["P1"][GRAIN] = 2
        traded = greedy_bank_trade(s, "P1", DEFAULT_PERSONAS["P1"])
        assert traded
        # One resource was traded away
        assert s.resources["P1"][WOOD] == 0
        assert s.resources["P1"][STONE] == 2  # received 1 Stone

    def test_no_bank_trade_if_nothing_needed(self):
        s = initial_state()
        # No settlements → no builds wanted → no trade
        traded = greedy_bank_trade(s, "P1", DEFAULT_PERSONAS["P1"])
        assert not traded
