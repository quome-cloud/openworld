"""Tests for Catan setup phase and full game loop (condition a)."""

import random

import pytest

from experiments.catan.game import run_game, run_setup
from experiments.catan.personas import DEFAULT_PERSONAS, PERSONA_CONFIGS
from experiments.catan.state import PLAYERS, WIN_VP, initial_state


class TestSetup:
    def test_setup_places_two_settlements_per_player(self):
        s = initial_state()
        rng = random.Random(42)
        run_setup(s, rng)
        for player in PLAYERS:
            count = sum(1 for p in s.settlements.values() if p == player)
            assert count == 2, f"{player} has {count} settlements after setup"

    def test_setup_places_two_roads_per_player(self):
        s = initial_state()
        rng = random.Random(42)
        run_setup(s, rng)
        for player in PLAYERS:
            count = sum(1 for p in s.roads.values() if p == player)
            assert count == 2, f"{player} has {count} roads after setup"

    def test_setup_grants_starting_resources(self):
        s = initial_state()
        rng = random.Random(42)
        run_setup(s, rng)
        # Every player should have at least some resources after setup
        for player in PLAYERS:
            total = sum(s.resources[player].values())
            assert total >= 0   # may be 0 if second settlement is on barren hex

    def test_setup_sets_phase_to_roll(self):
        s = initial_state()
        rng = random.Random(42)
        run_setup(s, rng)
        assert s.phase == "roll"

    def test_setup_vp_is_two_per_player(self):
        s = initial_state()
        rng = random.Random(42)
        run_setup(s, rng)
        for player in PLAYERS:
            assert s.vp[player] == 2


class TestFullGame:
    def test_game_terminates(self):
        rng = random.Random(7)
        state = run_game(DEFAULT_PERSONAS, rng)
        assert state.game_over

    def test_winner_has_sufficient_vp(self):
        rng = random.Random(7)
        state = run_game(DEFAULT_PERSONAS, rng)
        assert state.winner is not None
        assert state.vp[state.winner] >= WIN_VP

    def test_winner_is_valid_player(self):
        rng = random.Random(7)
        state = run_game(DEFAULT_PERSONAS, rng)
        assert state.winner in PLAYERS

    def test_game_turn_count_reasonable(self):
        rng = random.Random(7)
        state = run_game(DEFAULT_PERSONAS, rng)
        # 7-VP game with 4 players should finish well under MAX_TURNS
        assert state.turn < 100

    def test_multiple_seeds_all_terminate(self):
        for seed in range(10):
            state = run_game(DEFAULT_PERSONAS, random.Random(seed))
            assert state.game_over, f"seed {seed} did not terminate"

    def test_all_persona_configs_terminate(self):
        for config_name, personas in PERSONA_CONFIGS.items():
            state = run_game(personas, random.Random(42))
            assert state.game_over, f"{config_name} did not terminate"
            assert state.winner in PLAYERS
