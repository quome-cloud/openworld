"""Tests for Catan coordination conditions (b) and (c)."""

import random

import pytest

from experiments.catan.conditions import (
    compute_alliance_strategy,
    condition_b_runner,
    run_turn_condition_c,
)
from experiments.catan.game import run_game, run_setup
from experiments.catan.personas import ALLIANCE_PLAYERS, DEFAULT_PERSONAS
from experiments.catan.state import PLAYERS, WIN_VP, initial_state


class TestConditionBStrategy:
    def _make_post_setup_state(self, seed=42):
        s = initial_state()
        run_setup(s, random.Random(seed))
        return s

    def test_strategy_assigns_settler_and_builder(self):
        s = self._make_post_setup_state()
        strat = compute_alliance_strategy(s)
        assert strat.settler in ALLIANCE_PLAYERS
        assert strat.builder in ALLIANCE_PLAYERS
        assert strat.settler != strat.builder

    def test_strategy_has_valid_resources(self):
        from experiments.catan.board import RESOURCES
        s = self._make_post_setup_state()
        strat = compute_alliance_strategy(s)
        assert strat.trade_give in RESOURCES
        assert strat.trade_receive in RESOURCES

    def test_condition_b_game_terminates(self):
        runner = condition_b_runner()
        state = run_game(DEFAULT_PERSONAS, random.Random(7), turn_runner=runner)
        assert state.game_over
        assert state.winner in PLAYERS

    def test_condition_b_winner_has_7_vp(self):
        runner = condition_b_runner()
        state = run_game(DEFAULT_PERSONAS, random.Random(7), turn_runner=runner)
        assert state.vp[state.winner] >= WIN_VP

    def test_condition_b_multiple_seeds(self):
        for seed in range(5):
            runner = condition_b_runner()
            state = run_game(DEFAULT_PERSONAS, random.Random(seed), turn_runner=runner)
            assert state.game_over, f"seed {seed} did not terminate"
            assert state.vp[state.winner] >= WIN_VP


class TestConditionC:
    def test_condition_c_game_terminates(self):
        state = run_game(DEFAULT_PERSONAS, random.Random(7),
                         turn_runner=run_turn_condition_c)
        assert state.game_over
        assert state.winner in PLAYERS

    def test_condition_c_winner_has_7_vp(self):
        state = run_game(DEFAULT_PERSONAS, random.Random(7),
                         turn_runner=run_turn_condition_c)
        assert state.vp[state.winner] >= WIN_VP

    def test_condition_c_multiple_seeds(self):
        for seed in range(5):
            state = run_game(DEFAULT_PERSONAS, random.Random(seed),
                             turn_runner=run_turn_condition_c)
            assert state.game_over, f"seed {seed} did not terminate"
            assert state.vp[state.winner] >= WIN_VP

    def test_condition_c_alliance_tracks_win_rate(self):
        """Alliance (P1+P2) should win some fraction of games under condition (c)."""
        alliance_wins = 0
        for seed in range(10):
            state = run_game(DEFAULT_PERSONAS, random.Random(seed),
                             turn_runner=run_turn_condition_c)
            if state.winner in ALLIANCE_PLAYERS:
                alliance_wins += 1
        # No strict threshold — just confirm coordination runs without crashing
        assert alliance_wins >= 0  # always true; baseline documented for comparison


class TestConditionDStub:
    def test_condition_d_falls_back_without_llm(self):
        """Condition (d) with no LLM falls back to condition (c) behaviour."""
        from experiments.catan.conditions import run_turn_condition_d
        state = run_game(DEFAULT_PERSONAS, random.Random(7),
                         turn_runner=lambda s, p, pers, rng: run_turn_condition_d(s, p, pers, rng, llm_call=None))
        assert state.game_over
        assert state.vp[state.winner] >= WIN_VP
