"""Tests for adversarial condition and pilot runner."""

import random

import pytest

from experiments.catan.adversarial import make_adversarial_runner, _counter_alliance_active
from experiments.catan.conditions import condition_b_runner, run_turn_condition_c
from experiments.catan.game import run_game, run_turn_greedy
from experiments.catan.personas import ALLIANCE_PLAYERS, DEFAULT_PERSONAS
from experiments.catan.run_pilot import CONDITIONS, run_pilot
from experiments.catan.state import PLAYERS, WIN_VP, initial_state


class TestAdversarialCondition:
    def test_counter_alliance_inactive_before_turn_3(self):
        s = initial_state()
        s.turn = 8  # < 3 rounds × 4 players = 12
        s.vp = {"P1": 5, "P2": 5, "P3": 2, "P4": 2}
        assert not _counter_alliance_active(s)

    def test_counter_alliance_activates_when_alliance_leads(self):
        s = initial_state()
        s.turn = 20
        s.vp = {"P1": 4, "P2": 4, "P3": 2, "P4": 2}
        assert _counter_alliance_active(s)

    def test_counter_alliance_inactive_when_losing(self):
        s = initial_state()
        s.turn = 20
        s.vp = {"P1": 2, "P2": 2, "P3": 4, "P4": 4}
        assert not _counter_alliance_active(s)

    def test_adversarial_game_terminates(self):
        runner = make_adversarial_runner(run_turn_greedy)
        state = run_game(DEFAULT_PERSONAS, random.Random(7), turn_runner=runner)
        assert state.game_over
        assert state.winner in PLAYERS

    def test_adversarial_game_winner_has_7_vp(self):
        runner = make_adversarial_runner(run_turn_greedy)
        state = run_game(DEFAULT_PERSONAS, random.Random(7), turn_runner=runner)
        assert state.vp[state.winner] >= WIN_VP

    def test_adversarial_with_condition_c(self):
        runner = make_adversarial_runner(run_turn_condition_c)
        state = run_game(DEFAULT_PERSONAS, random.Random(42), turn_runner=runner)
        assert state.game_over
        assert state.vp[state.winner] >= WIN_VP


class TestPilotRunner:
    def test_pilot_returns_correct_row_count(self):
        rows = run_pilot(n_games=2)
        assert len(rows) == len(CONDITIONS) * 2  # 4 conditions × 2 games

    def test_pilot_all_games_terminate_at_7vp(self):
        rows = run_pilot(n_games=3)
        for r in rows:
            assert r["vp_" + r["winner"].lower()] >= WIN_VP, \
                f"condition={r['condition']} seed={r['seed']} winner={r['winner']} vp not 7+"

    def test_pilot_csv_written(self):
        from experiments.catan.run_pilot import PILOT_CSV
        run_pilot(n_games=2)
        assert PILOT_CSV.exists()

    def test_pilot_csv_has_required_columns(self):
        import csv
        from experiments.catan.run_pilot import PILOT_CSV
        run_pilot(n_games=1)
        with open(PILOT_CSV) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        for col in ["condition", "winner", "alliance_win", "turns", "vp_p1", "vp_p2"]:
            assert col in cols

    def test_pilot_all_conditions_represented(self):
        rows = run_pilot(n_games=1)
        conds = {r["condition"] for r in rows}
        assert conds == set(CONDITIONS.keys())

    def test_pilot_max_turns_within_gate(self):
        rows = run_pilot(n_games=5)
        for r in rows:
            assert r["turns"] <= 120, f"condition={r['condition']} exceeded MAX_TURNS"
