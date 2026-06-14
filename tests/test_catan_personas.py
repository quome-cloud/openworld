"""Tests for Catan persona configurations."""

from experiments.catan.personas import (
    ALLIANCE_PLAYERS,
    DEFAULT_PERSONAS,
    INDEPENDENT_PLAYERS,
    PERSONA_CONFIGS,
    SYM_AGGRESSIVE_PERSONAS,
    SYM_CONSERVATIVE_PERSONAS,
    Persona,
)


class TestPersonas:
    def test_all_configs_have_four_players(self):
        for name, config in PERSONA_CONFIGS.items():
            assert set(config.keys()) == {"P1", "P2", "P3", "P4"}, name

    def test_persona_fields_in_range(self):
        for config in PERSONA_CONFIGS.values():
            for p in config.values():
                assert 0.0 <= p.risk_tolerance <= 1.0
                assert 0.0 <= p.expansion_preference <= 1.0
                assert 0.0 <= p.trade_openness_adversary <= 1.0

    def test_default_alliance_asymmetry(self):
        # P1 is settler-biased, P2 is city-biased — intentional design
        assert DEFAULT_PERSONAS["P1"].expansion_preference > DEFAULT_PERSONAS["P2"].expansion_preference

    def test_sym_aggressive_equal_risk(self):
        p = SYM_AGGRESSIVE_PERSONAS
        assert p["P1"].risk_tolerance == p["P2"].risk_tolerance == 0.7

    def test_sym_conservative_lower_risk_than_default_p2(self):
        assert SYM_CONSERVATIVE_PERSONAS["P1"].risk_tolerance < DEFAULT_PERSONAS["P2"].risk_tolerance

    def test_alliance_and_independent_cover_all_players(self):
        all_players = set(ALLIANCE_PLAYERS) | set(INDEPENDENT_PLAYERS)
        assert all_players == {"P1", "P2", "P3", "P4"}

    def test_persona_is_frozen(self):
        p = DEFAULT_PERSONAS["P1"]
        import dataclasses
        assert dataclasses.is_dataclass(p)
        try:
            p.risk_tolerance = 0.9  # type: ignore
            assert False, "should have raised"
        except (AttributeError, dataclasses.FrozenInstanceError):
            pass

    def test_three_configs_exist(self):
        assert set(PERSONA_CONFIGS.keys()) == {"default", "sym_aggressive", "sym_conservative"}
