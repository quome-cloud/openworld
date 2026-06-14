"""Persona definitions for the simplified Catan alliance experiment.

Per design doc §2: P1+P2 are the alliance; P3+P4 play independently (or as
counter-alliance in the adversarial condition).  Three persona configurations
for the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Persona:
    player_id: str
    risk_tolerance: float          # [0,1] — willingness to take negative-EV trade for position
    expansion_preference: float    # [0,1] — 1.0=pure settler, 0.0=pure city-upgrader
    trade_openness_adversary: float  # [0,1] — willingness to trade with adversarial players


# ── Default configuration (asymmetric alliance roles) ─────────────────────────

DEFAULT_PERSONAS: Dict[str, Persona] = {
    "P1": Persona("P1", risk_tolerance=0.4, expansion_preference=0.7, trade_openness_adversary=0.3),
    "P2": Persona("P2", risk_tolerance=0.6, expansion_preference=0.3, trade_openness_adversary=0.5),
    "P3": Persona("P3", risk_tolerance=0.5, expansion_preference=0.5, trade_openness_adversary=0.5),
    "P4": Persona("P4", risk_tolerance=0.5, expansion_preference=0.5, trade_openness_adversary=0.5),
}

# ── Sym-aggressive configuration ──────────────────────────────────────────────

SYM_AGGRESSIVE_PERSONAS: Dict[str, Persona] = {
    "P1": Persona("P1", risk_tolerance=0.7, expansion_preference=0.5, trade_openness_adversary=0.5),
    "P2": Persona("P2", risk_tolerance=0.7, expansion_preference=0.5, trade_openness_adversary=0.5),
    "P3": Persona("P3", risk_tolerance=0.5, expansion_preference=0.5, trade_openness_adversary=0.5),
    "P4": Persona("P4", risk_tolerance=0.5, expansion_preference=0.5, trade_openness_adversary=0.5),
}

# ── Sym-conservative configuration ────────────────────────────────────────────

SYM_CONSERVATIVE_PERSONAS: Dict[str, Persona] = {
    "P1": Persona("P1", risk_tolerance=0.3, expansion_preference=0.5, trade_openness_adversary=0.3),
    "P2": Persona("P2", risk_tolerance=0.3, expansion_preference=0.5, trade_openness_adversary=0.3),
    "P3": Persona("P3", risk_tolerance=0.5, expansion_preference=0.5, trade_openness_adversary=0.5),
    "P4": Persona("P4", risk_tolerance=0.5, expansion_preference=0.5, trade_openness_adversary=0.5),
}

PERSONA_CONFIGS: Dict[str, Dict[str, Persona]] = {
    "default": DEFAULT_PERSONAS,
    "sym_aggressive": SYM_AGGRESSIVE_PERSONAS,
    "sym_conservative": SYM_CONSERVATIVE_PERSONAS,
}

ALLIANCE_PLAYERS = ("P1", "P2")
INDEPENDENT_PLAYERS = ("P3", "P4")
