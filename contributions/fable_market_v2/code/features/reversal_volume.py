#!/usr/bin/env python3
"""FEATURE FAMILY: REV — Short-term reversal conditioned on volume.

RESEARCH ONLY — paper trading / analysis. No live trading.

Economic rationale (causal story, written before fitting):
  Short-horizon price moves mix (a) information and (b) uninformed flow
  absorbed by liquidity providers. Flow-driven moves revert as inventory is
  laid off; information-driven moves do not (Nagel 2012, "Evaporating
  Liquidity"; Campbell-Grossman-Wang 1993). Volume separates the two: a move
  on unusually LOW volume is more likely price pressure in a thin market and
  should revert; a HIGH-volume move is more likely news being impounded.
  So the reversal edge should concentrate in low-volume movers — the volume
  interaction is the world-model claim, plain 5d reversal is the base rate.

Features:
  rev_5d(t)     = -(r5_i(t) - mean_j r5_j(t)),  r5 = 5-day return through t.
                  Units: negative relative fractional return over 5 days.
  rev_lowvol(t) = rev_5d(t) * 1[ volratio_i(t) < 1 ]
                  volratio = mean(volume, last 5d) / median(volume, trailing
                  63d ending t-5). Dimensionless indicator interaction.
  Windows (5d move, 63d volume base) fixed a priori.

Data dependencies: panel.ret, panel.volume.
Anti-lookahead: all windows trail t; the volume base window ends at t-5 so
  the event's own volume does not contaminate its baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAMILY = "REV"
FEATURES = ["rev_5d", "rev_lowvol"]

MOVE_WINDOW = 5
VOL_BASE_WINDOW = 63


def compute(panel) -> dict[str, pd.DataFrame]:
    ret, vol = panel.ret, panel.volume
    r5 = (1.0 + ret).rolling(MOVE_WINDOW).apply(np.prod, raw=True) - 1.0
    rev = -(r5.sub(r5.mean(axis=1), axis=0))

    vol_recent = vol.rolling(MOVE_WINDOW).mean()
    vol_base = vol.rolling(VOL_BASE_WINDOW).median().shift(MOVE_WINDOW)
    volratio = vol_recent / vol_base
    lowvol = (volratio < 1.0).astype(float).where(volratio.notna())

    mask = panel.close.notna()
    return {
        "rev_5d": rev.where(mask),
        "rev_lowvol": (rev * lowvol).where(mask),
    }
