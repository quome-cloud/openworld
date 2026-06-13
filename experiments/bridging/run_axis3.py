"""Axis 3: Spillover magnitude sweep.

Sweeps all spillover bonuses simultaneously across four uniform magnitude levels
while holding distribution (P2 baseline), slate size (K=7 archetype-seeded),
and persona population (N=100) constant.

Hypotheses tested:
  H3a: gap(C) − gap(A) increases monotonically with spillover magnitude
  H3b: at S1 (0.01), 95% bootstrap CI for C−A crosses zero (noise floor)
  H3c: majority vote catches up at S4 via accidental centrist alignment
  H3d: C_CN ≈ C_PP persists across all magnitude levels

Run from the openworld repo root:
    python -m experiments.bridging.run_axis3

Outputs:
    experiments/bridging/results/axis3_results.csv
    experiments/bridging/results/axis3_figure.svg
"""

from __future__ import annotations

import csv
import math
import pathlib
import random
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .conditions_c import condition_c_community_notes, condition_c_polarity_product
from .personas import ISSUES
from .policy import (
    SPILLOVER_CONFIGS,
    STANCES,
    PayoffTable,
    PolicyBundle,
    _config_hash,
    enumerate_bundles,
)
from .run_axis1 import generate_personas_custom
from .run_axis2 import _ARCHETYPES, _CHUNK, build_oracle_fast, generate_slate
from .simulation import _bundle_index, condition_a, condition_z

# ── Config ────────────────────────────────────────────────────────────────────

N_PERSONAS = 100
PERSONA_SEED = 42
K = 7
SLATE_TYPE = "archetype"
N_TRIALS = 50
BASE_SEED = 2000   # distinct from Axis 1 (1000) and Axis 2 (0)
CONDITIONS = ("Z", "A", "C_CN", "C_PP", "D")

# P2 baseline distribution
P2_COMPONENTS = [(0.40, -0.45, 0.25), (0.40, 0.45, 0.25), (0.20, 0.00, 0.15)]

# Magnitude levels: uniform bonus applied to all spillover patterns in a config
MAGNITUDES = [
    ("S1", 0.01),
    ("S2", 0.05),
    ("S3", 0.10),
    ("S4", 0.20),
]

# Base spillover patterns (issue conditions only, no amounts)
_CENTRIST_PATTERNS: List[Dict[str, int]] = [
    {"healthcare": 0, "fiscal": 0},
    {"climate": -1, "foreign_policy": 0},
    {"criminal_justice": -1, "education": 0},
]
_OFF_AXIS_PATTERNS: List[Dict[str, int]] = [
    {"climate": -2, "fiscal": 1},
    {"civil_rights": 1, "criminal_justice": -1},
    {"education": 2, "healthcare": -1},
]

BASE_PATTERNS = {
    "centrist": _CENTRIST_PATTERNS,
    "off_axis": _OFF_AXIS_PATTERNS,
}

RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def _make_spillover_config_name(base_cfg: str, mag_id: str) -> str:
    return f"{base_cfg}_{mag_id}"


def _register_magnitude_configs(magnitude: float) -> None:
    """Inject scaled spillover configs into SPILLOVER_CONFIGS for this magnitude."""
    for base_cfg, patterns in BASE_PATTERNS.items():
        for mag_id, mag_val in MAGNITUDES:
            if mag_val != magnitude:
                continue
            name = _make_spillover_config_name(base_cfg, mag_id)
            SPILLOVER_CONFIGS[name] = [(pattern, magnitude) for pattern in patterns]


def _register_all_magnitude_configs() -> None:
    for base_cfg, patterns in BASE_PATTERNS.items():
        for mag_id, mag_val in MAGNITUDES:
            name = _make_spillover_config_name(base_cfg, mag_id)
            SPILLOVER_CONFIGS[name] = [(pattern, mag_val) for pattern in patterns]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class Axis3Result:
    mag_id: str
    magnitude: float
    base_cfg: str
    condition: str
    trial: int
    gap_fraction: float
    G_achieved: float
    G_random: float
    G_oracle: float


def run_trial_axis3(
    trial_idx: int,
    personas,
    table: PayoffTable,
    mag_id: str,
    magnitude: float,
    base_cfg: str,
) -> List[Axis3Result]:
    trial_seed = BASE_SEED + trial_idx
    rng = random.Random(trial_seed)
    slate = generate_slate(K, SLATE_TYPE, trial_seed)

    g_random = table.g_random
    g_oracle = table.g_oracle
    results = []

    for cond in CONDITIONS:
        if cond == "Z":
            winner = condition_z(slate, rng)
        elif cond == "A":
            winner = condition_a(slate, personas)
        elif cond == "C_CN":
            winner = condition_c_community_notes(slate, personas)
        elif cond == "C_PP":
            winner = condition_c_polarity_product(slate, personas)
        elif cond == "D":
            results.append(Axis3Result(
                mag_id=mag_id, magnitude=magnitude, base_cfg=base_cfg,
                condition="D", trial=trial_idx,
                gap_fraction=1.0, G_achieved=g_oracle,
                G_random=g_random, G_oracle=g_oracle,
            ))
            continue
        else:
            raise ValueError(cond)

        g_achieved = table.g_values[_bundle_index(winner)]
        gap = table.gap_fraction(g_achieved)
        results.append(Axis3Result(
            mag_id=mag_id, magnitude=magnitude, base_cfg=base_cfg,
            condition=cond, trial=trial_idx,
            gap_fraction=gap, G_achieved=g_achieved,
            G_random=g_random, G_oracle=g_oracle,
        ))
    return results


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def _bootstrap_ci(
    values: List[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 99,
) -> Tuple[float, float]:
    """95% bootstrap CI on the median difference."""
    rng = random.Random(seed)
    n = len(values)
    medians = []
    for _ in range(n_boot):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot) - 1
    return medians[lo_idx], medians[hi_idx]


# ── Figure ────────────────────────────────────────────────────────────────────

def _svg_magnitude_figure(
    results: List[Axis3Result],
    out_path: pathlib.Path,
) -> None:
    """Two-panel SVG: centrist (left) and off_axis (right).
    X-axis: magnitude level; Y-axis: median gap_fraction.
    Lines per condition.
    """
    W, H = 900, 460
    PAD_L, PAD_R, PAD_T, PAD_B = 65, 20, 50, 70
    PANEL_W = (W - PAD_L - PAD_R - 40) // 2
    PANEL_H = H - PAD_T - PAD_B

    COLORS = {"Z": "#999", "A": "#4C72B0", "C_CN": "#DD8452", "C_PP": "#55A868", "D": "#C44E52"}
    MAG_VALS = [m[1] for m in MAGNITUDES]
    MAG_IDS = [m[0] for m in MAGNITUDES]
    X_POSITIONS = {mv: i / (len(MAG_VALS) - 1) for i, mv in enumerate(MAG_VALS)}
    BASE_CFGS = ("centrist", "off_axis")

    def px(mag_val: float, panel_left: float) -> float:
        return panel_left + X_POSITIONS[mag_val] * PANEL_W

    def py(gap: float) -> float:
        return PAD_T + PANEL_H * (1.0 - max(0.0, min(1.0, gap)))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="monospace" font-size="11">',
        f'<rect width="{W}" height="{H}" fill="#f9f9f9"/>',
        f'<text x="{W//2}" y="18" text-anchor="middle" font-size="13" '
        f'font-weight="bold" fill="#222">Axis 3: Gap Fraction vs Spillover Magnitude '
        f'(N={N_PERSONAS}, {N_TRIALS} trials/cell)</text>',
    ]

    for panel_idx, base_cfg in enumerate(BASE_CFGS):
        pl = PAD_L + panel_idx * (PANEL_W + 40)
        label = f"{base_cfg} spillover pattern"

        lines.append(
            f'<rect x="{pl}" y="{PAD_T}" width="{PANEL_W}" height="{PANEL_H}" '
            f'fill="white" stroke="#ccc" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{pl + PANEL_W//2}" y="{PAD_T - 6}" text-anchor="middle" '
            f'fill="#333" font-size="12" font-weight="bold">{label}</text>'
        )

        for ytick in [0.0, 0.25, 0.50, 0.75, 1.0]:
            y = py(ytick)
            lines.append(
                f'<line x1="{pl}" y1="{y:.1f}" x2="{pl+PANEL_W}" y2="{y:.1f}" '
                f'stroke="#ddd" stroke-dasharray="4,3"/>'
            )
            if panel_idx == 0:
                lines.append(
                    f'<text x="{pl-4}" y="{y+4:.1f}" text-anchor="end" fill="#555">'
                    f'{ytick:.2f}</text>'
                )

        for mv, mid in zip(MAG_VALS, MAG_IDS):
            x = px(mv, pl)
            lines.append(
                f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{PAD_T+PANEL_H}" '
                f'stroke="#eee"/>'
            )
            lines.append(
                f'<text x="{x:.1f}" y="{PAD_T+PANEL_H+16}" text-anchor="middle" fill="#555">'
                f'{mid} ({mv})</text>'
            )

        for cond in CONDITIONS:
            color = COLORS[cond]
            pts = []
            for mv in MAG_VALS:
                vals = [r.gap_fraction for r in results
                        if r.magnitude == mv and r.base_cfg == base_cfg and r.condition == cond]
                if vals:
                    med = statistics.median(vals)
                    pts.append((px(mv, pl), py(med)))
            if len(pts) >= 2:
                path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                lines.append(
                    f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2"/>'
                )
            for x, y in pts:
                lines.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" fill-opacity="0.85"/>'
                )

    # Legend
    lx, ly = PAD_L, H - 30
    for cond in CONDITIONS:
        lines.append(
            f'<line x1="{lx}" y1="{ly+5}" x2="{lx+18}" y2="{ly+5}" '
            f'stroke="{COLORS[cond]}" stroke-width="2.5"/>'
        )
        lines.append(f'<text x="{lx+22}" y="{ly+9}" fill="#333">{cond}</text>')
        lx += 65

    lines.append("</svg>")
    out_path.write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Register all magnitude configs into SPILLOVER_CONFIGS
    _register_all_magnitude_configs()

    print(f"Axis 3: Spillover magnitude sweep — N={N_PERSONAS}, K={K} archetype-seeded, "
          f"P2 baseline, {N_TRIALS} trials/cell")

    print("Generating P2 personas...")
    personas = generate_personas_custom(P2_COMPONENTS, n=N_PERSONAS, seed=PERSONA_SEED)
    mean_ideo = statistics.mean(p.latent_ideology for p in personas)
    std_ideo = statistics.stdev(p.latent_ideology for p in personas)
    print(f"  Ideology: mean={mean_ideo:+.3f}  std={std_ideo:.3f}")

    print("Enumerating 5^8 bundles (390,625)...")
    bundles = enumerate_bundles()

    all_results: List[Axis3Result] = []

    for base_cfg, patterns in BASE_PATTERNS.items():
        print(f"\n{'='*60}")
        print(f"Base config: {base_cfg}")
        for mag_id, magnitude in MAGNITUDES:
            cfg_name = _make_spillover_config_name(base_cfg, mag_id)
            print(f"\n  {mag_id} (magnitude={magnitude}) → config key '{cfg_name}'")

            t0 = time.time()
            table = build_oracle_fast(personas, cfg_name, bundles, cache=True)
            elapsed = time.time() - t0
            print(f"    oracle {elapsed:.1f}s  G_oracle={table.g_oracle:.4f}  "
                  f"G_random={table.g_random:.4f}  "
                  f"delta_oracle={table.g_oracle - table.g_random:.4f}")

            cell_results = []
            for trial_idx in range(N_TRIALS):
                res = run_trial_axis3(
                    trial_idx, personas, table, mag_id, magnitude, base_cfg,
                )
                cell_results.extend(res)
            all_results.extend(cell_results)

            for cond in ("A", "C_CN", "C_PP"):
                vals = [r.gap_fraction for r in cell_results if r.condition == cond]
                med = statistics.median(vals)
                ci_lo, ci_hi = _bootstrap_ci(vals)
                print(f"    {cond}: med={med:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    # Summary tables
    print("\n── C_CN vs A gap_delta by magnitude ────────────────────────────────")
    print(f"{'Mag':>4}  {'base_cfg':<12} {'C_CN':>6} {'A':>6} {'delta':>7}  "
          f"{'CI_lo':>6} {'CI_hi':>6}  H3b?")
    print("-" * 70)
    for base_cfg in ("centrist", "off_axis"):
        for mag_id, magnitude in MAGNITUDES:
            c_vals = [r.gap_fraction for r in all_results
                      if r.mag_id == mag_id and r.base_cfg == base_cfg and r.condition == "C_CN"]
            a_vals = [r.gap_fraction for r in all_results
                      if r.mag_id == mag_id and r.base_cfg == base_cfg and r.condition == "A"]
            diff_vals = [c - a for c, a in zip(
                sorted(c_vals), sorted(a_vals)
            )]
            c_med = statistics.median(c_vals)
            a_med = statistics.median(a_vals)
            delta = c_med - a_med
            ci_lo, ci_hi = _bootstrap_ci(diff_vals)
            h3b = "noise-floor" if ci_lo < 0.0 < ci_hi else ("sig" if ci_lo > 0.0 else "neg")
            print(f"{mag_id:>4}  {base_cfg:<12} {c_med:>6.3f} {a_med:>6.3f} {delta:>+7.3f}  "
                  f"{ci_lo:>6.3f} {ci_hi:>6.3f}  {h3b}")

    # H3d: C_CN vs C_PP consistency
    print("\n── H3d: C_CN vs C_PP consistency ───────────────────────────────────")
    print(f"{'Mag':>4}  {'base_cfg':<12} {'C_CN':>6} {'C_PP':>6} {'delta':>7}")
    print("-" * 50)
    for base_cfg in ("centrist", "off_axis"):
        for mag_id, magnitude in MAGNITUDES:
            cn_med = statistics.median([r.gap_fraction for r in all_results
                if r.mag_id == mag_id and r.base_cfg == base_cfg and r.condition == "C_CN"])
            pp_med = statistics.median([r.gap_fraction for r in all_results
                if r.mag_id == mag_id and r.base_cfg == base_cfg and r.condition == "C_PP"])
            print(f"{mag_id:>4}  {base_cfg:<12} {cn_med:>6.3f} {pp_med:>6.3f} {cn_med-pp_med:>+7.3f}")

    # Write CSV
    csv_path = RESULTS_DIR / "axis3_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "mag_id", "magnitude", "base_cfg", "condition", "trial",
            "gap_fraction", "G_achieved", "G_random", "G_oracle",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "mag_id": r.mag_id, "magnitude": f"{r.magnitude:.2f}",
                "base_cfg": r.base_cfg, "condition": r.condition,
                "trial": r.trial, "gap_fraction": f"{r.gap_fraction:.6f}",
                "G_achieved": f"{r.G_achieved:.6f}", "G_random": f"{r.G_random:.6f}",
                "G_oracle": f"{r.G_oracle:.6f}",
            })
    print(f"\nWrote {len(all_results)} rows → {csv_path}")

    svg_path = RESULTS_DIR / "axis3_figure.svg"
    _svg_magnitude_figure(all_results, svg_path)
    print(f"Wrote figure → {svg_path}")
    print("\nAxis 3 complete.")


if __name__ == "__main__":
    main()
