"""Axis 1: Population distribution sweep.

Tests whether the bridging advantage (C_CN, C_PP vs. A) is a function of
population faction structure rather than a universal property.

Five distributions × 2 spillover configs × 100 trials/cell (N=100 personas).
Plus a polarization sub-sweep: 5 σ_between levels × 30 trials (centrist only).

Hypotheses tested:
  H1a: unimodal (P1) → gap(C) − gap(A) ≈ 0 (no faction structure to exploit)
  H1b: asymmetric bimodal (P3) → bridging advantage > P2 symmetric
  H1c: trimodal (P4) → 1D factorization misspecifies; advantage smaller than P2
  H1d: polarization curve is non-monotone (peak near P2, degrades at P5)

Run from the openworld repo root:
    python -m experiments.bridging.run_axis1

Outputs:
    experiments/bridging/results/axis1_results.csv
    experiments/bridging/results/axis1_subsweep.csv
    experiments/bridging/results/axis1_figure.svg
    experiments/bridging/results/axis1_subsweep_figure.svg
"""

from __future__ import annotations

import csv
import math
import pathlib
import random
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .conditions_c import condition_c_community_notes, condition_c_polarity_product
from .personas import (
    ISSUES,
    Persona,
    _SBM_COMMUNITY_FRACTIONS,
    _assign_community,
    _clamp,
    _dirichlet,
    _sample_gamma,
)
from .policy import (
    SPILLOVER_CONFIGS,
    STANCES,
    PayoffTable,
    PolicyBundle,
    _CACHE_DIR,
    _config_hash,
    enumerate_bundles,
)
from .run_axis2 import (
    _ARCHETYPES,
    _CHUNK,
    build_oracle_fast,
    generate_slate,
)
from .simulation import _bundle_index, condition_a, condition_z

# ── Config ────────────────────────────────────────────────────────────────────

N_PERSONAS = 100
PERSONA_SEED = 42
K = 7
SLATE_TYPE = "archetype"
SPILLOVER_CFGS = ("centrist", "off_axis")
N_TRIALS = 100
N_TRIALS_SUBSWEEP = 30
BASE_SEED = 1000   # distinct from Axis 2 (BASE_SEED=0) to avoid cache collisions
CONDITIONS = ("Z", "A", "C_CN", "C_PP", "D")

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# ── Population distributions ──────────────────────────────────────────────────
# Each entry: (dist_id, dist_label, List[(weight, mean, std)])

DISTRIBUTIONS: List[Tuple[str, str, List[Tuple[float, float, float]]]] = [
    ("P1", "unimodal",        [(1.00,  0.00, 0.50)]),
    ("P2", "bimodal_sym",     [(0.40, -0.45, 0.25), (0.40, 0.45, 0.25), (0.20, 0.00, 0.15)]),
    ("P3", "bimodal_asym",    [(0.60, -0.40, 0.25), (0.30, 0.45, 0.25), (0.10, 0.00, 0.15)]),
    ("P4", "trimodal",        [(0.33, -0.50, 0.20), (0.33, 0.00, 0.20), (0.33, 0.50, 0.20)]),
    ("P5", "heavy_polar",     [(0.48, -0.80, 0.10), (0.48, 0.80, 0.10), (0.04, 0.00, 0.10)]),
]

# Polarization sub-sweep: σ_between values; symmetric bimodal 45/45/10, σ_within=0.25
SUBSWEEP_SEPARATIONS = [0.0, 0.30, 0.45, 0.60, 0.80]


# ── Persona generation with custom distribution ───────────────────────────────

def _sample_ideology_custom(
    components: List[Tuple[float, float, float]],
    rng: random.Random,
) -> float:
    """Sample from a Gaussian mixture. components: List[(weight, mean, std)]."""
    total_w = sum(c[0] for c in components)
    u = rng.random() * total_w
    cumulative = 0.0
    for weight, mean, std in components:
        cumulative += weight
        if u <= cumulative:
            return _clamp(rng.gauss(mean, std), -1.0, 1.0)
    weight, mean, std = components[-1]
    return _clamp(rng.gauss(mean, std), -1.0, 1.0)


def generate_personas_custom(
    components: List[Tuple[float, float, float]],
    n: int = 100,
    seed: int = 42,
    ideology_noise: float = 0.1,
) -> List[Persona]:
    """Generate N personas from a custom ideology mixture distribution.

    Args:
        components: Gaussian mixture components as List[(weight, mean, std)].
        n:          Number of personas.
        seed:       Random seed.
        ideology_noise: Per-issue ideal-point noise std (fraction of range).
    """
    rng = random.Random(seed)
    community_assignments = _assign_community(n, rng)

    personas = []
    for i in range(n):
        ideology = _sample_ideology_custom(components, rng)

        alpha = [0.5] * len(ISSUES)
        weights_raw = _dirichlet(alpha, rng)
        issue_weights = {issue: weights_raw[j] for j, issue in enumerate(ISSUES)}

        ideal_stances = {}
        for issue in ISSUES:
            base = ideology * 2.0
            noise = rng.gauss(0, ideology_noise * 2.0)
            ideal_stances[issue] = _clamp(base + noise, -2.0, 2.0)

        personas.append(Persona(
            persona_id=i,
            latent_ideology=ideology,
            issue_weights=issue_weights,
            ideal_stances=ideal_stances,
            network_community=community_assignments[i],
        ))

    return personas


def _subsweep_components(sigma_between: float) -> List[Tuple[float, float, float]]:
    """Build symmetric bimodal components for the polarization sub-sweep.

    μ_L = −σ_between/2, μ_R = +σ_between/2, σ_within = 0.25.
    Weights: 45% L + 45% R + 10% center N(0, 0.10).
    """
    sigma_within = 0.25
    mu_l = -sigma_between / 2.0
    mu_r =  sigma_between / 2.0
    return [
        (0.45, mu_l, sigma_within),
        (0.45, mu_r, sigma_within),
        (0.10, 0.00, 0.10),
    ]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class Axis1Result:
    dist_id: str
    dist_label: str
    spillover_config: str
    condition: str
    trial: int
    gap_fraction: float
    G_achieved: float
    G_random: float
    G_oracle: float
    minority_welfare: float  # 10th-percentile welfare score under this condition


def _minority_welfare(
    winner: PolicyBundle,
    personas: List[Persona],
    percentile: float = 0.10,
) -> float:
    """10th-percentile welfare score across personas for the winning bundle."""
    welfares = sorted(p.welfare(winner.stances) for p in personas)
    idx = max(0, int(math.floor(percentile * len(welfares))) - 1)
    return welfares[idx]


def run_trial_axis1(
    trial_idx: int,
    personas: List[Persona],
    table: PayoffTable,
    spillover_config: str,
    dist_id: str,
    dist_label: str,
) -> List[Axis1Result]:
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
            results.append(Axis1Result(
                dist_id=dist_id, dist_label=dist_label,
                spillover_config=spillover_config,
                condition="D", trial=trial_idx,
                gap_fraction=1.0, G_achieved=g_oracle,
                G_random=g_random, G_oracle=g_oracle,
                minority_welfare=_minority_welfare(table.oracle, personas),
            ))
            continue
        else:
            raise ValueError(cond)

        g_achieved = table.g_values[_bundle_index(winner)]
        gap = table.gap_fraction(g_achieved)
        results.append(Axis1Result(
            dist_id=dist_id, dist_label=dist_label,
            spillover_config=spillover_config,
            condition=cond, trial=trial_idx,
            gap_fraction=gap, G_achieved=g_achieved,
            G_random=g_random, G_oracle=g_oracle,
            minority_welfare=_minority_welfare(winner, personas),
        ))
    return results


# ── Sub-sweep result dataclass ────────────────────────────────────────────────

@dataclass
class SubsweepResult:
    sigma_between: float
    condition: str
    trial: int
    gap_fraction: float
    G_achieved: float
    G_random: float
    G_oracle: float


def run_trial_subsweep(
    trial_idx: int,
    personas: List[Persona],
    table: PayoffTable,
    sigma_between: float,
) -> List[SubsweepResult]:
    trial_seed = BASE_SEED + 5000 + trial_idx   # offset from main sweep seeds
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
            results.append(SubsweepResult(
                sigma_between=sigma_between, condition="D", trial=trial_idx,
                gap_fraction=1.0, G_achieved=g_oracle,
                G_random=g_random, G_oracle=g_oracle,
            ))
            continue
        else:
            raise ValueError(cond)

        g_achieved = table.g_values[_bundle_index(winner)]
        gap = table.gap_fraction(g_achieved)
        results.append(SubsweepResult(
            sigma_between=sigma_between, condition=cond, trial=trial_idx,
            gap_fraction=gap, G_achieved=g_achieved,
            G_random=g_random, G_oracle=g_oracle,
        ))
    return results


# ── Figures ───────────────────────────────────────────────────────────────────

def _svg_distribution_figure(
    results: List[Axis1Result],
    out_path: pathlib.Path,
) -> None:
    """Bar-chart: distributions (x) vs. median gap_fraction (y).

    One panel per condition subset; two bars per distribution (centrist / off_axis).
    Focuses on conditions A, C_CN, C_PP, D for readability.
    """
    W, H = 900, 480
    PAD_L, PAD_R, PAD_T, PAD_B = 65, 20, 50, 90
    PLOT_W = W - PAD_L - PAD_R
    PLOT_H = H - PAD_T - PAD_B

    SHOW_CONDS = ("A", "C_CN", "D")
    COND_COLORS = {"A": "#4C72B0", "C_CN": "#DD8452", "C_PP": "#55A868", "D": "#C44E52", "Z": "#999"}
    CFG_ALPHA = {"centrist": "ff", "off_axis": "99"}
    DIST_IDS = [d[0] for d in DISTRIBUTIONS]
    DIST_LABELS = [d[1] for d in DISTRIBUTIONS]
    N_DISTS = len(DIST_IDS)

    group_w = PLOT_W / N_DISTS
    bar_w = group_w / (len(SHOW_CONDS) * len(SPILLOVER_CFGS) + 1)

    def px(dist_i: int, bar_i: int) -> float:
        return PAD_L + dist_i * group_w + bar_i * bar_w + bar_w / 2

    def py(gap: float) -> float:
        return PAD_T + PLOT_H * (1.0 - max(0.0, min(1.0, gap)))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="monospace" font-size="11">',
        f'<rect width="{W}" height="{H}" fill="#f9f9f9"/>',
        f'<text x="{W//2}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#222">'
        f'Axis 1: Gap Fraction by Population Distribution (N={N_PERSONAS}, {N_TRIALS} trials/cell)</text>',
    ]

    # Plot area
    lines.append(
        f'<rect x="{PAD_L}" y="{PAD_T}" width="{PLOT_W}" height="{PLOT_H}" '
        f'fill="white" stroke="#ccc" stroke-width="1"/>'
    )

    # Y gridlines
    for ytick in [0.0, 0.25, 0.50, 0.75, 1.0]:
        y = py(ytick)
        lines.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L+PLOT_W}" y2="{y:.1f}" '
            f'stroke="#ddd" stroke-dasharray="4,3"/>'
        )
        lines.append(
            f'<text x="{PAD_L-4}" y="{y+4:.1f}" text-anchor="end" fill="#555">{ytick:.2f}</text>'
        )

    # Bars per distribution
    for di, (dist_id, dist_label) in enumerate(zip(DIST_IDS, DIST_LABELS)):
        # Group separator label
        cx = PAD_L + di * group_w + group_w / 2
        lines.append(
            f'<text x="{cx:.1f}" y="{PAD_T+PLOT_H+16}" text-anchor="middle" fill="#333">'
            f'{dist_id}</text>'
        )
        lines.append(
            f'<text x="{cx:.1f}" y="{PAD_T+PLOT_H+30}" text-anchor="middle" fill="#666" font-size="9">'
            f'{dist_label}</text>'
        )

        bar_i = 0
        for cond in SHOW_CONDS:
            color = COND_COLORS[cond]
            for cfg in SPILLOVER_CFGS:
                alpha_hex = CFG_ALPHA[cfg]
                vals = [r.gap_fraction for r in results
                        if r.dist_id == dist_id and r.condition == cond
                        and r.spillover_config == cfg]
                if not vals:
                    bar_i += 1
                    continue
                med = statistics.median(vals)
                x = px(di, bar_i)
                bar_top = py(med)
                bar_h = PAD_T + PLOT_H - bar_top
                lines.append(
                    f'<rect x="{x - bar_w/2:.1f}" y="{bar_top:.1f}" '
                    f'width="{bar_w:.1f}" height="{bar_h:.1f}" '
                    f'fill="{color}{alpha_hex}" stroke="none"/>'
                )
                bar_i += 1

    # Legend: conditions
    lx, ly = PAD_L, H - 35
    for cond in SHOW_CONDS:
        lines.append(
            f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{COND_COLORS[cond]}ff"/>'
        )
        lines.append(f'<text x="{lx+15}" y="{ly+10}" fill="#333">{cond}</text>')
        lx += 70
    lx += 15
    for cfg, alpha_hex in CFG_ALPHA.items():
        lines.append(
            f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="#666{alpha_hex}"/>'
        )
        lines.append(f'<text x="{lx+15}" y="{ly+10}" fill="#333">{cfg}</text>')
        lx += 100

    lines.append("</svg>")
    out_path.write_text("\n".join(lines))


def _svg_subsweep_figure(
    results: List[SubsweepResult],
    out_path: pathlib.Path,
) -> None:
    """Line chart: σ_between (x) vs. median gap_fraction (y), lines per condition."""
    W, H = 700, 420
    PAD_L, PAD_R, PAD_T, PAD_B = 65, 30, 50, 70
    PLOT_W = W - PAD_L - PAD_R
    PLOT_H = H - PAD_T - PAD_B

    COLORS = {"Z": "#999", "A": "#4C72B0", "C_CN": "#DD8452", "C_PP": "#55A868", "D": "#C44E52"}
    SHOW_CONDS = ("Z", "A", "C_CN", "C_PP", "D")

    x_vals = SUBSWEEP_SEPARATIONS
    x_min, x_max = 0.0, 0.80

    def px(s: float) -> float:
        return PAD_L + (s - x_min) / (x_max - x_min) * PLOT_W

    def py(gap: float) -> float:
        return PAD_T + PLOT_H * (1.0 - max(0.0, min(1.0, gap)))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="monospace" font-size="11">',
        f'<rect width="{W}" height="{H}" fill="#f9f9f9"/>',
        f'<text x="{W//2}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#222">'
        f'Axis 1b: Bridging Advantage vs. Polarization (N={N_PERSONAS}, {N_TRIALS_SUBSWEEP} trials)</text>',
        f'<text x="{W//2}" y="34" text-anchor="middle" font-size="10" fill="#555">'
        f'Symmetric bimodal, centrist spillover, K=7 archetype-seeded</text>',
        f'<rect x="{PAD_L}" y="{PAD_T+10}" width="{PLOT_W}" height="{PLOT_H}" '
        f'fill="white" stroke="#ccc" stroke-width="1"/>',
    ]

    # X axis label
    lines.append(
        f'<text x="{PAD_L + PLOT_W/2:.1f}" y="{PAD_T+PLOT_H+55}" text-anchor="middle" fill="#333">'
        f'σ_between (cluster separation)</text>'
    )
    # Y axis label
    lines.append(
        f'<text x="12" y="{PAD_T + PLOT_H/2:.1f}" text-anchor="middle" fill="#333" '
        f'transform="rotate(-90,12,{PAD_T + PLOT_H/2:.1f})">gap_fraction</text>'
    )

    # Y gridlines + ticks
    for ytick in [0.0, 0.25, 0.50, 0.75, 1.0]:
        y = py(ytick)
        lines.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L+PLOT_W}" y2="{y:.1f}" '
            f'stroke="#ddd" stroke-dasharray="4,3"/>'
        )
        lines.append(
            f'<text x="{PAD_L-4}" y="{y+14:.1f}" text-anchor="end" fill="#555">{ytick:.2f}</text>'
        )

    # X ticks
    for sv in x_vals:
        x = px(sv)
        lines.append(
            f'<line x1="{x:.1f}" y1="{PAD_T+10}" x2="{x:.1f}" y2="{PAD_T+PLOT_H+10}" '
            f'stroke="#eee"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{PAD_T+PLOT_H+25}" text-anchor="middle" fill="#555">'
            f'{sv:.2f}</text>'
        )

    # Lines per condition
    for cond in SHOW_CONDS:
        color = COLORS[cond]
        pts = []
        for sv in x_vals:
            vals = [r.gap_fraction for r in results
                    if abs(r.sigma_between - sv) < 1e-9 and r.condition == cond]
            if vals:
                med = statistics.median(vals)
                pts.append((px(sv), py(med)))
        if len(pts) >= 2:
            path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            lines.append(
                f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2.5"/>'
            )
        for x, y in pts:
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" fill-opacity="0.85"/>'
            )

    # Legend
    lx, ly = PAD_L, H - 25
    for cond in SHOW_CONDS:
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

    print(f"Axis 1: Distribution sweep — N={N_PERSONAS} personas/dist (seed={PERSONA_SEED}), "
          f"{N_TRIALS} trials/cell")

    print("Enumerating 5^8 bundles (390,625)...")
    bundles = enumerate_bundles()

    all_results: List[Axis1Result] = []

    # ── Main distribution sweep ───────────────────────────────────────────────
    for dist_id, dist_label, components in DISTRIBUTIONS:
        print(f"\n{'='*60}")
        print(f"Distribution {dist_id} ({dist_label}): {components}")
        personas = generate_personas_custom(components, n=N_PERSONAS, seed=PERSONA_SEED)

        mean_ideo = statistics.mean(p.latent_ideology for p in personas)
        std_ideo = statistics.stdev(p.latent_ideology for p in personas)
        print(f"  Ideology: mean={mean_ideo:+.3f}  std={std_ideo:.3f}")

        for cfg in SPILLOVER_CFGS:
            print(f"\n  Building oracle ('{cfg}', vectorized)...")
            t0 = time.time()
            table = build_oracle_fast(personas, cfg, bundles, cache=True)
            elapsed = time.time() - t0
            print(f"    {elapsed:.1f}s  G_oracle={table.g_oracle:.4f}  G_random={table.g_random:.4f}")

            cell_results = []
            for trial_idx in range(N_TRIALS):
                res = run_trial_axis1(
                    trial_idx, personas, table, cfg, dist_id, dist_label,
                )
                cell_results.extend(res)
            all_results.extend(cell_results)

            # Cell summary
            for cond in ("A", "C_CN", "D"):
                vals = [r.gap_fraction for r in cell_results if r.condition == cond]
                med = statistics.median(vals)
                print(f"    {cond}: med gap={med:.3f}")

    # Print distribution comparison table
    print("\n── C_CN vs A gap delta across distributions ─────────────────────────")
    print(f"{'Dist':<6} {'Config':<12}  C_CN   A      delta   verdict")
    print("-" * 60)
    for dist_id, dist_label, _ in DISTRIBUTIONS:
        for cfg in SPILLOVER_CFGS:
            c_med = statistics.median([r.gap_fraction for r in all_results
                if r.dist_id == dist_id and r.spillover_config == cfg and r.condition == "C_CN"])
            a_med = statistics.median([r.gap_fraction for r in all_results
                if r.dist_id == dist_id and r.spillover_config == cfg and r.condition == "A"])
            delta = c_med - a_med
            verdict = "H1a✓ (no struct)" if delta < 0.05 else (
                      "strong" if delta > 0.30 else "moderate")
            print(f"{dist_id:<6} {cfg:<12}  {c_med:.3f}  {a_med:.3f}  {delta:+.3f}  {verdict}")

    # Minority welfare P2 vs P3
    print("\n── Minority welfare (10th pctile) P2 vs P3 ──────────────────────────")
    for dist_id in ("P2", "P3"):
        for cond in ("A", "C_CN"):
            vals = [r.minority_welfare for r in all_results
                    if r.dist_id == dist_id and r.condition == cond
                    and r.spillover_config == "centrist"]
            if vals:
                med = statistics.median(vals)
                print(f"  {dist_id} {cond}: med minority welfare = {med:.4f}")

    # Write main CSV
    csv_path = RESULTS_DIR / "axis1_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "dist_id", "dist_label", "spillover_config", "condition", "trial",
            "gap_fraction", "G_achieved", "G_random", "G_oracle", "minority_welfare",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "dist_id": r.dist_id, "dist_label": r.dist_label,
                "spillover_config": r.spillover_config, "condition": r.condition,
                "trial": r.trial, "gap_fraction": f"{r.gap_fraction:.6f}",
                "G_achieved": f"{r.G_achieved:.6f}", "G_random": f"{r.G_random:.6f}",
                "G_oracle": f"{r.G_oracle:.6f}",
                "minority_welfare": f"{r.minority_welfare:.6f}",
            })
    print(f"\nWrote {len(all_results)} rows → {csv_path}")

    # Write distribution figure
    svg_path = RESULTS_DIR / "axis1_figure.svg"
    _svg_distribution_figure(all_results, svg_path)
    print(f"Wrote figure → {svg_path}")

    # ── Polarization sub-sweep ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Axis 1b: Polarization sub-sweep — {len(SUBSWEEP_SEPARATIONS)} levels × "
          f"{N_TRIALS_SUBSWEEP} trials (centrist spillover, N={N_PERSONAS})")

    subsweep_results: List[SubsweepResult] = []

    for sigma_between in SUBSWEEP_SEPARATIONS:
        components = _subsweep_components(sigma_between)
        print(f"\n  σ_between={sigma_between:.2f}: {components}")
        personas = generate_personas_custom(components, n=N_PERSONAS, seed=PERSONA_SEED)

        t0 = time.time()
        table = build_oracle_fast(personas, "centrist", bundles, cache=True)
        elapsed = time.time() - t0
        print(f"    oracle {elapsed:.1f}s  G_oracle={table.g_oracle:.4f}  G_random={table.g_random:.4f}")

        for trial_idx in range(N_TRIALS_SUBSWEEP):
            res = run_trial_subsweep(trial_idx, personas, table, sigma_between)
            subsweep_results.extend(res)

        for cond in ("A", "C_CN"):
            vals = [r.gap_fraction for r in subsweep_results
                    if abs(r.sigma_between - sigma_between) < 1e-9 and r.condition == cond]
            if vals:
                med = statistics.median(vals)
                print(f"    {cond}: med gap={med:.3f}")

    # Subsweep comparison
    print("\n── Sub-sweep polarization curve (C_CN − A delta) ────────────────────")
    print(f"{'σ_between':>10}   C_CN   A      delta")
    print("-" * 40)
    for sv in SUBSWEEP_SEPARATIONS:
        c_med = statistics.median([r.gap_fraction for r in subsweep_results
            if abs(r.sigma_between - sv) < 1e-9 and r.condition == "C_CN"])
        a_med = statistics.median([r.gap_fraction for r in subsweep_results
            if abs(r.sigma_between - sv) < 1e-9 and r.condition == "A"])
        print(f"  {sv:.2f}      {c_med:.3f}  {a_med:.3f}  {c_med-a_med:+.3f}")

    # Write subsweep CSV
    subsweep_csv = RESULTS_DIR / "axis1_subsweep.csv"
    with open(subsweep_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "sigma_between", "condition", "trial",
            "gap_fraction", "G_achieved", "G_random", "G_oracle",
        ])
        writer.writeheader()
        for r in subsweep_results:
            writer.writerow({
                "sigma_between": f"{r.sigma_between:.2f}", "condition": r.condition,
                "trial": r.trial, "gap_fraction": f"{r.gap_fraction:.6f}",
                "G_achieved": f"{r.G_achieved:.6f}", "G_random": f"{r.G_random:.6f}",
                "G_oracle": f"{r.G_oracle:.6f}",
            })
    print(f"\nWrote {len(subsweep_results)} rows → {subsweep_csv}")

    subsweep_svg = RESULTS_DIR / "axis1_subsweep_figure.svg"
    _svg_subsweep_figure(subsweep_results, subsweep_svg)
    print(f"Wrote subsweep figure → {subsweep_svg}")

    print("\nAxis 1 complete.")


if __name__ == "__main__":
    main()
