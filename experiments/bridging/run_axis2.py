"""Axis 2: Slate-size sweep — K ∈ {5, 7, 15, 30} × {archetype, random} × 2 spillover configs.

Tests whether the bridging advantage over majority vote is:
  (a) stable as K grows with archetype seeds present, and
  (b) emerges in fully-random slates at sufficiently large K.

Uses a numpy-vectorized oracle builder (chunked batching over bundles) to bring
N=300 oracle precomputation from ~30 min (pure Python) to ~5 s (numpy).

Run from the openworld repo root:
    python -m experiments.bridging.run_axis2

Outputs:
    experiments/bridging/results/axis2_results.csv
    experiments/bridging/results/axis2_figure.svg
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import pickle
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .conditions_c import condition_c_community_notes, condition_c_polarity_product
from .personas import ISSUES, Persona, generate_personas
from .policy import (
    SPILLOVER_CONFIGS,
    STANCES,
    PayoffTable,
    PolicyBundle,
    _CACHE_DIR,
    _config_hash,
    enumerate_bundles,
)
from .simulation import _bundle_index, condition_a, condition_z

# ── Config ────────────────────────────────────────────────────────────────────

N_PERSONAS = 100
PERSONA_SEED = 42
K_VALUES = (5, 7, 15, 30)
SLATE_TYPES = ("archetype", "random")
SPILLOVER_CFGS = ("centrist", "off_axis")
N_TRIALS = 100
BASE_SEED = 0
CONDITIONS = ("Z", "A", "C_CN", "C_PP", "D")

_ARCHETYPE_STANCES = (-2, -1, 0, 1, 2)
N_ARCHETYPES = 5

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
_CHUNK = 500    # bundles per numpy chunk; keeps peak tensor <10 MB at N=100


# ── Vectorized oracle builder ─────────────────────────────────────────────────

def build_oracle_fast(
    personas: List[Persona],
    spillover_config: str,
    bundles: List[PolicyBundle],
    cache: bool = True,
) -> PayoffTable:
    """Numpy-batched PayoffTable builder.  Same cache format as build_payoff_table().

    Processes bundles in chunks of _CHUNK to bound peak memory, then assembles
    the full G vector.  Typical wall-clock: <5s for N=300, M=390k bundles.
    """
    if cache:
        _CACHE_DIR.mkdir(exist_ok=True)
        key = _config_hash(personas, spillover_config)
        path = _CACHE_DIR / f"oracle_{key}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                table = pickle.load(fh)
            if len(table.bundles) == len(bundles):
                table = PayoffTable(
                    bundles=bundles,
                    g_values=table.g_values,
                    oracle=table.oracle,
                    g_oracle=table.g_oracle,
                    g_random=table.g_random,
                    oracle_index=table.oracle_index,
                )
            return table

    M = len(bundles)
    N = len(personas)

    # Pre-extract persona arrays — (N, 8); small, always safe
    ideals = np.array(
        [[p.ideal_stances[issue] for issue in ISSUES] for p in personas],
        dtype=np.float64,
    )
    weights = np.array(
        [[p.issue_weights[issue] for issue in ISSUES] for p in personas],
        dtype=np.float64,
    )

    spillovers = SPILLOVER_CONFIGS[spillover_config]
    g_values_np = np.empty(M, dtype=np.float64)

    # Process bundles in chunks: extract stances per-chunk to avoid pre-allocating
    # the full (M, 8) bundle matrix (25 MB) alongside the (C, N, 8) work tensor.
    for start in range(0, M, _CHUNK):
        chunk_bundles = bundles[start : start + _CHUNK]
        chunk_stances = np.array(
            [[b.stances[issue] for issue in ISSUES] for b in chunk_bundles],
            dtype=np.float64,
        )                                                     # (C, 8)
        C = len(chunk_bundles)

        # Welfare: (C, N, 8) → mean over N
        diffs = (chunk_stances[:, None, :] - ideals[None, :, :]) / 4.0
        welfare = ((1.0 - diffs ** 2) * weights[None, :, :]).sum(axis=2)  # (C, N)
        g_chunk = welfare.mean(axis=1)                        # (C,)

        # Spillover bonuses for this chunk
        for pattern, amount in spillovers:
            mask = np.ones(C, dtype=bool)
            for issue, required in pattern.items():
                j = ISSUES.index(issue)
                mask &= (chunk_stances[:, j] == required)
            g_chunk[mask] += amount

        g_values_np[start : start + C] = g_chunk

    g_values: List[float] = g_values_np.tolist()
    oracle_index = int(np.argmax(g_values_np))
    g_oracle = float(g_values_np[oracle_index])
    g_random = float(g_values_np.mean())

    table = PayoffTable(
        bundles=bundles,
        g_values=g_values,
        oracle=bundles[oracle_index],
        g_oracle=g_oracle,
        g_random=g_random,
        oracle_index=oracle_index,
    )

    if cache:
        with open(path, "wb") as fh:
            pickle.dump(table, fh)

    return table


# ── Slate generators ──────────────────────────────────────────────────────────

def _archetype_bundle(stance: int) -> PolicyBundle:
    return PolicyBundle(stances={issue: stance for issue in ISSUES})


_ARCHETYPES = [_archetype_bundle(s) for s in _ARCHETYPE_STANCES]


def generate_slate(K: int, slate_type: str, trial_seed: int) -> List[PolicyBundle]:
    """Return a K-bundle candidate slate.

    slate_type='archetype': 5 fixed archetypes + (K-5) random-per-trial bundles.
    slate_type='random': K fully random bundles per trial.

    K must be >= 5 for archetype type (enforced by K_VALUES = {5,7,15,30}).
    """
    rng = random.Random(trial_seed)
    if slate_type == "archetype":
        slate = list(_ARCHETYPES)  # always 5 archetypes
        n_random = max(0, K - N_ARCHETYPES)
        for _ in range(n_random):
            slate.append(PolicyBundle(stances={i: rng.choice(STANCES) for i in ISSUES}))
    else:
        slate = [
            PolicyBundle(stances={i: rng.choice(STANCES) for i in ISSUES})
            for _ in range(K)
        ]
    return slate


# ── Trial runner ──────────────────────────────────────────────────────────────

@dataclass
class Axis2Result:
    K: int
    slate_type: str
    spillover_config: str
    condition: str
    trial: int
    gap_fraction: float
    G_achieved: float
    G_random: float
    G_oracle: float


def run_trial_axis2(
    trial_idx: int,
    K: int,
    slate_type: str,
    personas: List[Persona],
    table: PayoffTable,
    spillover_config: str,
) -> List[Axis2Result]:
    trial_seed = BASE_SEED + trial_idx
    rng = random.Random(trial_seed)
    slate = generate_slate(K, slate_type, trial_seed)

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
            results.append(Axis2Result(
                K=K, slate_type=slate_type, spillover_config=spillover_config,
                condition="D", trial=trial_idx,
                gap_fraction=1.0, G_achieved=g_oracle,
                G_random=g_random, G_oracle=g_oracle,
            ))
            continue
        else:
            raise ValueError(cond)

        g_achieved = table.g_values[_bundle_index(winner)]
        gap = table.gap_fraction(g_achieved)
        results.append(Axis2Result(
            K=K, slate_type=slate_type, spillover_config=spillover_config,
            condition=cond, trial=trial_idx,
            gap_fraction=gap, G_achieved=g_achieved,
            G_random=g_random, G_oracle=g_oracle,
        ))
    return results


# ── Figure (SVG line chart: K vs gap_fraction) ────────────────────────────────

def _svg_line_chart(
    results: List[Axis2Result],
    out_path: pathlib.Path,
) -> None:
    """Two-panel SVG: archetype slate (left) and random slate (right).
    Each panel: K on x-axis, median gap_fraction on y-axis, lines per condition.
    """
    W, H = 1000, 480
    PAD_L, PAD_R, PAD_T, PAD_B = 60, 20, 45, 70
    PANEL_W = (W - PAD_L - PAD_R - 40) // 2
    PANEL_H = H - PAD_T - PAD_B

    COLORS = {"Z": "#999", "A": "#4C72B0", "C_CN": "#DD8452", "C_PP": "#55A868", "D": "#C44E52"}
    DASHES = {"centrist": "", "off_axis": "6,3"}
    K_VALS = list(K_VALUES)
    X_POSITIONS = {k: i / (len(K_VALS) - 1) for i, k in enumerate(K_VALS)}

    def px(x_frac: float, panel_left: float) -> float:
        return panel_left + x_frac * PANEL_W

    def py(gap: float) -> float:
        return PAD_T + PANEL_H * (1.0 - max(0.0, min(1.0, gap)))

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="monospace" font-size="11">',
        f'<rect width="{W}" height="{H}" fill="#f9f9f9"/>',
    ]

    # Title
    lines.append(
        f'<text x="{W//2}" y="18" text-anchor="middle" font-size="13" '
        f'font-weight="bold" fill="#222">Axis 2: Gap Fraction vs Slate Size K '
        f'(N={N_PERSONAS} personas, {N_TRIALS} trials/cell)</text>'
    )

    for panel_idx, slate_type in enumerate(SLATE_TYPES):
        pl = PAD_L + panel_idx * (PANEL_W + 40)
        panel_label = "Archetype-seeded slates" if slate_type == "archetype" else "Fully-random slates"

        # Panel background + label
        lines.append(
            f'<rect x="{pl}" y="{PAD_T}" width="{PANEL_W}" height="{PANEL_H}" '
            f'fill="white" stroke="#ccc" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{pl + PANEL_W//2}" y="{PAD_T - 6}" text-anchor="middle" '
            f'fill="#333" font-size="12" font-weight="bold">{panel_label}</text>'
        )

        # Y gridlines
        for ytick in [0.0, 0.25, 0.5, 0.75, 1.0]:
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

        # X axis labels
        for k in K_VALS:
            x = px(X_POSITIONS[k], pl)
            lines.append(
                f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{PAD_T+PANEL_H}" '
                f'stroke="#eee"/>'
            )
            lines.append(
                f'<text x="{x:.1f}" y="{PAD_T+PANEL_H+16}" text-anchor="middle" fill="#555">'
                f'K={k}</text>'
            )

        # Lines per condition × spillover_config
        for cond in CONDITIONS:
            color = COLORS[cond]
            for cfg in SPILLOVER_CFGS:
                dash = DASHES[cfg]
                pts = []
                for k in K_VALS:
                    vals = [r.gap_fraction for r in results
                            if r.K == k and r.slate_type == slate_type
                            and r.spillover_config == cfg and r.condition == cond]
                    if vals:
                        med = statistics.median(vals)
                        pts.append((px(X_POSITIONS[k], pl), py(med)))

                if len(pts) >= 2:
                    path_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                    lines.append(
                        f'<path d="{path_d}" fill="none" stroke="{color}" '
                        f'stroke-width="2" stroke-dasharray="{dash}"/>'
                    )
                for x, y in pts:
                    lines.append(
                        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" '
                        f'fill="{color}" fill-opacity="0.8"/>'
                    )

    # Legend: conditions
    lx, ly = PAD_L, H - 30
    for cond in CONDITIONS:
        lines.append(
            f'<line x1="{lx}" y1="{ly+5}" x2="{lx+20}" y2="{ly+5}" '
            f'stroke="{COLORS[cond]}" stroke-width="2.5"/>'
        )
        lines.append(f'<text x="{lx+24}" y="{ly+9}" fill="#333">{cond}</text>')
        lx += 70

    # Legend: spillover configs
    lx += 20
    for cfg, dash in DASHES.items():
        lines.append(
            f'<line x1="{lx}" y1="{ly+5}" x2="{lx+20}" y2="{ly+5}" '
            f'stroke="#888" stroke-width="2" stroke-dasharray="{dash}"/>'
        )
        lines.append(f'<text x="{lx+24}" y="{ly+9}" fill="#333">{cfg}</text>')
        lx += 90

    lines.append("</svg>")
    out_path.write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Axis 2: K-sweep — N={N_PERSONAS} personas (seed={PERSONA_SEED}), "
          f"{N_TRIALS} trials/cell")
    personas = generate_personas(n=N_PERSONAS, seed=PERSONA_SEED)

    print("Enumerating 5^8 bundles...")
    bundles = enumerate_bundles()

    all_results: List[Axis2Result] = []

    for cfg in SPILLOVER_CFGS:
        print(f"\nBuilding oracle table (N={N_PERSONAS}, '{cfg}', vectorized)...")
        import time
        t0 = time.time()
        table = build_oracle_fast(personas, cfg, bundles, cache=True)
        elapsed = time.time() - t0
        print(f"  {elapsed:.1f}s  G_oracle={table.g_oracle:.4f}  G_random={table.g_random:.4f}")

        for K in K_VALUES:
            for slate_type in SLATE_TYPES:
                cell_results = []
                for trial_idx in range(N_TRIALS):
                    res = run_trial_axis2(trial_idx, K, slate_type, personas, table, cfg)
                    cell_results.extend(res)
                all_results.extend(cell_results)

                # Print cell summary
                for cond in ("A", "C_CN", "D"):
                    vals = [r.gap_fraction for r in cell_results if r.condition == cond]
                    med = statistics.median(vals)
                    print(f"  K={K:2d} {slate_type:<10} {cfg:<10} {cond}: med={med:.3f}")

    # Write CSV
    csv_path = RESULTS_DIR / "axis2_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "K", "slate_type", "spillover_config", "condition", "trial",
            "gap_fraction", "G_achieved", "G_random", "G_oracle",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "K": r.K, "slate_type": r.slate_type,
                "spillover_config": r.spillover_config, "condition": r.condition,
                "trial": r.trial, "gap_fraction": f"{r.gap_fraction:.6f}",
                "G_achieved": f"{r.G_achieved:.6f}",
                "G_random": f"{r.G_random:.6f}", "G_oracle": f"{r.G_oracle:.6f}",
            })
    print(f"\nWrote {len(all_results)} rows → {csv_path}")

    # Print C vs A gap table
    print("\n── C_CN vs A gap_fraction delta ────────────────────────────────────")
    print(f"{'K':>4}  {'slate_type':<12} {'config':<12}  C_CN   A      delta")
    print("-" * 60)
    for K in K_VALUES:
        for slate_type in SLATE_TYPES:
            for cfg in SPILLOVER_CFGS:
                c_med = statistics.median([r.gap_fraction for r in all_results
                    if r.K == K and r.slate_type == slate_type
                    and r.spillover_config == cfg and r.condition == "C_CN"])
                a_med = statistics.median([r.gap_fraction for r in all_results
                    if r.K == K and r.slate_type == slate_type
                    and r.spillover_config == cfg and r.condition == "A"])
                verdict = "robust" if c_med - a_med > 0.05 else "C≈A"
                print(f"{K:>4}  {slate_type:<12} {cfg:<12}  "
                      f"{c_med:.3f}  {a_med:.3f}  {c_med-a_med:+.3f}  {verdict}")

    # Write SVG
    svg_path = RESULTS_DIR / "axis2_figure.svg"
    _svg_line_chart(all_results, svg_path)
    print(f"\nWrote figure → {svg_path}")

    # H2b verdict: does random-slate advantage emerge at K=30?
    print("\n── H2b: Random-slate threshold test ────────────────────────────────")
    for K in K_VALUES:
        for cfg in SPILLOVER_CFGS:
            c_med = statistics.median([r.gap_fraction for r in all_results
                if r.K == K and r.slate_type == "random"
                and r.spillover_config == cfg and r.condition == "C_CN"])
            a_med = statistics.median([r.gap_fraction for r in all_results
                if r.K == K and r.slate_type == "random"
                and r.spillover_config == cfg and r.condition == "A"])
            verdict = f"THRESHOLD (C>>A, delta={c_med-a_med:+.3f})" if c_med - a_med > 0.05 \
                else f"no emergence (delta={c_med-a_med:+.3f})"
            print(f"  K={K:2d} {cfg}: {verdict}")


if __name__ == "__main__":
    main()
