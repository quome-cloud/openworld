"""Cycle 5 experiment runner: 50 trials × 5 conditions × 2 spillover configs.

Run from the openworld repo root:
    python -m experiments.bridging.run_cycle5

Outputs:
    experiments/bridging/results/cycle5_results.csv
    experiments/bridging/results/gap_fraction_boxplot.svg
"""

from __future__ import annotations

import csv
import math
import pathlib
import statistics
import sys
from typing import Dict, List, Tuple

from .personas import generate_personas
from .policy import build_payoff_table, enumerate_bundles
from .simulation import run_trial

# ── Config ────────────────────────────────────────────────────────────────────

N_PERSONAS = 20
PERSONA_SEED = 0
N_TRIALS = 50
BASE_SEED = 0
CONDITIONS = ("Z", "A", "C_CN", "C_PP", "D")
SPILLOVER_CONFIGS = ("centrist", "off_axis")

RESULTS_DIR = pathlib.Path(__file__).parent / "results"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _boxplot_stats(values: List[float]) -> Dict:
    s = sorted(values)
    n = len(s)
    q1 = statistics.median(s[: n // 2])
    q3 = statistics.median(s[(n + 1) // 2 :])
    med = statistics.median(s)
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    whislo = min(v for v in s if v >= lo_fence)
    whishi = max(v for v in s if v <= hi_fence)
    outliers = [v for v in s if v < lo_fence or v > hi_fence]
    return {"q1": q1, "med": med, "q3": q3, "whislo": whislo, "whishi": whishi,
            "outliers": outliers, "mean": statistics.mean(s)}


def _svg_boxplot(
    stats_by_group: Dict[Tuple[str, str], Dict],
    configs: List[str],
    conditions: List[str],
    out_path: pathlib.Path,
) -> None:
    """Write a grouped boxplot SVG (configs side by side, conditions on x-axis)."""
    # Layout
    W, H = 900, 500
    LEFT, RIGHT, TOP, BOTTOM = 70, 30, 40, 90
    plot_w = W - LEFT - RIGHT
    plot_h = H - TOP - BOTTOM

    n_cond = len(conditions)
    n_conf = len(configs)
    group_w = plot_w / n_cond
    box_w = group_w / (n_conf + 1) * 0.7

    COLORS = {"centrist": "#4C72B0", "off_axis": "#DD8452"}
    COND_LABELS = {"Z": "Z\n(random)", "A": "A\n(majority)", "C_CN": "C_CN\n(Comm.Notes)",
                   "C_PP": "C_PP\n(polarity)", "D": "D\n(oracle)"}

    def yscale(v: float) -> float:
        # gap_fraction 0→1, mapped to plot_h
        return TOP + plot_h * (1.0 - max(0.0, min(1.0, v)))

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="monospace" font-size="11">',
    ]

    # Background
    lines.append(f'<rect width="{W}" height="{H}" fill="#f9f9f9"/>')

    # Y-axis gridlines + labels
    for ytick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = yscale(ytick)
        lines.append(
            f'<line x1="{LEFT}" y1="{y:.1f}" x2="{W-RIGHT}" y2="{y:.1f}" '
            f'stroke="#ddd" stroke-dasharray="4,3"/>'
        )
        lines.append(
            f'<text x="{LEFT-6}" y="{y+4:.1f}" text-anchor="end" fill="#555">'
            f'{ytick:.2f}</text>'
        )

    # Y-axis label
    lines.append(
        f'<text x="14" y="{TOP + plot_h/2:.0f}" text-anchor="middle" '
        f'transform="rotate(-90,14,{TOP + plot_h/2:.0f})" fill="#333" font-size="12">'
        f'Gap fraction</text>'
    )

    # Legend
    for ci, cfg in enumerate(configs):
        lx = LEFT + 10 + ci * 160
        lines.append(f'<rect x="{lx}" y="8" width="14" height="10" fill="{COLORS[cfg]}"/>')
        lines.append(f'<text x="{lx+18}" y="18" fill="#333">{cfg}</text>')

    # Boxes
    for ci, cond in enumerate(conditions):
        cx = LEFT + (ci + 0.5) * group_w

        for fi, cfg in enumerate(configs):
            stats = stats_by_group[(cfg, cond)]
            offset = (fi - (n_conf - 1) / 2) * (box_w + 4)
            bx = cx + offset

            q1y = yscale(stats["q1"])
            q3y = yscale(stats["q3"])
            medy = yscale(stats["med"])
            lowy = yscale(stats["whislo"])
            highy = yscale(stats["whishi"])
            color = COLORS[cfg]
            box_h = abs(q1y - q3y)
            box_top = min(q1y, q3y)

            # Whisker lines
            lines.append(
                f'<line x1="{bx:.1f}" y1="{highy:.1f}" x2="{bx:.1f}" y2="{q3y:.1f}" '
                f'stroke="{color}" stroke-width="1.5"/>'
            )
            lines.append(
                f'<line x1="{bx:.1f}" y1="{q1y:.1f}" x2="{bx:.1f}" y2="{lowy:.1f}" '
                f'stroke="{color}" stroke-width="1.5"/>'
            )
            # Whisker caps
            cap_w = box_w * 0.4
            for wy in (highy, lowy):
                lines.append(
                    f'<line x1="{bx-cap_w:.1f}" y1="{wy:.1f}" x2="{bx+cap_w:.1f}" y2="{wy:.1f}" '
                    f'stroke="{color}" stroke-width="1.5"/>'
                )
            # Box rect
            lines.append(
                f'<rect x="{bx-box_w/2:.1f}" y="{box_top:.1f}" '
                f'width="{box_w:.1f}" height="{box_h:.1f}" '
                f'fill="{color}" fill-opacity="0.3" stroke="{color}" stroke-width="1.5"/>'
            )
            # Median line
            lines.append(
                f'<line x1="{bx-box_w/2:.1f}" y1="{medy:.1f}" '
                f'x2="{bx+box_w/2:.1f}" y2="{medy:.1f}" '
                f'stroke="{color}" stroke-width="2.5"/>'
            )
            # Mean dot
            meany = yscale(stats["mean"])
            lines.append(
                f'<circle cx="{bx:.1f}" cy="{meany:.1f}" r="3" '
                f'fill="{color}" fill-opacity="0.7"/>'
            )
            # Outliers
            for ov in stats["outliers"]:
                oy = yscale(ov)
                lines.append(
                    f'<circle cx="{bx:.1f}" cy="{oy:.1f}" r="2.5" '
                    f'fill="none" stroke="{color}" stroke-width="1"/>'
                )

        # X-axis condition label
        label = COND_LABELS.get(cond, cond)
        for li, part in enumerate(label.split("\n")):
            lines.append(
                f'<text x="{cx:.1f}" y="{H - BOTTOM + 18 + li * 14}" '
                f'text-anchor="middle" fill="#333">{part}</text>'
            )

    # Axes
    lines.append(
        f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{TOP+plot_h}" '
        f'stroke="#333" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{LEFT}" y1="{TOP+plot_h}" x2="{W-RIGHT}" y2="{TOP+plot_h}" '
        f'stroke="#333" stroke-width="1.5"/>'
    )

    # Title
    lines.append(
        f'<text x="{W/2:.0f}" y="16" text-anchor="middle" fill="#222" font-size="13" '
        f'font-weight="bold">World 1 Bridging: Gap Fraction by Condition '
        f'(N={N_PERSONAS} personas, {N_TRIALS} trials)</text>'
    )

    lines.append("</svg>")
    out_path.write_text("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {N_PERSONAS} personas (seed={PERSONA_SEED})...")
    personas = generate_personas(n=N_PERSONAS, seed=PERSONA_SEED)

    print("Enumerating 5^8 = 390,625 bundles...")
    bundles = enumerate_bundles()

    all_results = []

    for cfg in SPILLOVER_CONFIGS:
        print(f"\nBuilding oracle table for '{cfg}' (cache if available)...")
        table = build_payoff_table(personas, cfg, bundles=bundles, cache=True)
        print(f"  G_oracle={table.g_oracle:.4f}  G_random={table.g_random:.4f}")

        for trial_idx in range(N_TRIALS):
            results = run_trial(
                trial_idx, personas, table, cfg,
                conditions=CONDITIONS, base_seed=BASE_SEED,
            )
            all_results.extend(results)
            if trial_idx % 10 == 9 or trial_idx == N_TRIALS - 1:
                gaps = {r.condition: r.gap_fraction for r in results}
                print(f"  trial {trial_idx+1:3d}/{N_TRIALS} "
                      + "  ".join(f"{c}={gaps[c]:.3f}" for c in CONDITIONS))

    # Write CSV
    csv_path = RESULTS_DIR / "cycle5_results.csv"
    fieldnames = ["spillover_config", "condition", "trial", "gap_fraction",
                  "G_achieved", "G_random", "G_oracle"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "spillover_config": r.spillover_config,
                "condition": r.condition,
                "trial": r.trial,
                "gap_fraction": f"{r.gap_fraction:.6f}",
                "G_achieved": f"{r.G_achieved:.6f}",
                "G_random": f"{r.G_random:.6f}",
                "G_oracle": f"{r.G_oracle:.6f}",
            })
    print(f"\nWrote {len(all_results)} rows → {csv_path}")

    # Compute boxplot stats
    stats_by_group: Dict[Tuple[str, str], Dict] = {}
    for cfg in SPILLOVER_CONFIGS:
        for cond in CONDITIONS:
            vals = [r.gap_fraction for r in all_results
                    if r.spillover_config == cfg and r.condition == cond]
            stats_by_group[(cfg, cond)] = _boxplot_stats(vals)

    # Print summary table
    print("\n── Gap Fraction Summary (median ± IQR) ──────────────────────────────")
    header = f"{'Condition':<10}" + "".join(f"  {c:<22}" for c in SPILLOVER_CONFIGS)
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        row = f"{cond:<10}"
        for cfg in SPILLOVER_CONFIGS:
            s = stats_by_group[(cfg, cond)]
            row += f"  med={s['med']:.3f} mean={s['mean']:.3f} IQR=[{s['q1']:.3f},{s['q3']:.3f}]"
        print(row)

    # Write SVG
    svg_path = RESULTS_DIR / "gap_fraction_boxplot.svg"
    _svg_boxplot(stats_by_group, SPILLOVER_CONFIGS, list(CONDITIONS), svg_path)
    print(f"\nWrote figure → {svg_path}")

    # Print centrism-laundering test result
    print("\n── Centrism-Laundering Null Test ────────────────────────────────────")
    for cond in ("A", "C_CN", "C_PP"):
        c_med = stats_by_group[("centrist", cond)]["med"]
        o_med = stats_by_group[("off_axis", cond)]["med"]
        verdict = "HOLDS (C≈A off_axis)" if cond != "Z" and abs(c_med - o_med) < 0.05 else "REJECTED (C>>A off_axis)"
        print(f"  {cond}: centrist_med={c_med:.3f}  off_axis_med={o_med:.3f}  → {verdict}")


if __name__ == "__main__":
    main()
