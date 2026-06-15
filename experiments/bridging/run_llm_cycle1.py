"""LLM-persona validation runner — Cycle 1 (T355).

Replaces parametric welfare-based endorsements with 20 LLM-prompted personas
(claude-haiku-4-5-20251001, temp=0) across Conditions A and C_CN.

Same slate structure as cycle5 (5 archetypes + 2 random per trial, N=20 parametric
personas for the payoff table, seed=0, 50 trials). The welfare function and spillover
bonuses are unchanged — only endorsement generation is replaced.

Run from the openworld repo root:
    python -m experiments.bridging.run_llm_cycle1

Outputs:
    experiments/bridging/results/llm_cycle1_results.csv
    experiments/bridging/results/llm_cycle1_summary.txt
    experiments/bridging/llm_cache.json  (persists across runs for reproducibility)
"""

from __future__ import annotations

import csv
import os
import pathlib
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple

import anthropic
import numpy as np

from .conditions_c import fit_community_notes
from .llm_endorsement import build_llm_endorsement_matrix, load_cache, save_cache
from .llm_personas import get_personas
from .personas import generate_personas
from .policy import PolicyBundle, build_payoff_table, enumerate_bundles
from .simulation import _bundle_index, generate_candidate_slate

# ── Config ────────────────────────────────────────────────────────────────────

N_PERSONAS = 20
PERSONA_SEED = 0
N_TRIALS = 50
BASE_SEED = 0
CONDITIONS = ("A", "C_CN")
SPILLOVER_CONFIGS = ("centrist", "off_axis")

RESULTS_DIR = pathlib.Path(__file__).parent / "results"
CACHE_PATH = pathlib.Path(__file__).parent / "llm_cache.json"

# Parametric baselines from cycle5 (for comparison table)
PARAMETRIC_BASELINES: Dict[Tuple[str, str], float] = {
    ("A",    "centrist"): 0.306,
    ("A",    "off_axis"): 0.495,
    ("C_CN", "centrist"): 0.793,
    ("C_CN", "off_axis"): 0.904,
}


# ── Trial result ──────────────────────────────────────────────────────────────

@dataclass
class LLMTrialResult:
    spillover_config: str
    condition: str
    trial: int
    gap_fraction: float
    G_achieved: float
    G_random: float
    G_oracle: float
    n_llm_calls: int
    cache_hit_rate: float


# ── Condition helpers ─────────────────────────────────────────────────────────

def _condition_a_llm(slate: List[PolicyBundle], E: np.ndarray) -> PolicyBundle:
    """Majority vote using LLM endorsement matrix E (N_personas × N_bundles)."""
    counts = E.sum(axis=0)
    max_count = float(counts.max())
    tied = [slate[j] for j, c in enumerate(counts) if c == max_count]
    return min(tied, key=lambda b: b.to_tuple())


def _condition_c_cn_llm(
    slate: List[PolicyBundle],
    E: np.ndarray,
    lambda_i: float = 0.15,
    lambda_f: float = 0.03,
) -> PolicyBundle:
    """Community Notes matrix factorisation on LLM endorsement matrix."""
    i_n = fit_community_notes(E, lambda_i=lambda_i, lambda_f=lambda_f)
    max_score = float(i_n.max())
    tied = [slate[j] for j, s in enumerate(i_n) if abs(s - max_score) < 1e-9]
    return min(tied, key=lambda b: b.to_tuple())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
    client = anthropic.Anthropic(api_key=api_key)

    print(f"LLM-persona validation: {N_TRIALS} trials × {len(CONDITIONS)} conditions × {len(SPILLOVER_CONFIGS)} spillover configs")
    print(f"Model: claude-haiku-4-5-20251001, temp=0, cache: {CACHE_PATH}")

    llm_personas = get_personas()
    print(f"Loaded {len(llm_personas)} LLM personas")

    print(f"Generating {N_PERSONAS} parametric personas (seed={PERSONA_SEED}) for payoff table...")
    param_personas = generate_personas(n=N_PERSONAS, seed=PERSONA_SEED)

    print("Enumerating 5^8 = 390,625 bundles...")
    all_bundles = enumerate_bundles()

    cache = load_cache(CACHE_PATH)
    print(f"Cache loaded: {len(cache)} entries")

    total_api_calls = 0
    total_cache_hits = 0
    all_results: List[LLMTrialResult] = []

    for cfg in SPILLOVER_CONFIGS:
        print(f"\nBuilding oracle table (spillover_config={cfg!r})...")
        table = build_payoff_table(param_personas, cfg, bundles=all_bundles, cache=True)
        print(f"  G_oracle={table.g_oracle:.4f}  G_random={table.g_random:.4f}")

        for trial_idx in range(N_TRIALS):
            trial_seed = BASE_SEED + trial_idx
            slate = generate_candidate_slate(param_personas, trial_seed)

            E, n_calls, n_hits = build_llm_endorsement_matrix(
                llm_personas, slate, client, cache, CACHE_PATH
            )
            total_api_calls += n_calls
            total_cache_hits += n_hits
            total = n_calls + n_hits
            hit_rate = n_hits / total if total > 0 else 0.0

            trial_gaps: Dict[str, float] = {}
            for cond in CONDITIONS:
                if cond == "A":
                    winner = _condition_a_llm(slate, E)
                elif cond == "C_CN":
                    winner = _condition_c_cn_llm(slate, E)
                else:
                    raise ValueError(cond)

                g_achieved = table.g_values[_bundle_index(winner)]
                gap = table.gap_fraction(g_achieved)
                trial_gaps[cond] = gap

                all_results.append(LLMTrialResult(
                    spillover_config=cfg,
                    condition=cond,
                    trial=trial_idx,
                    gap_fraction=gap,
                    G_achieved=g_achieved,
                    G_random=table.g_random,
                    G_oracle=table.g_oracle,
                    n_llm_calls=n_calls,
                    cache_hit_rate=hit_rate,
                ))

            print(
                f"  [{cfg}] trial {trial_idx:2d}: "
                + "  ".join(f"{c}={trial_gaps[c]:.3f}" for c in CONDITIONS)
                + f"  api_calls={n_calls}  cache_hit={hit_rate:.0%}"
                + f"  total_api={total_api_calls}"
            )

    # Write CSV
    csv_path = RESULTS_DIR / "llm_cycle1_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "spillover_config", "condition", "trial", "gap_fraction",
            "G_achieved", "G_random", "G_oracle", "n_llm_calls", "cache_hit_rate",
        ])
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
                "n_llm_calls": r.n_llm_calls,
                "cache_hit_rate": f"{r.cache_hit_rate:.4f}",
            })
    print(f"\nWrote {len(all_results)} rows → {csv_path}")
    print(f"Total API calls: {total_api_calls}  cache hits: {total_cache_hits}  "
          f"overall hit rate: {total_cache_hits/(total_api_calls+total_cache_hits):.1%}")

    # Summary
    lines = _build_summary(all_results, total_api_calls, total_cache_hits)
    summary = "\n".join(lines)
    print("\n" + summary)

    txt_path = RESULTS_DIR / "llm_cycle1_summary.txt"
    txt_path.write_text(summary + "\n")
    print(f"\nWrote summary → {txt_path}")

    # Ensure final cache state is flushed
    save_cache(cache, CACHE_PATH)
    print(f"Cache saved: {len(cache)} entries → {CACHE_PATH}")


def _build_summary(
    results: List[LLMTrialResult],
    total_api_calls: int,
    total_cache_hits: int,
) -> List[str]:
    lines = [
        "LLM-Persona Validation: Cycle 1 Results",
        f"Model: claude-haiku-4-5-20251001 | N_personas=20 | N_trials={N_TRIALS} | "
        f"API calls={total_api_calls} | cache hits={total_cache_hits}",
        "",
        "Gap fraction: median (IQR) across 50 trials",
        "",
    ]

    # Per-condition per-spillover stats
    header = f"{'Condition':<10}" + "".join(f"  {cfg:<28}" for cfg in SPILLOVER_CONFIGS)
    lines.append(header)
    lines.append("-" * 75)

    for cond in CONDITIONS:
        row = f"{cond:<10}"
        for cfg in SPILLOVER_CONFIGS:
            vals = sorted(r.gap_fraction for r in results
                          if r.condition == cond and r.spillover_config == cfg)
            n = len(vals)
            med = statistics.median(vals)
            q1 = statistics.median(vals[: n // 2])
            q3 = statistics.median(vals[(n + 1) // 2:])
            row += f"  med={med:.3f} IQR=[{q1:.3f},{q3:.3f}]       "
        lines.append(row)

    lines.append("")
    lines.append("Comparison: LLM-driven vs parametric baseline (cycle5)")
    lines.append("")
    lines.append(
        f"{'':12} {'Spillover':12} {'Param (med)':>12} {'LLM (med)':>10} {'Delta':>8} {'Validation'}"
    )
    lines.append("-" * 70)

    all_validated = True
    for cond in CONDITIONS:
        for cfg in SPILLOVER_CONFIGS:
            param_med = PARAMETRIC_BASELINES[(cond, cfg)]
            vals = sorted(r.gap_fraction for r in results
                          if r.condition == cond and r.spillover_config == cfg)
            llm_med = statistics.median(vals)
            delta = llm_med - param_med
            within_15 = abs(delta) <= 0.15
            verdict = "within ±0.15" if within_15 else "DIVERGED (>0.15)"
            if not within_15:
                all_validated = False
            lines.append(
                f"  {cond:<10} {cfg:<12} {param_med:>12.3f} {llm_med:>10.3f} {delta:>+8.3f}  {verdict}"
            )

    lines.append("")

    # Direction check
    for cfg in SPILLOVER_CONFIGS:
        a_med = statistics.median(r.gap_fraction for r in results
                                  if r.condition == "A" and r.spillover_config == cfg)
        cn_med = statistics.median(r.gap_fraction for r in results
                                   if r.condition == "C_CN" and r.spillover_config == cfg)
        direction_ok = cn_med > a_med
        lines.append(
            f"  {cfg}: C_CN={cn_med:.3f}  A={a_med:.3f}  "
            f"delta={cn_med-a_med:+.3f}  "
            f"→ {'C_CN > A ✓' if direction_ok else 'DIRECTION FAILED ✗'}"
        )

    lines.append("")
    lines.append(
        "VERDICT: "
        + ("Direction-preserving validation confirmed" if all_validated
           else "Partial or diverged — see deltas above")
    )

    return lines


if __name__ == "__main__":
    main()
