#!/usr/bin/env python3
"""Run the full BALROG Baba Is AI suite and save results."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import warnings
warnings.filterwarnings('ignore')

from balrog_solver import run_suite

if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    summary = run_suite(output_dir=output_dir, verbose=True)
    print('\n=== FINAL SUMMARY ===')
    print(f"Score: {summary['score']:.1%}")
    print(f"Solved: {summary['total_solved']}/{summary['total_episodes']}")
    print(f"SOTA:   75.7% (Gemini-3.1-Pro-Thinking)")
    print(f"Delta:  {summary['delta_vs_sota']:+.1%}")
