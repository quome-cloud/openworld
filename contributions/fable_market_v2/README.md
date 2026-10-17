# E-market v2 — fable_market_v2

RESEARCH ONLY. Paper trading and analysis exclusively. No real capital, no
brokerage interaction, no trade recommendations. Nothing in this directory is
investment advice. (Compliance standing rule; see T385.)

Second iteration of the market world-model experiment: cross-sectional S&P 500
panel, structural feature library + transparent ridge model layer, DEV vs
POST-CUTOFF contamination split (synthesis model cutoff: Jan 2026), permutation
gates, drop-one ablations, and a hash-committed prospective prediction ledger.

Read `FABLE_MARKET_V2_REPORT.md`. Reproduce:

```
python3 code/fetch_data.py        # cache OHLCV (skips if cached)
python3 code/fetch_retry.py       # recover rate-limit casualties
python3 code/fetch_earnings.py    # cache earnings dates + surprises
python3 code/tests/test_features.py
python3 code/run_eval.py --stage features
python3 code/run_eval.py --stage dev     # DEV sanity (pre-cutoff, unscored)
# predictions must be registered (results/predictions_registered.json) before:
python3 code/run_eval.py --stage post    # scored post-cutoff gate
python3 code/make_report.py dictionary|dev|post
python3 code/seed_ledger.py       # prospective ledger seed
```

Data caches (`data/`) make everything after the three fetch scripts re-runnable
offline.
