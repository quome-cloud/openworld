# Market World-Model Experiment

**RESEARCH ONLY — NOT INVESTMENT ADVICE.**

This experiment applies the verified world-model methodology from the ARC-AGI-3 paper to stock market prediction. An LLM (Prism, A004) writes world models as Python code, we verify them with a perturbation gate, and evaluate final performance on a locked holdout split.

---

## What this experiment tests

Whether a language-model-generated market world model can pass three sequential filters:
1. Positive cost-adjusted Sharpe on held-out validation data
2. Failure to verify on label-shuffled (null) data (perturbation gate)
3. Positive cost-adjusted Sharpe on a completely untouched final holdout

The experiment's value is not "finding alpha" — it is demonstrating that the verification discipline correctly identifies overfit models before they reach final evaluation.

---

## Why this is more credible than typical AI-trading demos

### 1. Single-shot final holdout
The final holdout split (`2025-04-01` to `2026-06-30`) is **never read** during strategy development. A guard function in `build_splits.py` enforces this. The final holdout is evaluated exactly once, after all world model selection is complete. There is no "optimizing on test" — we commit predictions (`predictions.json`) before touching the holdout.

### 2. Perturbation gate
Inspired by ARC-AGI-3's verification step: if a world model still shows positive Sharpe when trained on **label-shuffled** (randomized) returns, it is fitting noise patterns in the feature space, not genuine structure. Such models are rejected even if their nominal validation Sharpe is high.

### 3. Multiple-testing ledger
Every world model variant tried is recorded in `MULTIPLE_TESTING_LEDGER.md` — no omissions. Reported Sharpe is deflated by `sqrt(1 / n_variants_tried)` to account for data mining bias. If you try 100 strategies and cherry-pick the best, the deflated Sharpe reflects that.

---

## Data

- **Universe**: SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, META, NVDA, BRK-B, JPM, JNJ, XOM (12 symbols)
- **Frequency**: Daily OHLCV
- **Source**: Yahoo Finance (via yfinance)

### Splits

| Split | Dates | Purpose |
|-------|-------|---------|
| TRAIN | 2010-01-01 to 2023-12-31 | World model fitting |
| VALIDATION | 2024-01-01 to 2025-03-31 | Strategy selection + perturbation gate |
| FINAL_HOLDOUT | 2025-04-01 to 2026-06-30 | Single-shot final evaluation |

Split checksums are stored in `data/split_checksums.json` and verified before any evaluation.

---

## Protocol

```
1. download_data.py   → data/raw_ohlcv.csv
2. build_splits.py    → data/{train,validation,final_holdout}.csv + checksums
3. Write world model  → world_models/wm_NN_name.py  (implements get_signals())
4. backtest.py        → run on VALIDATION split only
5. verify.py          → perturbation gate on VALIDATION split
6. MULTIPLE_TESTING_LEDGER.md → log every variant with deflated Sharpe
7. --- strategy selection complete ---
8. Unlock final holdout → evaluate ONCE
9. Compare to pre-committed predictions in predictions.json
```

### Exit criteria for validation (v2 gate — benchmark-adjusted)

A world model must meet **all three**:
- **Excess Sharpe > 0**: cost-adjusted Sharpe on validation must exceed buy-and-hold Sharpe (v2 correction)
- **E2a permutation p < 0.05**: excess Sharpe is not explained by drift-present null (50 shuffles)
- **E2b permutation p < 0.05**: excess Sharpe is not explained by zero-drift (demeaned) null

The naive v1 criterion (absolute Sharpe > 0) was identified as fail-open in bull markets: in 2024–2025 (SPY Sharpe ≈ 1.2), any "mostly long" strategy passes the absolute-Sharpe bar regardless of timing skill. The v2 gate scores *excess* over buy-and-hold, making the null meaningful in any regime.

---

## How to run

```bash
# Install dependencies (if not present)
pip install yfinance pandas numpy matplotlib

# Full pipeline
python run_experiment.py

# Or step by step:
python download_data.py
python build_splits.py
python backtest.py --world_model world_models/wm_01_example.py
python verify.py --world_model world_models/wm_01_example.py
```

---

## File map

```
researchy_market_worldmodel/
├── README.md                   # this file
├── MULTIPLE_TESTING_LEDGER.md  # running record of all variants evaluated
├── predictions.json            # pre-committed predictions (written before holdout eval)
├── download_data.py            # download OHLCV from Yahoo Finance
├── build_splits.py             # create + checksum train/val/holdout splits
├── verify.py                   # perturbation gate
├── backtest.py                 # backtesting engine (validation only)
├── run_experiment.py           # main orchestration script
├── data/
│   ├── raw_ohlcv.csv           # downloaded data (not committed if large)
│   ├── train.csv               # 2010–2023
│   ├── validation.csv          # 2024–2025-Q1
│   ├── final_holdout.csv       # 2025-Q2–2026-Q2 (LOCKED)
│   └── split_checksums.json    # sha256 of each split file
├── world_models/
│   ├── wm_00_placeholder.py    # interface stub
│   └── wm_NN_name.py           # each new world model
├── results/                    # output JSONs per world model
└── plots/                      # equity curves, survival funnel
```

---

**RESEARCH ONLY — NOT INVESTMENT ADVICE.**

Built by Prism (A004), researchy team, D.O.H.
