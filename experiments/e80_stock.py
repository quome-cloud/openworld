"""E80 (stock) - world-time compute STRESS TEST on real markets (honest, expect weak).

Each ticker is a WORLD; an example is a trading day described by recent technical features; the
label is the REAL realized next-day direction (up/down). Train on many ticker-worlds, test on
STRICTLY held-out tickers. Real data + ground-truth outcomes, but near-chance predictability --
so a FLAT result here is expected and fine; this domain probes whether the mechanism survives a
near-efficient real domain, it is never the lead claim.

Consumed by e80_common (build_worlds + CONFIG). Needs `yfinance` (installed on box).
"""

CONFIG = {
    "ladder": [2, 8, 16, 32, 60],
    "abl_n": 32,
    "abl_noise": [0.0, 0.15, 0.30, 0.45, 0.60, 0.80, 1.0],
    "cap": 120,            # trading days per ticker-world
    "n_test": 25,
    "seeds": [0, 1],
    "base": "Qwen/Qwen2.5-0.5B-Instruct",
}

# ~90 liquid large-caps -> plenty of ticker-worlds (download failures are skipped).
TICKERS = ("AAPL MSFT GOOGL AMZN META NVDA TSLA JPM V JNJ WMT PG MA HD CVX ABBV PFE KO PEP "
           "BAC COST MRK TMO ACN MCD CSCO ADBE LIN ABT DHR WFC TXN NEE PM UNP MS RTX HON QCOM "
           "LOW INTC AMD CAT GS IBM AMAT BA GE SBUX MMM CVS T VZ DIS NKE ORCL CRM INTU AMGN "
           "ISRG NOW BKNG ADP GILD MDT TJX REGN VRTX PANW LMT SPGI ELV CB MO DUK SO BDX CL "
           "EOG SLB APD ITW MMC AON PNC USB").split()


def _bucket(x, edges, labels):
    for e, lab in zip(edges, labels):
        if x < e:
            return lab
    return labels[-1]


def build_worlds():
    import yfinance as yf
    worlds = {}
    for tk in TICKERS:
        try:
            df = yf.download(tk, period="3y", interval="1d", progress=False, auto_adjust=True)
            if hasattr(df.columns, "get_level_values"):   # flatten yfinance MultiIndex columns
                df.columns = df.columns.get_level_values(0)
            close = [float(x) for x in df["Close"].dropna().tolist()]
            vol = [float(x) for x in df["Volume"].dropna().tolist()]
        except Exception as e:  # noqa: BLE001
            print(f"  [stock] skip {tk}: {repr(e)[:60]}", flush=True)
            continue
        if len(close) < 80:
            continue
        rets = [0.0] + [(close[i] / close[i - 1] - 1.0) for i in range(1, len(close))]
        rows = []
        for t in range(20, len(close) - 1):
            r1 = rets[t] * 100
            r5 = (close[t] / close[t - 5] - 1) * 100
            r10 = (close[t] / close[t - 10] - 1) * 100
            ma20 = sum(close[t - 20:t]) / 20
            v20 = sum(vol[t - 20:t]) / 20 if sum(vol[t - 20:t]) else 1
            sd10 = (sum((rets[t - k] * 100) ** 2 for k in range(10)) / 10) ** 0.5
            feats = (f"1-day return {r1:+.1f}%, 5-day {r5:+.1f}%, 10-day {r10:+.1f}%, "
                     f"10-day volatility {sd10:.1f}%, price {'above' if close[t] > ma20 else 'below'} "
                     f"its 20-day average, volume {vol[t] / v20:.1f}x its 20-day average")
            lab = "up" if rets[t + 1] >= 0 else "down"
            rows.append({"prompt": ("You are a market technician. Recent signals for a stock:\n"
                                    f"  {feats}\nWill the NEXT trading day close up or down? "
                                    "Reply with ONLY 'up' or 'down'."),
                         "label": lab})
        if len(rows) >= 60 and len({r["label"] for r in rows}) == 2:
            worlds[tk] = {"classes": ["up", "down"], "rows": rows}
            print(f"  [stock] world {tk}: {len(rows)} days", flush=True)
    return worlds


if __name__ == "__main__":
    w = build_worlds()
    print(f"stock worlds: {len(w)}")
