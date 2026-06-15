"""Run a startup call transcript through the E51 growth world model — an investor
diagnostic.

Feed a pitch/diligence call transcript (e.g. a Google Meet chat export). An LLM
TextPerceptor extracts the growth factors (team, market, product-market fit,
grit, capital) the E51 model uses; the model then forward-simulates THIS startup
many times and reports: the factor read, the outcome distribution (survival,
median/p90 value), and the BINDING CONSTRAINT — which factor, if improved, would
move this startup's expected outcome most. That last part is the useful bit for an
investor: it turns a call into "what actually has to be true, and what's the
limiting factor."

  python datasets/openworld-startup/investor_diagnostic.py [path/to/transcript.txt]

Honest caveats: the factor extraction is an LLM read of one call (noisy,
gameable); E51 is a stylized model, not a return forecast. Use as a structured
lens, not a verdict.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from e51_startups import FACTORS, simulate, BURN  # noqa: E402
from openworld import Observation, OllamaLLM, TextPerceptor  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "sample_pitch_call.txt"
K = 600                          # Monte-Carlo trajectories for this one startup

EXTRACT_SYSTEM = (
    "You are an investment analyst. From the call transcript, score the startup on "
    "each factor from 0.0 (very weak) to 1.0 (exceptional), grounded in concrete "
    "evidence in the text. Reply ONLY as the requested fields. Be calibrated: 0.5 "
    "is average for a funded startup. Factors: team (founder fit + caliber), "
    "market (size/timing), pmf (product-market fit: retention, usage, pull), grit "
    "(focus, resilience, skin in the game), capital_runway_months (cash / burn)."
)


def extract(llm, text):
    fields = ["team", "market", "pmf", "grit", "capital_runway_months"]
    raw = TextPerceptor(llm, produces=fields).perceive(Observation("text", text, t=0))

    def num(x, default):
        try:
            v = float("".join(c for c in str(x) if c.isdigit() or c in ".-"))
        except ValueError:
            return default
        return v
    f = {k: min(1.0, max(0.0, num(raw.get(k), 0.5))) for k in fields[:4]}
    f["capital_runway_months"] = max(1.0, num(raw.get("capital_runway_months"), 8))
    return f


def project(f, capital_k):
    batch = {k: np.full(K, f[k]) for k in ("team", "market", "pmf", "grit")}
    batch["capital"] = np.full(K, capital_k)
    value, alive, _ = simulate(batch)
    return value, alive


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    text = path.read_text()
    llm = OllamaLLM(model="qwen2.5:7b", temperature=0.0, timeout=240,
                    options={"num_ctx": 8192})
    f = extract(llm, text)
    capital_k = f["capital_runway_months"] * BURN          # runway months -> $k cash

    value, alive = project(f, capital_k)
    survive = float(alive.mean())
    med = float(np.median(value)); p90 = float(np.quantile(value, 0.9))

    # binding constraint: which factor, lifted, moves THIS startup's median value most
    sens = {}
    base_med = med
    for fac in ("team", "market", "pmf", "grit"):
        g = dict(f); g[fac] = min(1.0, f[fac] + 0.2)
        v2, _ = project(g, capital_k)
        sens[fac] = float(np.median(v2)) - base_med
    g_cap = dict(f)
    v_cap, _ = project(g_cap, capital_k * 2)               # double the runway/cash
    sens["capital"] = float(np.median(v_cap)) - base_med
    binding = max(sens, key=sens.get)

    print(f"Investor diagnostic — {path.name}\n")
    print("  factor read (LLM from transcript):")
    for k in ("team", "market", "pmf", "grit"):
        print(f"    {k:<8} {f[k]:.2f}")
    print(f"    runway   {f['capital_runway_months']:.0f} months (~${capital_k:.0f}k)")
    print(f"\n  model projection ({K} runs): survival {survive:.0%}, "
          f"median value ${med:.0f}k, p90 ${p90:.0f}k")
    print(f"  binding constraint: {binding.upper()} "
          f"(+0.2 → +${sens[binding]:.0f}k median; full ranking: " +
          ", ".join(f"{k} +${sens[k]:.0f}k" for k in sorted(sens, key=sens.get, reverse=True))
          + ")")
    print("\n  >> Structured lens on one call, not a verdict. E51 is stylized; "
          "the factor read is a noisy LLM judgment. Not investment advice.")


if __name__ == "__main__":
    main()
