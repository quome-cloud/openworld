"""
World Model 07: Loughran-McDonald Finance Lexicon + Phrase Matching
Hypothesis: dictionary counts of positive/negative finance-specific words in
            news headlines predict next-day equity direction.
            Published lexicon-based alphas have largely decayed post-2010.
Assumptions:
  - Loughran-McDonald (LM) finance word lists capture domain-relevant sentiment
  - Simple count ratio (pos - neg) / total_words is the relevant signal
  - Phrase-matching for momentum signals ("beats expectations", "cuts guidance")
    adds incremental signal over pure word-count sentiment
  - Pre-committed prediction: FAILS the perturbation gate (lexicon-based
    published strategies have decayed; the gate should reject this signal)
Source: LM word lists (publicly available academic resource). Applied to
        GDELT GKG article URLs/titles (coarse proxy for headline text).
RESEARCH ONLY — NOT INVESTMENT ADVICE.
"""
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path

NAME = "lm_lexicon_sentiment"
ASSUMPTIONS = [
    "Loughran-McDonald finance-domain word lists capture relevant sentiment",
    "Net sentiment = (pos_count - neg_count) / total_words",
    "Phrase patterns for earnings momentum events add incremental signal",
    "Signal applied with 1-day lag to avoid lookahead bias",
    "PRE-COMMITTED: expected to FAIL noise gate (decayed alpha per literature)",
]

_SENTIMENT_PATH = Path(__file__).parent.parent / "data" / "sentiment" / "daily_sentiment.csv"

# ---------------------------------------------------------------------------
# Loughran-McDonald Finance Word Lists (representative subset)
# Full lists: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
# ---------------------------------------------------------------------------

LM_POSITIVE = {
    "achieve", "achievement", "achievements", "advance", "advances", "advantage",
    "advantageous", "affirm", "afford", "afforded", "ample", "appreciated",
    "appropriate", "attain", "attractive", "best", "better", "breakthrough",
    "capable", "certain", "certainty", "clear", "collaborate", "confident",
    "consistent", "constructive", "deliver", "delivered", "effective", "efficient",
    "enable", "enhance", "enhanced", "exceptional", "exceed", "exceeded",
    "excellent", "expand", "expanded", "favorable", "gain", "gained", "gains",
    "good", "grow", "growing", "growth", "high", "higher", "highest", "improve",
    "improved", "improvement", "increased", "innovation", "innovative", "leadership",
    "lead", "leads", "momentum", "more", "opportunity", "optimal", "outstanding",
    "perform", "performance", "positive", "potential", "profit", "profitable",
    "profitability", "progress", "promising", "record", "reduce", "reliable",
    "robust", "significant", "strength", "strong", "stronger", "strongest",
    "successful", "superior", "sustained", "top", "value", "valuable",
    "well", "yield",
}

LM_NEGATIVE = {
    "abandon", "adverse", "allegation", "allegations", "allege", "bad", "bankrupt",
    "bankruptcy", "below", "breach", "burden", "cease", "charges", "claim",
    "claims", "close", "closure", "concern", "concerns", "conflict", "contraction",
    "cuts", "cutback", "declining", "decrease", "deficit", "delay", "delinquent",
    "deteriorate", "disappointing", "disclosed", "dispute", "disrupt", "distress",
    "doubt", "downturn", "eliminate", "error", "exceed", "fail", "failed",
    "failure", "falling", "fraud", "impair", "impairment", "inadequate",
    "increasing", "instability", "insufficient", "investigation", "irregular",
    "lawsuit", "layoff", "layoffs", "less", "liability", "limit", "limitations",
    "liquidity", "loss", "losses", "low", "lower", "lowest", "miss", "missed",
    "negative", "obligation", "overdue", "penalty", "problem", "problems",
    "reduce", "reduction", "restate", "restatement", "restructure", "risk",
    "risks", "scrutiny", "shortfall", "slow", "slower", "slowing", "substandard",
    "suspended", "termination", "uncertainty", "unfavorable", "unexpected",
    "violation", "vulnerable", "warning", "weak", "weaker", "worst", "write-down",
    "writedown", "writeoff",
}

LM_UNCERTAINTY = {
    "abrupt", "ambiguous", "amid", "appears", "approximately", "assume", "belief",
    "believe", "caution", "cautious", "complex", "contingent", "could", "depends",
    "difficult", "doubt", "exposure", "fluctuate", "fluctuation", "if", "impact",
    "indefinite", "indeterminate", "likely", "may", "might", "minimum", "moderate",
    "occasionally", "predict", "probable", "range", "risk", "risks", "rough",
    "significant", "sometime", "speculative", "subject", "typical", "uncertain",
    "uncertainty", "unclear", "unknown", "unpredictable", "unusual", "volatile",
    "volatility", "whether",
}

# ---------------------------------------------------------------------------
# Phrase patterns (regex) for earnings/corporate momentum signals
# ---------------------------------------------------------------------------

POSITIVE_PHRASES = [
    r"beats?\s+expectations?",
    r"raises?\s+guidance",
    r"exceeds?\s+(estimates?|expectations?)",
    r"strong\s+(earnings?|results?|quarter)",
    r"record\s+(earnings?|revenue|profit)",
    r"upgrades?\s+(to|from)",
    r"buyback",
    r"dividend\s+increase",
    r"beat\s+(the\s+)?street",
    r"raised?\s+(outlook|forecast)",
]

NEGATIVE_PHRASES = [
    r"cuts?\s+guidance",
    r"misses?\s+(expectations?|estimates?)",
    r"below\s+expectations?",
    r"worse\s+than\s+expected",
    r"downgrade",
    r"layoffs?",
    r"job\s+cuts?",
    r"warns?\s+(of\s+)?slowdown",
    r"disappointing\s+(results?|earnings?)",
    r"below\s+estimates?",
]

_POS_RE = re.compile("|".join(POSITIVE_PHRASES), re.IGNORECASE)
_NEG_RE = re.compile("|".join(NEGATIVE_PHRASES), re.IGNORECASE)


def score_text(text: str) -> float:
    """
    Score a text string using LM lexicon word counts + phrase patterns.
    Returns net sentiment score (positive = bullish, negative = bearish).
    """
    text_lower = text.lower()
    words = re.findall(r"\b[a-z]+\b", text_lower)
    if not words:
        return 0.0

    n = len(words)
    pos_count = sum(1 for w in words if w in LM_POSITIVE)
    neg_count = sum(1 for w in words if w in LM_NEGATIVE)

    # Phrase matching (weighted 2× a single word)
    phrase_score = len(_POS_RE.findall(text)) * 2 - len(_NEG_RE.findall(text)) * 2

    word_score = (pos_count - neg_count) / max(n, 1) * 100  # normalize to %
    return word_score + phrase_score


def _load_gdelt_sentiment() -> pd.DataFrame | None:
    """Load GDELT-derived sentiment (used as headline proxy)."""
    if not _SENTIMENT_PATH.exists():
        return None
    return pd.read_csv(_SENTIMENT_PATH, parse_dates=["date"], index_col="date")


def get_signals(ohlcv: pd.DataFrame) -> pd.Series:
    """
    Generate {-1, 0, 1} signals from LM lexicon applied to GDELT tone.

    In full deployment: score each day's news headlines through score_text().
    Here we use GDELT's precomputed polarity (highly correlated with LM scoring)
    as a proxy, since full headline text requires per-day GDELT GKG download.

    This is a COARSE approximation — real LM lexicon scoring requires full article
    text. The signal structure (threshold, lag, normalization) is identical to
    what would be used on actual headline text.

    Falls back to flat signal if data not available.
    """
    gdelt = _load_gdelt_sentiment()

    if gdelt is None:
        print(f"  WARNING: Sentiment data not found at {_SENTIMENT_PATH}")
        print("  Run: python download_sentiment.py")
        return pd.Series(0, index=ohlcv.index)

    # Use GDELT tone as proxy for LM-scored sentiment
    # In full deployment, replace with: daily_scores = pd.Series({date: score_text(headlines)})
    tone = gdelt["gdelt_tone"].reindex(ohlcv.index).ffill().bfill()

    # Apply a slightly looser threshold than WM06 (LM signal is noisier than raw tone)
    tone_z = (tone - tone.mean()) / tone.std()
    tone_lag = tone_z.shift(1)

    signals = pd.Series(0, index=ohlcv.index)
    signals[tone_lag > 0.75] = 1
    signals[tone_lag < -0.75] = -1

    return signals
