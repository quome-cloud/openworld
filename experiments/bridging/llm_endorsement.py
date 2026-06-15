"""Cache-aware LLM endorsement for the bridging LLM validation (T355).

Each (persona, bundle) pair is resolved via:
  1. Cache lookup by sha256(persona_id + "|" + canonical_bundle_str)
  2. On miss: one Haiku API call; result written to cache immediately

Bundle → human-readable format uses the stance label mapping from the design doc.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .llm_personas import LLMPersona
from .personas import ISSUES
from .policy import PolicyBundle

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 80
TEMPERATURE = 0.0

_STANCE_LABELS = {
    -2: "strongly progressive",
    -1: "lean progressive",
     0: "centrist compromise",
     1: "lean conservative",
     2: "strongly conservative",
}

_ISSUE_DISPLAY = {
    "immigration":       "Immigration",
    "healthcare":        "Healthcare",
    "climate":           "Climate",
    "fiscal":            "Fiscal",
    "foreign_policy":    "Foreign policy",
    "civil_rights":      "Civil rights",
    "education":         "Education",
    "criminal_justice":  "Criminal justice",
}


def _bundle_user_message(bundle: PolicyBundle) -> str:
    lines = ["Policy bundle:"]
    for issue in ISSUES:
        stance = bundle.stances[issue]
        lines.append(f"- {_ISSUE_DISPLAY[issue]}: {_STANCE_LABELS[stance]}")
    return "\n".join(lines)


def _cache_key(persona_id: str, bundle: PolicyBundle) -> str:
    canonical = str(bundle.to_tuple())
    raw = f"{persona_id}|{canonical}"
    return hashlib.sha256(raw.encode()).hexdigest()


def load_cache(cache_path: Path) -> Dict[str, Dict]:
    if cache_path.exists():
        with open(cache_path) as fh:
            return json.load(fh)
    return {}


def save_cache(cache: Dict[str, Dict], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(cache, fh, indent=2)


def endorse(
    persona: LLMPersona,
    bundle: PolicyBundle,
    client,
    cache: Dict[str, Dict],
    cache_path: Optional[Path] = None,
) -> Tuple[bool, bool]:
    """Return (endorsed: bool, cache_hit: bool).

    Checks cache first. On miss, calls the API, parses output, updates cache,
    and writes cache to disk if cache_path is provided.

    Conservative default on parse failure: NOT-ENDORSE (False).
    """
    key = _cache_key(persona.persona_id, bundle)

    if key in cache:
        entry = cache[key]
        return entry["label"] == "ENDORSE", True

    user_msg = _bundle_user_message(bundle)
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=persona.system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw_text = response.content[0].text.strip()
    except Exception as exc:
        logger.warning("API error for %s: %s — defaulting to NOT-ENDORSE", persona.persona_id, exc)
        raw_text = "NOT-ENDORSE\n(api error)"

    lines = raw_text.split("\n", 1)
    first_line = lines[0].strip().upper()
    reason = lines[1].strip() if len(lines) > 1 else ""

    if first_line == "ENDORSE":
        label = "ENDORSE"
    elif first_line == "NOT-ENDORSE":
        label = "NOT-ENDORSE"
    else:
        logger.warning(
            "Unexpected output from %s: %r — defaulting to NOT-ENDORSE",
            persona.persona_id, first_line,
        )
        label = "NOT-ENDORSE"

    cache[key] = {"label": label, "reason": reason}
    if cache_path is not None:
        save_cache(cache, cache_path)

    return label == "ENDORSE", False


def build_llm_endorsement_matrix(
    personas: List[LLMPersona],
    slate: List[PolicyBundle],
    client,
    cache: Dict[str, Dict],
    cache_path: Optional[Path] = None,
) -> Tuple[np.ndarray, int, int]:
    """Build (N_personas × N_bundles) binary endorsement matrix via LLM calls.

    Returns (E, n_api_calls, n_cache_hits).
    E[i, j] = 1.0 if persona i endorses bundle j, else 0.0.
    """
    N, M = len(personas), len(slate)
    E = np.zeros((N, M), dtype=np.float64)
    n_api_calls = 0
    n_cache_hits = 0

    for i, persona in enumerate(personas):
        for j, bundle in enumerate(slate):
            endorsed, hit = endorse(persona, bundle, client, cache, cache_path)
            E[i, j] = 1.0 if endorsed else 0.0
            if hit:
                n_cache_hits += 1
            else:
                n_api_calls += 1

    return E, n_api_calls, n_cache_hits
