"""Content-addressable memory with semantic recall (stdlib only).

The brain's unconscious needs more than exact-key lookup: a paraphrased query
("capital city of France?") should still surface the right memory ("France ->
Paris"). `MemoryStore` does that with a character n-gram bag-of-grams vector and
cosine similarity -- deterministic, dependency-free, good enough to beat exact-match
on paraphrases without an embedding model. An optional `embed=` hook lets you swap
in a real embedding function.
"""

from __future__ import annotations

import heapq

import math
from typing import Any, Callable, List, Optional, Tuple


def _ngrams(text: str, n: int = 3) -> dict:
    s = "  " + str(text).lower().strip() + "  "
    counts: dict = {}
    for i in range(len(s) - n + 1):
        g = s[i:i + n]
        counts[g] = counts.get(g, 0) + 1
    return counts


def _cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(g, 0) for g, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class MemoryStore:
    """Add (cue, value) pairs; recall by semantic similarity or exact cue.

    Default similarity is char-trigram cosine (stdlib). Pass `embed=fn` (text ->
    sequence of floats) to use a real embedding model instead.
    """

    def __init__(self, n: int = 3, embed: Optional[Callable[[str], List[float]]] = None):
        self.n = n
        self._embed = embed
        self._items: List[Tuple[str, Any, Any]] = []   # (cue, value, vector)

    def add(self, cue: str, value: Any) -> None:
        self._items.append((cue, value, self._vec(cue)))

    def _vec(self, text: str):
        return list(self._embed(text)) if self._embed else _ngrams(text, self.n)

    def _sim(self, a, b) -> float:
        if self._embed:                                # dense cosine
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a)) or 1.0
            nb = math.sqrt(sum(y * y for y in b)) or 1.0
            return dot / (na * nb)
        return _cosine(a, b)

    def recall(self, query: str, k: int = 1) -> List[Tuple[str, Any, float]]:
        """Top-k (cue, value, score) by similarity to `query`."""
        qv = self._vec(query)
        scored = heapq.nlargest(
            k, ((self._sim(qv, v), cue, val) for cue, val, v in self._items),
            key=lambda t: t[0])
        return [(cue, val, round(s, 4)) for s, cue, val in scored]

    def exact(self, cue: str) -> Any:
        for c, val, _ in self._items:
            if c == cue:
                return val
        return None

    def __len__(self) -> int:
        return len(self._items)
