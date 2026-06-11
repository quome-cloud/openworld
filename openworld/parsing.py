"""Robust extraction of code blocks and JSON objects from LLM output."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Return the contents of the first fenced code block, or the raw text."""
    match = _CODE_FENCE.search(text)
    return (match.group(1) if match else text).strip()


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Return the first balanced JSON object found in `text`, or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            char = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
