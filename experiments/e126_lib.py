"""Robust helpers for the E126 local-coder full-game pipeline (shared by qwen/ornith/etc runs).

Extracted so the failure modes have unit tests. The bugs these fix (observed on the ornith:35b run):
  1. history budget was a hard-coded 24000 chars, UNRELATED to num_ctx -> the prompt overflowed an
     8192-token window, Ollama truncated the FRONT (the task instructions), and the model returned
     EMPTY ~40% of rounds. `history_budget(num_ctx, ...)` ties the budget to the context window with a
     response reserve, so the task instructions always survive.
  2. `extract_code` fed prose back as "code" when the model emitted no fenced block; now it returns ""
     for non-code so the round is flagged and retried instead of running garbage.
"""
from __future__ import annotations
import re

# Conservative: code/JSON tokenizes denser than prose. Lower chars/token => smaller budget => safer.
CHARS_PER_TOKEN = 3.2


def history_budget(num_ctx: int, task_chars: int = 1500,
                   response_reserve_tokens: int = 1400, hard_cap: int = 60000) -> int:
    """Char budget for the rolling history so that task_template + history + a response all fit in
    num_ctx. Scales with the context window; never returns negative; capped so a huge num_ctx does not
    build a multi-hundred-KB prompt."""
    ctx_chars = int(num_ctx * CHARS_PER_TOKEN)
    reserve_chars = int(response_reserve_tokens * CHARS_PER_TOKEN)
    budget = ctx_chars - reserve_chars - task_chars
    return max(0, min(budget, hard_cap))


def build_history(log, budget: int) -> str:
    """Rolling transcript of prior rounds (code + stdout), newest kept first, trimmed to `budget`
    chars so the full prompt fits num_ctx. Always keeps at least the most recent round."""
    if not log:
        return "(none yet -- start by exploring)"
    parts = []
    for rec in log:
        code = rec["script"][:1800]
        out = rec["stdout"][-1800:]
        parts.append(
            f"===== round {rec['round']} (best_levels after this round = {rec.get('best_after', 0)}) =====\n"
            f"--- YOUR SCRIPT ---\n{code}\n--- ITS STDOUT ---\n{out}"
        )
    kept, total = [], 0
    for p in reversed(parts):            # most recent first
        if kept and total + len(p) > budget:
            break
        kept.append(p)
        total += len(p)
    return "\n\n".join(reversed(kept))


_CODE_SIGNALS = ("import ", "def ", "g.step", "g.reset", "Game(", "for ", "while ", "print(", "np.")


def extract_code(text: str) -> str:
    """Return runnable code from a model response. Prefer a fenced ```python block. If there is no
    fence, return the text ONLY when it actually looks like code (has code signals); otherwise return
    "" so the caller treats the round as empty and retries -- never feed prose to the interpreter."""
    if not text:
        return ""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    stripped = text.strip()
    if stripped and any(sig in stripped for sig in _CODE_SIGNALS):
        return stripped
    return ""
