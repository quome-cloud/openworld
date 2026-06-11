"""Agents-as-a-judge: an LLM that selects among candidate behaviors and
scores trajectories against a rubric.

Two uses:
- Judge.choose(options, context): pick the best of N candidate actions,
  patches, or plans. Pair with an agent that samples several proposals to
  get judge-guided action selection.
- Judge.score_trajectory(trajectory, rubric): grade a whole episode 0-10
  against a written rubric, returning a float usable as an evaluation metric
  or wrapped in an Objective.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from .llm import BaseLLM
from .parsing import extract_json
from .simulation import Trajectory

CHOOSE_SYSTEM = (
    "You are an expert judge. You are shown a context and a numbered list of "
    "candidate options. Choose the single best option. Reply with ONLY a JSON "
    'object: {"choice": <number>, "reason": "<short>"}.'
)

SCORE_SYSTEM = (
    "You are an expert judge grading the outcome of a simulated episode "
    "against a rubric. Reply with ONLY a JSON object: "
    '{"score": <number between 0 and 10>, "reason": "<short>"}.'
)


class Judge:
    """An LLM-backed judge for behavior selection and trajectory grading."""

    def __init__(self, llm: BaseLLM, criteria: str = "", name: str = "judge"):
        self.llm = llm
        self.criteria = criteria
        self.name = name

    def choose(
        self,
        options: Sequence[Any],
        context: str = "",
        default: int = 0,
    ) -> int:
        """Return the index of the best option (defaults to `default` when the
        judge's reply cannot be parsed or is out of range)."""
        if len(options) == 1:
            return 0
        numbered = "\n\n".join(
            f"--- Option {i} ---\n{option}" for i, option in enumerate(options)
        )
        prompt = (
            (f"Judging criteria: {self.criteria}\n\n" if self.criteria else "")
            + (f"Context:\n{context}\n\n" if context else "")
            + f"Candidates:\n{numbered}\n\nChoose the best option."
        )
        reply = self.llm.ask(prompt, system=CHOOSE_SYSTEM)
        parsed = extract_json(reply)
        choice = parsed.get("choice") if parsed else None
        if choice is None:
            # Fall back to the first bare integer in the reply.
            match = re.search(r"\b(\d+)\b", reply)
            choice = int(match.group(1)) if match else None
        if isinstance(choice, (int, float)) and 0 <= int(choice) < len(options):
            return int(choice)
        return default

    def score_trajectory(
        self,
        trajectory: Trajectory,
        rubric: str,
        max_steps_shown: int = 20,
    ) -> Optional[float]:
        """Grade an episode 0-10 against `rubric`. Returns None when the
        judge's reply cannot be parsed."""
        shown = trajectory.steps[:max_steps_shown]
        lines = [
            f"step {r.step}: agent={r.agent} action={r.action.name}"
            f"({r.action.params}) -> state={dict(r.state)}"
            for r in shown
        ]
        if len(trajectory.steps) > max_steps_shown:
            lines.append(f"... ({len(trajectory.steps) - max_steps_shown} more steps)")
        prompt = (
            f"Rubric: {rubric}\n\n"
            f"Initial state: {dict(trajectory.initial_state)}\n"
            f"Episode:\n" + "\n".join(lines) + "\n"
            f"Final state: {dict(trajectory.final_state)}\n\n"
            "Grade this episode against the rubric."
        )
        reply = self.llm.ask(prompt, system=SCORE_SYSTEM)
        parsed = extract_json(reply)
        if parsed and isinstance(parsed.get("score"), (int, float)):
            return max(0.0, min(10.0, float(parsed["score"])))
        return None
