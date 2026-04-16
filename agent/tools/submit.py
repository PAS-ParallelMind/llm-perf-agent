"""Final-answer submission tool."""
from __future__ import annotations

from ..submission import set_code
from .base import tool


@tool(
    "Submit the final solution code for this problem. Call exactly ONCE "
    "when the solution has been built and verified. The code string must "
    "contain ONLY the required function definition (no main, no extra "
    "includes — the evaluation harness provides those). Calling this tool "
    "terminates the agent loop for this problem.",
    code="The final function definition as a single code string",
)
def submit_solution(code: str) -> str:
    set_code(code)
    return f"submission accepted ({len(code)} chars)"
