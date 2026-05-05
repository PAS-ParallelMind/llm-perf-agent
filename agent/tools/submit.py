"""Final-answer submission tool."""
from __future__ import annotations

from ..submission import set_code
from .base import tool


@tool(
    "Submit the final solution code for this problem. Call exactly ONCE "
    "when the solution has been built and verified. For single-function "
    "tasks, pass the final source code. For multi-file workspace tasks, pass "
    "a concise summary of the files changed; the evaluator reads the modified "
    "workspace files. Calling this tool terminates the agent loop.",
    code="Final source code, or a concise summary for multi-file workspace tasks",
)
def submit_solution(code: str) -> str:
    set_code(code)
    return f"submission accepted ({len(code)} chars)"
