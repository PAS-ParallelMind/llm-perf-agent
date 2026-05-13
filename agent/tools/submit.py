"""Final-answer submission tool."""
from __future__ import annotations

from ..submission import set_code
from .base import tool


@tool(
    "Submit the final solution. Call exactly ONCE when the solution has "
    "been built and verified. Provide your final source code in `code`. "
    "Optionally use `notes` to add an explanatory summary — e.g. which "
    "files you changed, what trade-offs you picked. Different benchmarks "
    "consume different fields: function-level / single-file evaluators "
    "compile `code`; workspace-level evaluators read the workspace files "
    "directly and may use `notes` for context. Calling this terminates "
    "the agent loop.",
    code="Final source code (or empty string if the evaluator reads "
         "workspace files directly).",
    notes="Optional explanatory text — change summary, trade-offs, etc. "
          "Default empty.",
)
def submit_solution(code: str, notes: str = "") -> str:
    set_code(code, notes)
    return (f"submission accepted (code={len(code)} chars"
            + (f", notes={len(notes)} chars" if notes else "") + ")")
