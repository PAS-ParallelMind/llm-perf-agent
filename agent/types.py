"""Core dataclasses passed through the agent loop."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTask:
    """Input to the agent loop."""

    id: str                              # unique problem identifier
    instruction: str                     # natural-language prompt
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Output from the agent loop."""

    task_id: str
    code: str                            # submitted (or fallback-extracted) code
    raw_reply: str                       # agent's final text reply
    trace: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    elapsed_s: float = 0.0
    submitted: bool = False              # True if submit_solution was called
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
