"""Adapter base classes: unified task/result format + benchmark ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTask:
    """Benchmark-agnostic input to the agent."""

    id: str                          # unique problem identifier
    instruction: str                 # natural-language prompt for the agent
    metadata: dict[str, Any] = field(default_factory=dict)  # pass-through


@dataclass
class AgentResult:
    """Benchmark-agnostic output from the agent."""

    task_id: str
    code: str                        # submitted (or fallback-extracted) code
    raw_reply: str                   # agent's final text reply
    trace: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    elapsed_s: float = 0.0
    submitted: bool = False          # True if submit_solution was called
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(ABC):
    """Converts between a specific benchmark format and the agent's unified format."""

    @abstractmethod
    def load(self, path: str, **kwargs: Any) -> list[AgentTask]:
        """Read benchmark data and return a list of AgentTasks."""
        ...

    @abstractmethod
    def export(self, results: list[AgentResult], output_path: str) -> None:
        """Write agent results in the format the benchmark evaluator expects."""
        ...
