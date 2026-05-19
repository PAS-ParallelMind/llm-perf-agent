"""Shared dataclasses for the chat agent.

Most state lives on the :class:`ChatAgent` instance directly. This module
holds dataclasses that need to be importable from multiple places (e.g.
session metadata exported to disk).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionMeta:
    """Top-level info about an interactive chat session — written to
    ``<session_dir>/session.json`` after each turn for inspection."""
    name: str
    started_at: str                                # ISO-8601 UTC
    agent_model: str
    turns: int = 0
    total_steps: int = 0
    total_elapsed_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
