"""Per-problem submission slot (thread-safe).

The ``submit_solution`` tool writes the final code here; ``loop.Agent``
checks it after each step and terminates when set.
"""
from __future__ import annotations

import threading

_local = threading.local()


def reset() -> None:
    _local.current = None


def set_code(code: str) -> None:
    _local.current = code


def get() -> str | None:
    return getattr(_local, "current", None)
