"""Per-problem submission slot (thread-safe).

The ``submit_solution`` tool writes the final ``code`` (and optional
``notes``) here; ``loop.Agent`` checks it after each step and terminates
when set. Both fields are returned in ``AgentResult``; downstream
evaluators decide which to consume:

* ParEval / ParallelMind eval read ``code`` and compile it.
* CUDAMicroBench-style benchmarks read the workspace files and may use
  ``notes`` as a human-readable change summary.
"""
from __future__ import annotations

import threading

_local = threading.local()


def reset() -> None:
    _local.current = None
    _local.notes = None


def set_code(code: str, notes: str | None = None) -> None:
    _local.current = code
    _local.notes = notes or None


def get() -> str | None:
    return getattr(_local, "current", None)


def get_notes() -> str | None:
    return getattr(_local, "notes", None)
