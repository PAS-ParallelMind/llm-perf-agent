"""Shell execution tool. Runs with cwd pinned to the session workspace."""
from __future__ import annotations

import subprocess

from ..workspace import get_root
from .base import tool

MAX_OUT = 20_000


@tool(
    "Execute a shell command inside the session workspace and return "
    "combined stdout/stderr. Useful for nvidia-smi, curl, and other "
    "host-level probes.",
    command="Shell command to run",
    timeout="Timeout in seconds (default 120)",
)
def bash(command: str, timeout: int = 120) -> str:
    try:
        p = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(get_root()),
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    out = (p.stdout or "") + (p.stderr or "")
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + "\n... [truncated]"
    return f"[exit={p.returncode}]\n{out}"
