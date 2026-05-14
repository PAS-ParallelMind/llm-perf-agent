"""Shell execution tool. Runs with cwd pinned to the workspace root."""
from __future__ import annotations

import subprocess

from ..cuda import cuda_env as _cuda_env
from ..workspace import get_root
from .base import tool

MAX_OUT = 20_000


@tool(
    "Execute a shell command inside the workspace root and return combined "
    "stdout/stderr.",
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
            env=_cuda_env(),
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    out = (p.stdout or "") + (p.stderr or "")
    if len(out) > MAX_OUT:
        out = out[:MAX_OUT] + "\n... [truncated]"
    return f"[exit={p.returncode}]\n{out}"
