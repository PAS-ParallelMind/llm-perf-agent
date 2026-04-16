"""Workspace sandbox: all file tools are confined to a single root directory.

Paths passed to tools may be relative (resolved against the root) or
absolute (must already be inside the root). Anything that resolves outside
the root — via ``..``, symlinks, or an absolute prefix — raises
``PermissionError``. Bash / build / run tools also use the root as cwd.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

# Default workspace lives inside the repo: <project_root>/workspace
# (project_root = parent of the `agent/` package dir).
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "workspace"

_LOCAL = threading.local()


def set_root(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    _LOCAL.root = p
    return p


def get_root() -> Path:
    root = getattr(_LOCAL, "root", None)
    if root is None:
        root = set_root(os.environ.get("AGENT_WORKSPACE") or DEFAULT_ROOT)
    return root


def resolve(path: str) -> Path:
    """Resolve ``path`` to an absolute Path inside the workspace root.

    Raises ``PermissionError`` if the target escapes the root.
    """
    root = get_root()
    raw = Path(path)
    candidate = raw if raw.is_absolute() else (root / raw)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as e:
        raise PermissionError(
            f"path escapes workspace {root}: {path!r}"
        ) from e
    return resolved
