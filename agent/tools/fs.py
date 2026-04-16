"""Filesystem tools: read / write / edit / glob / grep. Sandboxed to the
workspace root (see agent/workspace.py)."""
from __future__ import annotations

import os
import re

from ..workspace import get_root, resolve
from .base import tool

MAX_READ = 200_000  # bytes


@tool(
    "Read a text file (relative to workspace) and return contents with "
    "1-indexed line numbers.",
    path="Path relative to the workspace root",
)
def read_file(path: str) -> str:
    fp = resolve(path)
    with open(fp, "r", errors="replace") as f:
        data = f.read(MAX_READ + 1)
    truncated = len(data) > MAX_READ
    data = data[:MAX_READ]
    lines = data.splitlines()
    numbered = "\n".join(f"{i + 1:6d}  {ln}" for i, ln in enumerate(lines))
    if truncated:
        numbered += "\n... [truncated]"
    return numbered or "(empty file)"


@tool(
    "Write (overwrite) a text file. Creates parent dirs if missing. "
    "Confined to the workspace root.",
    path="Target path relative to workspace root",
    content="Full file contents",
)
def write_file(path: str, content: str) -> str:
    fp = resolve(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)
    return f"wrote {len(content)} bytes to {fp.relative_to(get_root())}"


@tool(
    "Replace an exact string in a file. old_string must match exactly once.",
    path="File to edit (relative to workspace)",
    old_string="String to find",
    new_string="Replacement string",
)
def edit_file(path: str, old_string: str, new_string: str) -> str:
    fp = resolve(path)
    src = fp.read_text()
    count = src.count(old_string)
    if count == 0:
        return "ERROR: old_string not found"
    if count > 1:
        return f"ERROR: old_string matches {count} times, must be unique"
    fp.write_text(src.replace(old_string, new_string, 1))
    return f"edited {fp.relative_to(get_root())}"


@tool(
    "Glob files by pattern, rooted at the workspace (e.g. '**/*.cu').",
    pattern="Glob pattern (relative to workspace, supports **)",
)
def glob(pattern: str) -> str:
    root = get_root()
    if os.path.isabs(pattern):
        return "ERROR: glob pattern must be relative to workspace"
    hits = [str(p.relative_to(root)) for p in root.glob(pattern)]
    return "\n".join(hits[:200]) if hits else "(no matches)"


@tool(
    "Regex search across files under a workspace-relative path.",
    pattern="Python regex",
    path="Root directory relative to workspace (default '.')",
)
def grep(pattern: str, path: str = ".") -> str:
    rx = re.compile(pattern)
    base = resolve(path)
    root = get_root()
    out: list[str] = []
    for r, _, files in os.walk(base):
        if any(seg in r for seg in (".git", "__pycache__", "node_modules")):
            continue
        for fn in files:
            fp = os.path.join(r, fn)
            try:
                with open(fp, "r", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            rel = os.path.relpath(fp, root)
                            out.append(f"{rel}:{i}: {line.rstrip()}")
                            if len(out) >= 200:
                                return "\n".join(out) + "\n... [truncated]"
            except OSError:
                pass
    return "\n".join(out) if out else "(no matches)"
