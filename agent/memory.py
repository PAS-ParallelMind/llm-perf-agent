"""Markdown-file memory store (CLAUDE.md style).

- ``MEMORY.md`` is an always-loaded index of ``- [title](file.md) — hook``
  lines. Each entry points to an individual memory file in the same dir.
- Each memory file has YAML-ish frontmatter (name/description/type) and a
  free-form body.
- ``load_index()`` returns the concatenated index text for the system prompt.
- ``save(name, type_, description, body)`` writes a file and appends an
  index line if missing.
- Also exposes ``remember`` / ``recall`` as tools so the model can manage
  its own memory mid-conversation.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from .tools.base import tool

MEM_DIR = Path(os.environ.get("AGENT_MEMORY_DIR", "memory")).resolve()
INDEX = MEM_DIR / "MEMORY.md"


def _ensure() -> None:
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX.exists():
        INDEX.write_text("# Memory Index\n\n")


def load_index() -> str:
    """Return the full MEMORY.md for injection into the system prompt."""
    _ensure()
    return INDEX.read_text()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "memory"


@tool(
    "Save a memory entry (user/feedback/project/reference) to the persistent "
    "memory store. Use for facts that should survive across conversations.",
    name="Short title of the memory",
    type="One of: user, feedback, project, reference",
    description="One-line description used for relevance",
    body="Full memory body in markdown",
)
def remember(name: str, type: str, description: str, body: str) -> str:
    _ensure()
    fname = _slug(name) + ".md"
    fpath = MEM_DIR / fname
    fpath.write_text(
        f"---\nname: {name}\ndescription: {description}\ntype: {type}\n---\n\n{body}\n"
    )
    line = f"- [{name}]({fname}) — {description}\n"
    idx = INDEX.read_text()
    if fname not in idx:
        INDEX.write_text(idx.rstrip() + "\n" + line)
    return f"saved memory '{name}' to {fname}"


@tool(
    "Read the full content of a memory file by filename.",
    filename="Memory file name as listed in MEMORY.md",
)
def recall(filename: str) -> str:
    _ensure()
    fpath = MEM_DIR / filename
    if not fpath.exists():
        return f"ERROR: no such memory file: {filename}"
    return fpath.read_text()
