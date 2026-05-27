"""On-demand procedure playbooks ("skills").

A *skill* is a markdown file that encodes a multi-step workflow the
agent can pull into the conversation when the situation calls for it.
Unlike the system prompt (always loaded, always paid for), a skill is
loaded only when the model invokes it — keeping the always-on context
small while leaving room for specialised playbooks.

Layout: ``skills/<name>.md``, resolved from ``$AGENT_SKILLS_DIR``
(default ``skills/``). Each file can have YAML-style frontmatter::

    ---
    name: deployment_planning
    description: <one-line description shown in list_skills>
    when_to_use: <one-line trigger description>
    ---

    <free-form markdown body — the playbook the model follows>

The two tools:
* ``list_skills`` — catalog (name + description + when_to_use)
* ``invoke_skill(name)`` — returns the body as a tool result so the model
  sees the playbook in its next turn.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from .base import tool


SKILLS_DIR = Path(os.environ.get("AGENT_SKILLS_DIR", "skills")).resolve()

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


def _parse_skill(path: Path) -> dict:
    """Parse a skill file into ``{name, description, when_to_use, body}``."""
    text = path.read_text()
    meta = {"name": path.stem, "description": "", "when_to_use": ""}
    body = text
    m = _FRONTMATTER_RE.match(text)
    if m:
        fm, body = m.group(1), m.group(2)
        for line in fm.splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if k in meta:
                meta[k] = v
    meta["body"] = body.strip()
    meta["path"] = str(path)
    return meta


def _all_skills() -> list[dict]:
    if not SKILLS_DIR.is_dir():
        return []
    return [_parse_skill(p) for p in sorted(SKILLS_DIR.glob("*.md"))]


@tool(
    "List the procedure playbooks (skills) available to invoke. Each entry "
    "is a multi-step workflow you can pull into the conversation via "
    "`invoke_skill` when the situation matches its description. Call this "
    "first if you're unsure whether a playbook exists for what the user "
    "is asking."
)
def list_skills() -> str:
    skills = _all_skills()
    if not skills:
        return "(no skills available)"
    lines = [f"{len(skills)} skill(s) available:"]
    for s in skills:
        lines.append(f"- {s['name']} — {s['description']}")
        if s["when_to_use"]:
            lines.append(f"    when to use: {s['when_to_use']}")
    return "\n".join(lines)


@tool(
    "Load a procedure playbook (skill) into the conversation. Returns the "
    "skill's full markdown body, which you should then follow as your "
    "playbook for the rest of the task — its rules supersede the general "
    "system-prompt guidance where they conflict (e.g. it may sanction "
    "sweeping behaviour that the base prompt discourages).",
    name="Name of the skill, as listed by `list_skills` (file stem of the "
         "skill markdown).",
)
def invoke_skill(name: str) -> str:
    skills = _all_skills()
    for s in skills:
        if s["name"] == name:
            return s["body"]
    available = ", ".join(s["name"] for s in skills) or "(none)"
    return (f"ERROR: unknown skill {name!r}. Available: {available}. "
            f"Call `list_skills` to see descriptions.")
