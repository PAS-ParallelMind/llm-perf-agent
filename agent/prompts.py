"""System prompt assembly.

Task-specific guidance lives in ``agent.yaml`` (``system_prompt`` field).
This module only handles (a) a minimal default fallback, and (b)
appending the persistent memory index to whatever task prompt was given.
"""
from __future__ import annotations

_MEMORY_BLOCK = """
## Memory
Persistent notes live under ./memory. Use `remember` to save durable
facts and `recall` to re-read a file by name.

### MEMORY.md
{memory_index}
"""

_DEFAULT_TASK_PROMPT = (
    "You are a helpful engineering agent. Use the available tools to "
    "complete the user's request."
)


def build_system_prompt(
    memory_index: str,
    task_prompt: str | None = None,
) -> str:
    """Compose the final system prompt from a task prompt + memory index."""
    task = (task_prompt or _DEFAULT_TASK_PROMPT).rstrip()
    memory = _MEMORY_BLOCK.format(memory_index=memory_index.strip() or "(empty)")
    return f"{task}\n{memory}"
