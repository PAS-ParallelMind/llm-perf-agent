"""A fake Engine for --dry-run: scripts a tiny tool-calling trajectory so
you can exercise the loop without a live vLLM server."""
from __future__ import annotations

import itertools
import json
from types import SimpleNamespace
from typing import Any


def _msg(content: str = "", tool_calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _call(cid: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=cid,
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


class FakeEngine:
    """Returns a canned sequence of assistant messages per user turn."""

    model = "fake-dry-run"

    def __init__(self) -> None:
        self._turn_counter = itertools.count()

    def _script(self, turn: int) -> list[SimpleNamespace]:
        # a 3-step trajectory: glob -> read a memory file -> final answer
        return [
            _msg(
                "Let me look around first.",
                [_call(f"c{turn}a", "glob", {"pattern": "agent/**/*.py"})],
            ),
            _msg(
                "I'll save a note to memory.",
                [
                    _call(
                        f"c{turn}b",
                        "remember",
                        {
                            "name": "dry run demo",
                            "type": "project",
                            "description": "Proof that the dry-run loop executes tool calls",
                            "body": "Triggered via FakeEngine in --dry-run mode.",
                        },
                    )
                ],
            ),
            _msg(
                "[dry-run] Loop works: globbed files, wrote a memory entry, "
                "and returning a final answer with no further tool calls.",
                None,
            ),
        ]

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> SimpleNamespace:
        # derive which step of the current user turn we're on from the
        # message history: count tool results since the last user message
        step = 0
        for m in reversed(messages):
            if m.get("role") == "user":
                break
            if m.get("role") == "tool":
                step += 1
        turn = sum(1 for m in messages if m.get("role") == "user")
        script = self._script(turn)
        return script[min(step, len(script) - 1)]
