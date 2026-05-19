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
    """Returns a canned sequence of assistant messages per user turn.

    Two-step trajectory: call the (placeholder) memory_estimate tool, then
    return a final text reply with no more tool calls — the loop treats
    that text as the assistant's answer to the user.
    """

    model = "fake-dry-run"
    reasoning = False

    def __init__(self) -> None:
        self._turn_counter = itertools.count()

    def _script(self, turn: int) -> list[SimpleNamespace]:
        return [
            _msg(
                "Let me sketch the memory footprint first.",
                [_call(f"c{turn}a", "memory_estimate",
                       {"spec": "{\"model\":\"demo-7B\",\"dtype\":\"fp16\"}"})],
            ),
            _msg(
                "[dry-run] FakeEngine reply: I called memory_estimate (which "
                "is a placeholder for now) and would interpret the result "
                "here. Replace FakeEngine with a real Engine to talk to a "
                "live vLLM server.",
                None,
            ),
        ]

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> SimpleNamespace:
        # Derive which step of the current user turn we're on from the
        # message history: count tool results since the last user message.
        step = 0
        for m in reversed(messages):
            if m.get("role") == "user":
                break
            if m.get("role") == "tool":
                step += 1
        turn = sum(1 for m in messages if m.get("role") == "user")
        script = self._script(turn)
        return script[min(step, len(script) - 1)]
