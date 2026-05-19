"""Chat-mode tool-calling loop.

One user turn = up to ``max_steps`` LLM calls. The loop dispatches tool
calls each iteration and ends the turn when the model returns an
assistant message with no tool_calls — that text is the reply to the
user. Messages persist across turns so the conversation is multi-turn.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from .engine import Engine
from .memory import load_index
from .prompts import build_system_prompt
from .tools.base import dispatch, schemas

console = Console()

_MAX_TOOL_CALLS_PER_TURN = 4
_MAX_IDENTICAL_TOOL_CALLS = 3
_TOOL_RESULT_TRUNCATE = 16_000

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _tool_calls_from_text(content: str) -> list[dict[str, Any]]:
    """Fallback for models that print tool-call JSON instead of returning
    OpenAI ``tool_calls``. Keeps smaller / local models usable when the
    server-side tool parser misses a call."""
    candidates = _JSON_BLOCK_RE.findall(content) or [content]
    calls: list[dict[str, Any]] = []
    for raw in candidates:
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            args = item.get("arguments", {})
            fn = item.get("function")
            if isinstance(fn, dict):
                name = fn.get("name", name)
                args = fn.get("arguments", args)
            if isinstance(name, str) and name:
                calls.append({
                    "id": f"text_tool_{len(calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args if isinstance(args, str) else json.dumps(args),
                    },
                })
    return calls


def _normalize_tool_calls(msg: Any, content: str) -> list[dict[str, Any]]:
    if getattr(msg, "tool_calls", None):
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return _tool_calls_from_text(content)


def _tool_signature(tc: dict[str, Any]) -> str:
    return json.dumps({
        "name": tc["function"]["name"],
        "arguments": tc["function"]["arguments"],
    }, sort_keys=True)


def _dedupe_and_cap(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        sig = _tool_signature(tc)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(tc)
        if len(out) >= _MAX_TOOL_CALLS_PER_TURN:
            break
    return out


@dataclass
class TurnResult:
    """Result of one user turn — what the loop returned and how it got there."""
    reply: str                                     # final assistant text
    steps: int                                     # tool-call iterations consumed
    elapsed_s: float
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False                        # hit max_steps without yielding text


class ChatAgent:
    """Multi-turn chat agent with tool-calling.

    Holds a persistent ``messages`` list across calls to :meth:`chat`. The
    system prompt is refreshed at the start of each turn so memory edits
    show up immediately. ``tool_call_log`` accumulates every dispatch for
    the whole session, for trace export.
    """

    def __init__(
        self,
        engine: Engine,
        max_steps: int = 20,
        system_prompt: str | None = None,
    ) -> None:
        self.engine = engine
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.messages: list[dict[str, Any]] = [
            {"role": "system",
             "content": build_system_prompt(load_index(), self.system_prompt)}
        ]
        self.tool_call_log: list[dict[str, Any]] = []
        self.tool_call_counts: dict[str, int] = {}
        self.turn_count = 0

    def reset(self) -> None:
        """Drop conversation history and re-seed the system message."""
        self.messages = [
            {"role": "system",
             "content": build_system_prompt(load_index(), self.system_prompt)}
        ]
        self.tool_call_log = []
        self.tool_call_counts = {}
        self.turn_count = 0

    def _refresh_system(self) -> None:
        self.messages[0] = {
            "role": "system",
            "content": build_system_prompt(load_index(), self.system_prompt),
        }

    def chat(self, user_message: str) -> TurnResult:
        """Run one user turn and return the assistant's final reply."""
        self._refresh_system()
        self.messages.append({"role": "user", "content": user_message})
        self.turn_count += 1
        tool_schemas = schemas()
        start = time.monotonic()
        turn_tool_log: list[dict[str, Any]] = []
        truncated = False
        final_reply = ""

        for step in range(self.max_steps):
            llm_start = time.monotonic()
            msg = self.engine.chat(self.messages, tools=tool_schemas)
            llm_elapsed_ms = round((time.monotonic() - llm_start) * 1000)
            llm_log = {
                "turn": self.turn_count,
                "step": step + 1,
                "tool": "<llm>",
                "arguments": "",
                "result": "",
                "elapsed_ms": llm_elapsed_ms,
            }
            turn_tool_log.append(llm_log)
            self.tool_call_log.append(llm_log)

            content = msg.content or ""
            if self.engine.reasoning:
                reasoning = (getattr(msg, "reasoning", None)
                             or getattr(msg, "reasoning_content", None))
                if reasoning:
                    content = f"{reasoning}\n{content}" if content else reasoning

            tool_calls = _dedupe_and_cap(_normalize_tool_calls(msg, content))
            asst: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                asst["tool_calls"] = tool_calls
            self.messages.append(asst)

            # No tool calls = this is the assistant's reply to the user.
            if not tool_calls:
                final_reply = content
                break

            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                console.print(f"[dim]· tool[/] [cyan]{name}[/] {args}")

                tc_start = time.monotonic()
                sig = _tool_signature(tc)
                self.tool_call_counts[sig] = self.tool_call_counts.get(sig, 0) + 1
                if self.tool_call_counts[sig] > _MAX_IDENTICAL_TOOL_CALLS:
                    result = (
                        "ERROR: identical tool call repeated too many times. "
                        "Inspect the previous observation and change the "
                        "arguments; do not retry the same failing call."
                    )
                else:
                    result = dispatch(name, args)
                tc_elapsed = time.monotonic() - tc_start

                if len(result) > _TOOL_RESULT_TRUNCATE:
                    result = result[:_TOOL_RESULT_TRUNCATE] + "\n... [truncated observation]"

                steps_left = self.max_steps - (step + 1)
                result += f"\n\n[turn {self.turn_count} · step {step + 1}/{self.max_steps} ({steps_left} left)]"
                preview = "\n".join(result.splitlines()[:10])
                console.print(f"[dim]  ↳ {preview}[/]")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": result,
                })
                entry = {
                    "turn": self.turn_count,
                    "step": step + 1,
                    "tool": name,
                    "arguments": args,
                    "result": result,
                    "elapsed_ms": round(tc_elapsed * 1000),
                }
                turn_tool_log.append(entry)
                self.tool_call_log.append(entry)
        else:
            truncated = True
            final_reply = (
                f"[max_steps={self.max_steps} reached for turn {self.turn_count} "
                "without a final reply]"
            )

        return TurnResult(
            reply=final_reply,
            steps=step + 1,
            elapsed_s=round(time.monotonic() - start, 2),
            tool_calls=turn_tool_log,
            truncated=truncated,
        )
