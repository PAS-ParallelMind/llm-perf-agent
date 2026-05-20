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

from .engine import ContextLengthExceeded, Engine
from .memory import load_index
from .prompts import build_system_prompt
from .tools.base import dispatch, schemas

console = Console()

_MAX_TOOL_CALLS_PER_TURN = 4
_MAX_IDENTICAL_TOOL_CALLS = 3
_TOOL_RESULT_TRUNCATE = 16_000

# Context budgeting. We don't have the server's tokenizer, so estimate
# tokens from characters. A *low* chars/token ratio under-estimates the
# budget (trims sooner) — safer against overflow. The reactive
# trim-and-retry below is the backstop when the estimate is off.
_CHARS_PER_TOKEN = 3.5
_CTX_RESERVE_TOKENS = 1024   # headroom kept below the window

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
        # Char budget for the prompt we may send: window minus the output
        # reservation minus headroom, converted to chars. Falls back to a
        # 32k-window assumption for engines that don't expose the fields.
        window = getattr(engine, "max_model_len", 32768)
        out_reserve = getattr(engine, "max_output_tokens", 2048)
        budget_tokens = max(window - out_reserve - _CTX_RESERVE_TOKENS, 1024)
        self._input_budget_chars = int(budget_tokens * _CHARS_PER_TOKEN)
        self.messages: list[dict[str, Any]] = [
            {"role": "system",
             "content": build_system_prompt(load_index(), self.system_prompt)}
        ]
        self.tool_call_log: list[dict[str, Any]] = []
        self.tool_call_counts: dict[str, int] = {}
        # Per-call snapshots of the exact request payload sent to the
        # engine (messages + tool schemas + sampling params), for debugging.
        self.llm_requests: list[dict[str, Any]] = []
        self.turn_count = 0

    def reset(self) -> None:
        """Drop conversation history and re-seed the system message."""
        self.messages = [
            {"role": "system",
             "content": build_system_prompt(load_index(), self.system_prompt)}
        ]
        self.tool_call_log = []
        self.tool_call_counts = {}
        self.llm_requests = []
        self.turn_count = 0

    def _refresh_system(self) -> None:
        self.messages[0] = {
            "role": "system",
            "content": build_system_prompt(load_index(), self.system_prompt),
        }

    def _context_chars(self, tool_schemas: list[dict[str, Any]]) -> int:
        """Rough size of what we'd send the model: messages + tool schemas."""
        return (len(json.dumps(self.messages, default=str))
                + len(json.dumps(tool_schemas, default=str)))

    def _trim_context(self, tool_schemas: list[dict[str, Any]],
                      aggressive: bool = False) -> int:
        """Elide the bodies of the oldest tool-result messages until the
        estimated context fits the input budget. Tool outputs are the
        bulk of the context and become stale once the model has acted on
        them, so they're trimmed oldest-first; the system prompt and the
        user/assistant thread are left intact. Returns the count elided."""
        budget = self._input_budget_chars // (2 if aggressive else 1)
        elided = 0
        for m in self.messages:
            if self._context_chars(tool_schemas) <= budget:
                break
            if m.get("role") != "tool":
                continue
            body = m.get("content", "")
            if body.startswith("[elided"):
                continue
            m["content"] = (f"[elided earlier {m.get('name', 'tool')} result "
                            f"({len(body)} chars) to fit context — re-run the "
                            f"tool if you need it again]")
            elided += 1
        return elided

    def _forced_synthesis(self, tool_schemas: list[dict[str, Any]],
                          step: int) -> str:
        """Wrap-up call made when the step budget is hit: ask the model for
        a final answer with tools disabled, so it must synthesize from what
        it already gathered rather than keep tool-calling. The reply is
        appended to the conversation; the 'stop' nudge is ephemeral."""
        console.print("[yellow]· step budget reached — requesting a final "
                      "summary with tools disabled[/]")
        if self._context_chars(tool_schemas) > self._input_budget_chars:
            self._trim_context(tool_schemas, aggressive=True)
        nudge = {
            "role": "user",
            "content": ("(Tool-call budget reached — do not call any more "
                        "tools. Based on the results so far, give your best "
                        "final answer now, and note anything you couldn't "
                        "finish.)"),
        }
        try:
            msg = self.engine.chat(self.messages + [nudge], tools=None)
        except Exception as e:  # don't let the wrap-up itself crash the turn
            return (f"[reached the {self.max_steps}-step tool budget; the "
                    f"final-summary call failed: {type(e).__name__}: {e}]")

        req = getattr(self.engine, "last_request", None)
        if req is not None:
            self.llm_requests.append({"turn": self.turn_count, "step": step,
                                      "elapsed_ms": 0, "synthesis": True, **req})

        reply = msg.content or ""
        if self.engine.reasoning and not reply:
            reply = (getattr(msg, "reasoning", None)
                     or getattr(msg, "reasoning_content", None) or "")
        self.messages.append({"role": "assistant", "content": reply})
        return reply or "[reached the tool-call budget; no final summary produced]"

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
            # Proactively trim before sending so we stay under the window.
            if self._context_chars(tool_schemas) > self._input_budget_chars:
                n = self._trim_context(tool_schemas)
                if n:
                    console.print(f"[dim]· context trim: elided {n} old tool "
                                  f"result(s) to fit the window[/]")

            llm_start = time.monotonic()
            try:
                msg = self.engine.chat(self.messages, tools=tool_schemas)
            except ContextLengthExceeded:
                # Estimate was off — trim hard and retry once.
                n = self._trim_context(tool_schemas, aggressive=True)
                console.print(f"[yellow]· context overflow: elided {n} old tool "
                              f"result(s), retrying[/]")
                msg = self.engine.chat(self.messages, tools=tool_schemas)
            llm_elapsed_ms = round((time.monotonic() - llm_start) * 1000)

            # Capture the exact request payload that was just sent, for the
            # trace / webui prompt-inspector.
            req = getattr(self.engine, "last_request", None)
            if req is not None:
                self.llm_requests.append({
                    "turn": self.turn_count,
                    "step": step + 1,
                    "elapsed_ms": llm_elapsed_ms,
                    **req,
                })
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
            # Step budget exhausted while still tool-calling. Force one
            # tools-disabled call so the model synthesizes an answer from
            # what it gathered, instead of returning a useless placeholder.
            truncated = True
            final_reply = self._forced_synthesis(tool_schemas, step + 1)

        return TurnResult(
            reply=final_reply,
            steps=step + 1,
            elapsed_s=round(time.monotonic() - start, 2),
            tool_calls=turn_tool_log,
            truncated=truncated,
        )
