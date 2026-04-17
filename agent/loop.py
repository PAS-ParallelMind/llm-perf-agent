"""Agentic tool-calling loop."""
from __future__ import annotations

import time
from typing import Any

from rich.console import Console

from . import submission
from .adapters.base import AgentResult, AgentTask
from .engine import Engine
from .memory import load_index
from .prompts import build_system_prompt
from .tools.base import dispatch, schemas

console = Console()


class Agent:
    def __init__(self, engine: Engine, max_steps: int = 20) -> None:
        self.engine = engine
        self.max_steps = max_steps
        self.step_count = 0
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(load_index())}
        ]
        self.tool_call_log: list[dict[str, Any]] = []

    def refresh_system(self) -> None:
        self.messages[0] = {
            "role": "system",
            "content": build_system_prompt(load_index()),
        }

    def run(
        self,
        task: AgentTask | str,
        time_budget_s: float | None = None,
    ) -> AgentResult:
        """Run the agent loop. Accepts an AgentTask or a plain string."""
        if isinstance(task, str):
            task = AgentTask(id="interactive", instruction=task)

        self.refresh_system()
        self.messages.append({"role": "user", "content": task.instruction})
        tool_schemas = schemas()
        start = time.monotonic()

        final_reply = ""

        for step in range(self.max_steps):
            self.step_count = step + 1

            if time_budget_s is not None and time.monotonic() - start > time_budget_s:
                final_reply = f"[time budget {time_budget_s}s exceeded after {step} steps]"
                break

            msg = self.engine.chat(self.messages, tools=tool_schemas)

            content = msg.content or ""
            if self.engine.reasoning:
                reasoning = getattr(msg, "reasoning", None) or getattr(
                    msg, "reasoning_content", None
                )
                if reasoning:
                    content = f"<thinking>\n{reasoning}\n</thinking>\n{content}"
            asst: dict[str, Any] = {"role": "assistant", "content": content}
            if getattr(msg, "tool_calls", None):
                asst["tool_calls"] = [
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
            self.messages.append(asst)

            if not getattr(msg, "tool_calls", None):
                final_reply = msg.content or ""
                break

            for tc in msg.tool_calls:
                name = tc.function.name
                args = tc.function.arguments
                console.print(f"[dim]\u00b7 tool[/] [cyan]{name}[/] {args}")

                tc_start = time.monotonic()
                result = dispatch(name, args)
                tc_elapsed = time.monotonic() - tc_start

                if len(result) > 8000:
                    result = result[:8000] + "\n... [truncated observation]"
                preview = "\n".join(result.splitlines()[:10])
                console.print(f"[dim]  \u21b3 {preview}[/]")

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": result,
                    }
                )
                self.tool_call_log.append({
                    "step": self.step_count,
                    "tool": name,
                    "arguments": args,
                    "result": result,
                    "elapsed_ms": round(tc_elapsed * 1000),
                })

            # submission short-circuits the loop
            if submission.get() is not None:
                final_reply = msg.content or "[submitted]"
                break
        else:
            final_reply = "[max steps reached without final answer]"

        elapsed = time.monotonic() - start
        return AgentResult(
            task_id=task.id,
            code=submission.get() or "",
            raw_reply=final_reply,
            trace=list(self.messages),
            tool_calls=list(self.tool_call_log),
            steps=self.step_count,
            elapsed_s=round(elapsed, 2),
            submitted=bool(submission.get()),
            metadata=dict(task.metadata),
        )
