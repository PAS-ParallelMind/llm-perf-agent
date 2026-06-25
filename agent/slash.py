"""Shared slash-command dispatch for both the CLI REPL and the webui.

Slash commands let the user pre-load deterministic state into the
conversation (skill body via ``/plan``) or trigger
maintenance actions (``/reset``, ``/tools``, ``/help``) without going
through the model. Both front-ends route through :func:`dispatch_slash`
so the semantics stay identical.

``/exit`` is REPL-only and handled separately (the webui ignores it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loop import ChatAgent


HELP_TEXT = (
    "- `/reset` — clear conversation history\n"
    "- `/tools` — list registered tools\n"
    "- `/plan` — load the deployment_planning playbook for the next turn\n"
    "- `/help` — show this message"
)


@dataclass
class SlashResult:
    """Outcome of a slash-command dispatch.

    * ``response`` — text to surface to the user (None if there's
      nothing to show, e.g. ``/exit``).
    """
    response: str | None


def _stage_planning_skill(agent: "ChatAgent") -> str | None:
    """Append a synthetic ``invoke_skill(deployment_planning)`` tool-call
    + result pair into the conversation so the next real agent turn sees
    the playbook already loaded — no LLM routing decision needed.

    Returns ``None`` on success, or the error string if the underlying
    dispatch failed (so the caller can surface it as the slash response).
    """
    from .tools.base import dispatch

    body = dispatch("invoke_skill", '{"name": "deployment_planning"}')
    if body.startswith("ERROR"):
        return body

    tc_id = f"slash_plan_{agent.turn_count}"
    agent.messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": tc_id,
            "type": "function",
            "function": {
                "name": "invoke_skill",
                "arguments": '{"name": "deployment_planning"}',
            },
        }],
    })
    agent.messages.append({
        "role": "tool",
        "tool_call_id": tc_id,
        "name": "invoke_skill",
        "content": body,
    })
    return None


def dispatch_slash(cmd: str, agent: "ChatAgent") -> SlashResult | None:
    """Try to interpret ``cmd`` as a slash command.

    Returns ``None`` if it isn't a slash command (so the caller routes
    it to ``agent.chat`` as a normal user message). Returns a
    :class:`SlashResult` otherwise; the slash turn is logged into
    ``agent.messages`` as a (user, assistant) pair so front-ends that
    re-render from the message list (e.g. the webui) see it.

    ``/exit`` / ``/quit`` are REPL-only and not handled here — the REPL
    intercepts them before calling this function.
    """
    if not cmd.startswith("/"):
        return None
    cmd = cmd.strip()

    if cmd == "/help":
        response = HELP_TEXT
    elif cmd == "/reset":
        agent.reset()
        # After reset(), agent.messages is back to just the system prompt;
        # the (user, assistant) pair we append below becomes the only
        # subsequent turn so the user sees a confirmation in an otherwise
        # empty chat.
        response = "conversation reset."
    elif cmd == "/tools":
        from .tools import TOOLS
        response = "\n".join(f"- `{n}`" for n in sorted(TOOLS)) or "(none)"
    elif cmd == "/plan":
        err = _stage_planning_skill(agent)
        if err is not None:
            response = err
        else:
            response = (
                "planning mode loaded. The deployment_planning playbook is "
                "now in the conversation context — describe what you want "
                "to deploy and the agent will start at Stage 1."
            )
    else:
        # Unknown slash command — bubble up so the caller can either
        # show the hint or route it to the model unchanged.
        response = f"unknown command: {cmd}  (try /help)"

    # Log the slash turn as a normal (user, assistant) exchange so
    # message-list-driven front-ends (webui) render it like any other turn.
    # /plan's tool-call+result pair has already been appended above; we
    # insert the user message before them so the chronology stays right.
    if cmd == "/plan" and not response.startswith("ERROR") and len(agent.messages) >= 2:
        insert_at = len(agent.messages) - 2
        agent.messages.insert(insert_at, {"role": "user", "content": cmd})
    else:
        agent.messages.append({"role": "user", "content": cmd})
    agent.messages.append({"role": "assistant", "content": response})

    return SlashResult(response=response)
