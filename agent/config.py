"""ChatConfig — schema for the LLM-inference-perf chat agent.

A single YAML drives a session. Only the agent's own LLM and session
bookkeeping live here; the tools are responsible for whatever parameters
they need, per call.

    agent:                       # the LLM that *drives* the agent
      model:             openai/Qwen3-Coder-30B-A3B-Instruct
      base_url:          http://localhost:8001/v1
      api_key:           EMPTY
      temperature:       0.0
      max_output_tokens: 4096     # cap on tokens GENERATED per response
      max_model_len:     32768    # server context window (= vLLM --max-model-len)
      reasoning:         false
      max_steps:         20       # tool-call cap per user turn

    session:
      dir:          runs/chat     # session working dir (trace, tool outputs)
      name:         null          # auto: chat-YYYYMMDD-HHMMSS if null

    system_prompt: |              # inline or system_prompt_file: <path>
      You are an LLM inference performance engineer ...
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Expand ${VAR} / $VAR references inside string fields when the value is wrapped
# in that exact form. Keeps secrets out of the file.
_ENVVAR_RE = re.compile(r"^\s*\$\{?(\w+)\}?\s*$")


def _expand_env(s: str | None) -> str | None:
    if not isinstance(s, str):
        return s
    m = _ENVVAR_RE.match(s)
    if not m:
        return s
    val = os.environ.get(m.group(1))
    if val is None:
        raise RuntimeError(f"env var {m.group(1)!r} referenced in config is not set")
    return val


def _expand_dict(d: dict) -> dict:
    return {k: _expand_env(v) if isinstance(v, str) else v for k, v in d.items()}


@dataclass
class ModelConfig:
    """An OpenAI-compatible endpoint — the agent's brain."""
    name: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float | None = 0.0
    # Cap on tokens GENERATED per response (sent to the API as max_tokens).
    max_output_tokens: int = 4096
    # The server's context window (vLLM --max-model-len). The loop keeps
    # prompt + max_output_tokens within this by trimming old tool results.
    max_model_len: int = 32768
    reasoning: bool = False
    timeout: float = 900.0
    max_retries: int = 2


@dataclass
class AgentSettings:
    model: ModelConfig
    # Hard cap on tool-call iterations per user turn. The loop returns the
    # final assistant reply once it stops calling tools.
    max_steps: int = 20


@dataclass
class SessionConfig:
    # Parent dir for session subdirs (working files, trace.json, etc).
    dir: str = "runs/chat"
    # If null, a chat-<timestamp> name is generated at startup.
    name: str | None = None


@dataclass
class ChatConfig:
    agent: AgentSettings
    session: SessionConfig = field(default_factory=SessionConfig)
    # Inline (preferred) or path. Inline wins when both are set.
    system_prompt: str | None = None
    system_prompt_file: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ChatConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}

        agent_raw = dict(raw.get("agent") or {})
        model_raw = _expand_dict(dict(agent_raw.pop("model", {}) or {}))
        if "name" not in model_raw:
            raise ValueError("config.agent.model.name is required")
        agent = AgentSettings(
            model=ModelConfig(**model_raw),
            **{k: v for k, v in agent_raw.items() if k in {"max_steps"}},
        )

        session = SessionConfig(**(raw.get("session") or {}))
        return cls(
            agent=agent,
            session=session,
            system_prompt=raw.get("system_prompt"),
            system_prompt_file=raw.get("system_prompt_file"),
        )

    def resolved_system_prompt(self) -> str | None:
        if self.system_prompt is not None:
            return self.system_prompt
        if self.system_prompt_file:
            return Path(self.system_prompt_file).read_text()
        return None
