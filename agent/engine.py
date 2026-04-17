"""Thin wrapper around a vLLM OpenAI-compatible server."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class Engine:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.2
    max_tokens: int = 2048
    reasoning: bool = False

    def __post_init__(self) -> None:
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs).choices[0].message
