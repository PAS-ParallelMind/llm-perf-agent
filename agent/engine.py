"""Thin wrapper around a vLLM OpenAI-compatible server."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

# Retry only on transient transport-level errors. Don't retry on
# 4xx (bad request, auth) since those won't change on a second try.
_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError)


@dataclass
class Engine:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    # Set to None to omit the param entirely (some Anthropic models
    # reject `temperature`).
    temperature: float | None = 0.2
    max_tokens: int = 2048
    reasoning: bool = False
    # Per-request HTTP timeout (sec). vLLM under heavy concurrency can
    # stall a single request well past the openai-python default 600s,
    # which would kill an otherwise-fine run.
    timeout: float = 900.0
    # Total retry attempts on transient errors (timeouts, conn drops).
    max_retries: int = 2

    def __post_init__(self) -> None:
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(**kwargs).choices[0].message
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                # Linear backoff: 5s, 15s. Keeps total worst-case bounded
                # at timeout*(retries+1) + sum(backoff).
                time.sleep(5 + 10 * attempt)
        assert last_exc is not None
        raise last_exc
