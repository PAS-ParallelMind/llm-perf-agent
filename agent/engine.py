"""Thin wrapper around a vLLM OpenAI-compatible server."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

# Retry only on transient transport-level errors. Don't retry on
# 4xx (bad request, auth) since those won't change on a second try.
_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError)


class ContextLengthExceeded(Exception):
    """Raised when the server rejects a request for exceeding its context
    window. The loop catches this to trim old context and retry, rather
    than crashing the turn."""


def _is_context_length_error(exc: BadRequestError) -> bool:
    msg = str(getattr(exc, "message", "") or exc).lower()
    return ("context length" in msg or "context window" in msg
            or "max_model_len" in msg or "maximum context" in msg
            or "longer than the maximum" in msg)


@dataclass
class Engine:
    model: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    # Set to None to omit the param entirely (some Anthropic models
    # reject `temperature`).
    temperature: float | None = 0.2
    # Cap on tokens GENERATED per response (sent to the API as max_tokens).
    max_output_tokens: int = 2048
    # The server's context window (vLLM --max-model-len); used by the loop
    # to budget how much prompt it can send alongside max_output_tokens.
    max_model_len: int = 32768
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
        # Snapshot of the most recent request payload sent to the server,
        # for trace/debugging. Populated by chat().
        self.last_request: dict[str, Any] | None = None

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_output_tokens,
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Snapshot exactly what we send. messages is copied because the
        # loop keeps appending to the live list after this call returns.
        self.last_request = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
            "tools": list(tools or []),
            "params": {k: v for k, v in kwargs.items()
                       if k in ("temperature", "max_tokens", "tool_choice")},
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(**kwargs).choices[0].message
            except BadRequestError as exc:
                # Context-overflow is recoverable by the loop (trim + retry);
                # re-raise as our own type. Other 400s are real bugs — re-raise.
                if _is_context_length_error(exc):
                    raise ContextLengthExceeded(str(exc)) from exc
                raise
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                # Linear backoff: 5s, 15s. Keeps total worst-case bounded
                # at timeout*(retries+1) + sum(backoff).
                time.sleep(5 + 10 * attempt)
        assert last_exc is not None
        raise last_exc
