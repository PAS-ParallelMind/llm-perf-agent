"""Tool registry: minimal decorator + OpenAI-schema export + dispatch."""
from __future__ import annotations

import inspect
import json
from typing import Any, Callable

TOOLS: dict[str, dict[str, Any]] = {}


_PY2JSON = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def tool(description: str, /, **param_desc: str) -> Callable:
    """Register a function as a tool.

    Annotations are used to generate a JSON schema. Each parameter can
    optionally be described via ``param_desc``.
    """

    def deco(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        props: dict[str, Any] = {}
        required: list[str] = []
        for name, p in sig.parameters.items():
            ann = p.annotation if p.annotation is not inspect._empty else str
            type_name = getattr(ann, "__name__", "str")
            props[name] = {
                "type": _PY2JSON.get(type_name, "string"),
                "description": param_desc.get(name, ""),
            }
            if p.default is inspect._empty:
                required.append(name)
        TOOLS[fn.__name__] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                },
            },
        }
        return fn

    return deco


def schemas() -> list[dict[str, Any]]:
    return [t["schema"] for t in TOOLS.values()]


def dispatch(name: str, arguments: str | dict) -> str:
    if name not in TOOLS:
        return f"ERROR: unknown tool {name!r}"
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as e:
            return (
                f"ERROR: malformed tool arguments JSON: {e}. "
                f"Arguments received (truncated): {arguments[:200]!r}"
            )
    else:
        args = arguments or {}
    try:
        result = TOOLS[name]["fn"](**args)
    except Exception as e:  # surface errors back to the model
        return f"ERROR: {type(e).__name__}: {e}"
    return result if isinstance(result, str) else json.dumps(result, default=str)
