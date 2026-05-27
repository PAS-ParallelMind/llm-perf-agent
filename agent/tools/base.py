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


def _coerce_args(fn: Callable, args: dict) -> dict:
    """Best-effort coerce string args to int/float/bool when the function
    signature asks for them. Some models emit ``"offset": "200"`` instead of
    ``"offset": 200``; without coercion the call blows up on the first
    comparison."""
    sig = inspect.signature(fn)
    out: dict[str, Any] = {}
    for k, v in args.items():
        p = sig.parameters.get(k)
        if p is None or not isinstance(v, str):
            out[k] = v
            continue
        ann = p.annotation
        # `from __future__ import annotations` turns annotations into strings,
        # so check both forms.
        ann_name = ann if isinstance(ann, str) else getattr(ann, "__name__", "")
        try:
            if ann_name == "int":
                out[k] = int(v)
            elif ann_name == "float":
                out[k] = float(v)
            elif ann_name == "bool":
                out[k] = v.lower() in ("true", "1", "yes")
            else:
                out[k] = v
        except (ValueError, TypeError):
            out[k] = v
    return out


def dispatch(name: str, arguments: str | dict) -> str:
    if name not in TOOLS:
        # Surface the valid names so the model can self-correct without
        # waiting for the user to remind it which tools actually exist.
        return (f"ERROR: unknown tool {name!r}. "
                f"Available tools: {', '.join(sorted(TOOLS))}")
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
    fn = TOOLS[name]["fn"]
    args = _coerce_args(fn, args)
    try:
        result = fn(**args)
    except Exception as e:  # surface errors back to the model
        return f"ERROR: {type(e).__name__}: {e}"
    return result if isinstance(result, str) else json.dumps(result, default=str)
