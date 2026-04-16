"""Normalize ParEval-style outputs so they concatenate cleanly with the prompt.

ParEval's driver does ``prompt + "\\n" + output`` and expects the prompt's
trailing ``{`` to open the function and the output to supply the body
(ending with the closing ``}``). LLM outputs often wrap in ``{ ... }`` or
re-emit the full signature — this script strips those.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_CODE_BLOCK = re.compile(r"```(?:cpp|c\+\+|c)?\s*\n?(.*?)```", re.S)


def extract_code(text: str) -> str:
    if not text:
        return ""
    m = _CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


def _signature_last_line(prompt: str) -> str:
    return prompt.rstrip().splitlines()[-1].strip()


def _signature_head(prompt: str) -> str:
    """Return the function signature without the trailing ``{``."""
    last = _signature_last_line(prompt).rstrip()
    if last.endswith("{"):
        last = last[:-1].rstrip()
    return last


def _canonicalize(s: str) -> str:
    """Collapse whitespace and normalise ``const T &`` ↔ ``T const&`` so that
    semantically-identical C++ signatures compare equal."""
    s = " ".join(s.split())                       # collapse whitespace
    # const T & → T const& (move const after type)
    s = re.sub(r'\bconst\s+(\w[\w:]*(?:\s*<[^>]*>)?)\s*&', r'\1 const&', s)
    # remove spaces around & and *
    s = re.sub(r'\s*([&*])\s*', r'\1', s)
    return s


def _strip_signature(body: str, prompt: str) -> str:
    """If body starts with (or contains) the function signature, cut up to and
    including the first ``{`` that follows it."""
    head = _signature_head(prompt)
    if not head:
        return body

    # Try exact match first
    idx = body.find(head)
    if idx != -1:
        brace = body.find("{", idx + len(head))
        if brace != -1:
            return body[brace + 1 :]

    # Fuzzy match: canonicalize both and scan for the signature
    canon_head = _canonicalize(head)
    # Try to find a line range in body that matches the canonical signature
    # by scanning for the function name and checking the surrounding text
    func_match = re.search(r'\b(\w+)\s*\(', head)
    if not func_match:
        return body
    func_name = func_match.group(1)

    # Find all occurrences of the function name in body
    for m in re.finditer(r'\b' + re.escape(func_name) + r'\s*\(', body):
        start = m.start()
        # Walk backwards to capture return type
        line_start = body.rfind('\n', 0, start)
        line_start = 0 if line_start == -1 else line_start + 1
        # Find the opening brace after this position
        brace = body.find("{", m.end())
        if brace == -1:
            continue
        candidate = body[line_start:brace].strip()
        if _canonicalize(candidate) == canon_head:
            return body[brace + 1 :]

    return body


def _strip_outer_braces(body: str) -> str:
    """If the body is wrapped in a single outer ``{ ... }`` scope, strip
    the outer opening brace (but keep the closing one — the function still
    needs to close)."""
    s = body.lstrip()
    if not s.startswith("{"):
        return body
    # Only strip if there's a matching close somewhere later. We keep the
    # closing ``}`` in place because the driver's prompt opens the function
    # with ``{`` and expects the output to close it.
    return s[1:]


def normalize(prompt: str, output: str) -> str:
    code = extract_code(output)
    code = _strip_signature(code, prompt)
    code = _strip_outer_braces(code)
    return code.rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    for entry in data:
        prompt = entry.get("prompt", "")
        outs = entry.get("outputs") or []
        entry["outputs"] = [normalize(prompt, o) for o in outs]
    Path(args.output).write_text(json.dumps(data, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
