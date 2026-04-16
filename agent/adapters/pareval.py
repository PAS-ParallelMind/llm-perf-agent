"""ParEval benchmark adapter.

- load():   reads generation-prompts.json → list[AgentTask]
- export(): list[AgentResult] → JSON accepted by ParEval drivers/run-all.py
            (includes output normalization)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .base import AgentResult, AgentTask, BenchmarkAdapter

# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------

_USER_TEMPLATE = """\
Solve the following ParEval benchmark problem.

Name: {name}
Parallelism model: {parallelism_model}
Problem type: {problem_type}

## Starter code (the signature you must implement)
```cpp
{prompt}
```
"""


def _build_instruction(entry: dict[str, Any]) -> str:
    return _USER_TEMPLATE.format(
        name=entry["name"],
        parallelism_model=entry["parallelism_model"],
        problem_type=entry["problem_type"],
        prompt=entry["prompt"],
    )


# ---------------------------------------------------------------------------
# Export / normalize helpers  (ported from scripts/normalize_outputs.py)
# ---------------------------------------------------------------------------

_CODE_BLOCK = re.compile(r"```(?:cpp|c\+\+|c)?\s*\n?(.*?)```", re.S)


def _extract_code(text: str) -> str:
    if not text:
        return ""
    m = _CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


def _signature_head(prompt: str) -> str:
    """Return the last line of the prompt (the function signature) without
    the trailing ``{``."""
    last = prompt.rstrip().splitlines()[-1].strip()
    if last.endswith("{"):
        last = last[:-1].rstrip()
    return last


def _canonicalize(s: str) -> str:
    """Collapse whitespace and normalise ``const T &`` <-> ``T const&``."""
    s = " ".join(s.split())
    s = re.sub(r'\bconst\s+(\w[\w:]*(?:\s*<[^>]*>)?)\s*&', r'\1 const&', s)
    s = re.sub(r'\s*([&*])\s*', r'\1', s)
    return s


def _strip_signature(body: str, prompt: str) -> str:
    head = _signature_head(prompt)
    if not head:
        return body

    # exact match
    idx = body.find(head)
    if idx != -1:
        brace = body.find("{", idx + len(head))
        if brace != -1:
            return body[brace + 1:]

    # fuzzy match via canonicalization
    canon_head = _canonicalize(head)
    func_match = re.search(r'\b(\w+)\s*\(', head)
    if not func_match:
        return body
    func_name = func_match.group(1)

    for m in re.finditer(r'\b' + re.escape(func_name) + r'\s*\(', body):
        line_start = body.rfind('\n', 0, m.start())
        line_start = 0 if line_start == -1 else line_start + 1
        brace = body.find("{", m.end())
        if brace == -1:
            continue
        candidate = body[line_start:brace].strip()
        if _canonicalize(candidate) == canon_head:
            return body[brace + 1:]

    return body


def _strip_outer_braces(body: str) -> str:
    s = body.lstrip()
    if not s.startswith("{"):
        return body
    return s[1:]


def normalize(prompt: str, output: str) -> str:
    """Normalize agent output so ``prompt + '\\n' + result`` is valid C++."""
    code = _extract_code(output)
    code = _strip_signature(code, prompt)
    code = _strip_outer_braces(code)
    return code.rstrip() + "\n"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ParevalAdapter(BenchmarkAdapter):
    """Adapter for the ParEval parallel-code benchmark."""

    def load(
        self,
        path: str,
        *,
        parallelism: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[AgentTask]:
        data = json.loads(Path(path).read_text())

        if parallelism:
            data = [e for e in data if e.get("parallelism_model") == parallelism]
        if limit:
            data = data[:limit]

        tasks: list[AgentTask] = []
        for entry in data:
            tasks.append(AgentTask(
                id=entry["name"],
                instruction=_build_instruction(entry),
                metadata=entry,   # keep all original fields for export
            ))
        return tasks

    def export(self, results: list[AgentResult], output_path: str) -> None:
        """Write results in the JSON format that ``drivers/run-all.py`` expects.

        Each entry keeps the original ParEval fields and gains an ``outputs``
        list (length 1) with the normalized code.
        """
        out: list[dict[str, Any]] = []
        for r in results:
            entry = dict(r.metadata)  # original ParEval fields
            prompt = entry.get("prompt", "")
            normalized = normalize(prompt, r.code) if r.code else ""
            entry["outputs"] = [normalized]
            # extra agent metadata (not used by run-all.py, but useful)
            entry["agent_steps"] = r.steps
            entry["agent_elapsed_s"] = r.elapsed_s
            entry["agent_submitted"] = r.submitted
            if r.error:
                entry["agent_error"] = r.error
            out.append(entry)

        Path(output_path).write_text(json.dumps(out, indent=2))
